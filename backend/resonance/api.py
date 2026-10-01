"""HTTP API for Resonance Lab.

Route signatures here ARE the contract (see docs/CONTRACTS.md). The frontend
generates types from the OpenAPI document this app produces (`make types`).

Sessions live in an in-memory registry while the server runs and are persisted to
SQLite as event logs (resonance/store.py); unknown ids are looked up in the store and
reloaded as replay sessions. The backend never auto-advances: the frontend calls `step`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from resonance import schemas as S
from resonance.conditions import with_description
from resonance.config import PROJECT_ROOT, load_config
from resonance.env.timing_task import TimingTask
from resonance.experiments import run_experiment as run_experiment_batch
from resonance.experiments import to_csv
from resonance.music.features import symbolic_features
from resonance.music.motifs import base_phrases
from resonance.music.synth import audio_features, wav_bytes
from resonance.music.transforms import apply_transform
from resonance.presets import PRESETS, preset_session_config, run_preset_sessions
from resonance.session import Session, now_iso
from resonance.store import Store

VERSION = "0.1.0"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
T = TypeVar("T")


class RunPresetRequest(BaseModel):
    seed: int | None = None
    # Optional, additive (WS1): run with a different config / episode count (e.g. quick tests).
    config: S.ExperimentConfig | None = None
    episodes: int | None = None


class TransformRequest(BaseModel):
    phrase: S.Phrase
    transform: S.Perturbation


def model_status(config: S.ModelConfig) -> S.ModelStatus:
    """Status from the WS3 model layer if importable; otherwise provider 'none'."""
    try:
        from resonance.models.registry import status  # type: ignore[import-not-found]

        return status(config)
    except Exception as exc:  # absent or failing model layer
        return S.ModelStatus(
            provider="none",
            model_id="",
            available=False,
            tested_in_this_environment=False,
            accepts_audio=False,
            input_modality="none",
            detail=f"model layer unavailable ({type(exc).__name__}); local policy only",
        )


class SPAStaticFiles(StaticFiles):
    """Static frontend with SPA fallback: unknown non-/api paths serve index.html."""

    async def get_response(self, path: str, scope: Any) -> Any:  # noqa: ANN401
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and not path.startswith("api"):
                return await super().get_response("index.html", scope)
            raise
        if response.status_code == 404 and not path.startswith("api"):
            return await super().get_response("index.html", scope)
        return response


def _guard(fn: Callable[[], T]) -> T:
    """Map simulation errors to HTTP codes: KeyError 404, ValueError 400, RuntimeError 409."""
    try:
        return fn()
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\"")) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def create_app(
    db_path: str | Path | None = None, serve_frontend: bool = True, frontend_dir: str | Path | None = None
) -> FastAPI:
    app = FastAPI(title="Resonance Lab API", version=VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    store = Store(db_path)
    registry: dict[str, Session] = {}
    lock = threading.RLock()
    app.state.store = store
    app.state.registry = registry

    def get_session(session_id: str) -> Session:
        with lock:
            if session_id in registry:
                return registry[session_id]
            record = store.load_session_record(session_id)
            if record is None:
                raise HTTPException(status_code=404, detail=f"session {session_id} not found")
            summary, config, events = record
            session = Session.from_events(
                config, summary.condition, summary.seed, events,
                session_id=summary.id, preset=summary.preset, created_at=summary.created_at,
            )
            registry[session_id] = session
            return session

    def register(session: Session) -> None:
        with lock:
            registry[session.id] = session
            events = session.recorded_events() if session.mode == "replay" else session.events
            store.save_session(session.summary(), session.initial_config, events)

    def persist_new(session: Session, events: list[S.Event]) -> None:
        if session.mode != "replay" and events:
            store.append_events(session.summary(), events)

    @app.get("/api/health", response_model=S.HealthResponse)
    def health() -> S.HealthResponse:
        return S.HealthResponse(status="ok", version=VERSION, model=model_status(load_config().model))

    @app.get("/api/config/default", response_model=S.ExperimentConfig)
    def default_config() -> S.ExperimentConfig:
        return load_config()

    @app.get("/api/patterns", response_model=list[S.TimingPattern])
    def patterns() -> list[S.TimingPattern]:
        task = load_config().task
        return TimingTask(task.n_patterns, task.partial_credit).patterns()

    @app.get("/api/motifs", response_model=list[S.Phrase])
    def motifs() -> list[S.Phrase]:
        music = load_config().music
        return base_phrases(music.motif_bank, n=music.n_motifs)

    @app.get("/api/presets", response_model=list[S.PresetInfo])
    def presets() -> list[S.PresetInfo]:
        return list(PRESETS.values())

    @app.post("/api/presets/{name}/run", response_model=S.PresetResult)
    def run_preset(name: str, req: RunPresetRequest) -> S.PresetResult:
        if name not in PRESETS:
            raise HTTPException(status_code=404, detail=f"unknown preset {name}")
        seed = req.seed if req.seed is not None else (req.config.seed if req.config else load_config().seed)
        episodes = req.episodes or (req.config.episodes if req.config else None)
        result, sessions = _guard(lambda: run_preset_sessions(name, seed, req.config, episodes))
        for s in sessions:
            register(s)
        return result

    @app.post("/api/sessions", response_model=S.SessionSnapshot)
    def create_session(req: S.CreateSessionRequest) -> S.SessionSnapshot:
        def build() -> Session:
            if req.preset:
                episodes = req.config.episodes if req.config else None
                config, condition, script = preset_session_config(req.preset, req.config, episodes)
                return Session(config, condition, config.seed, preset=req.preset, script=script)
            config = req.config or load_config()
            return Session(config, with_description(req.condition), config.seed)

        session = _guard(build)
        register(session)
        return session.snapshot()

    @app.get("/api/sessions", response_model=list[S.SessionSummary])
    def list_sessions() -> list[S.SessionSummary]:
        with lock:
            live = {sid: s.summary() for sid, s in registry.items()}
        # Stored sessions not in memory reopen as replay sessions; list them as such.
        stored = [s.model_copy(update={"mode": "replay"}) for s in store.list_sessions() if s.id not in live]
        return sorted([*live.values(), *stored], key=lambda s: s.created_at, reverse=True)

    @app.get("/api/sessions/{session_id}", response_model=S.SessionSnapshot)
    def get_session_route(session_id: str) -> S.SessionSnapshot:
        return get_session(session_id).snapshot()

    @app.post("/api/sessions/{session_id}/step", response_model=S.StepResult)
    def step(session_id: str, req: S.StepRequest) -> S.StepResult:
        session = get_session(session_id)
        with lock:
            events = _guard(lambda: session.step(req.n))
            persist_new(session, events)
            return S.StepResult(events=events, snapshot=session.snapshot())

    @app.post("/api/sessions/{session_id}/reset", response_model=S.SessionSnapshot)
    def reset(session_id: str) -> S.SessionSnapshot:
        session = get_session(session_id)
        if session.mode == "replay":
            fresh = Session.from_events(
                session.initial_config, session.condition, session.seed, session.recorded_events(),
                session_id=session.id, preset=session.preset, created_at=session.created_at,
            )
            with lock:
                registry[session_id] = fresh
            return fresh.snapshot()
        fresh = _guard(session.reset_copy)
        register(fresh)
        return fresh.snapshot()

    @app.post("/api/sessions/{session_id}/intervene", response_model=S.StepResult)
    def intervene(session_id: str, req: S.Intervention) -> S.StepResult:
        session = get_session(session_id)
        with lock:
            event = _guard(lambda: session.intervene(req))
            persist_new(session, [event])
            return S.StepResult(events=[event], snapshot=session.snapshot())

    @app.post("/api/sessions/{session_id}/human_phrase", response_model=S.StepResult)
    def human_phrase(session_id: str, req: S.HumanPhraseRequest) -> S.StepResult:
        session = get_session(session_id)
        with lock:
            events = _guard(lambda: session.human_phrase(req))
            persist_new(session, events)
            return S.StepResult(events=events, snapshot=session.snapshot())

    @app.get("/api/sessions/{session_id}/events", response_model=list[S.Event])
    def events(session_id: str, from_seq: int = 0, limit: int = 500) -> list[S.Event]:
        session = get_session(session_id)
        return [e for e in session.events if e.seq >= from_seq][: max(0, limit)]

    @app.get("/api/sessions/{session_id}/agents/{agent_id}/inspect", response_model=S.AgentInspection)
    def inspect(session_id: str, agent_id: str) -> S.AgentInspection:
        session = get_session(session_id)
        return _guard(lambda: session.inspect(agent_id))

    @app.get("/api/sessions/{session_id}/export", response_model=S.RunExport)
    def export(session_id: str) -> S.RunExport:
        return get_session(session_id).export()

    @app.post("/api/sessions/import", response_model=S.SessionSnapshot)
    def import_session(req: S.RunExport) -> S.SessionSnapshot:
        session = _guard(lambda: Session.from_export(req))
        register(session)
        return session.snapshot()

    @app.post("/api/sessions/{session_id}/replay/step", response_model=S.StepResult)
    def replay_step(session_id: str, req: S.StepRequest) -> S.StepResult:
        session = get_session(session_id)
        with lock:
            events = _guard(lambda: session.replay_step(req.n))
            return S.StepResult(events=events, snapshot=session.snapshot())

    @app.post("/api/phrases/features", response_model=S.PhraseFeatures)
    def phrase_features(phrase: S.Phrase) -> S.PhraseFeatures:
        weights = load_config().music.feature_weights
        return S.PhraseFeatures(
            phrase_id=phrase.id, symbolic=symbolic_features(phrase, weights), audio=audio_features(phrase)
        )

    @app.post("/api/phrases/render", response_class=Response, responses={200: {"content": {"audio/wav": {}}}})
    def phrase_render(phrase: S.Phrase) -> Response:
        return Response(content=wav_bytes(phrase), media_type="audio/wav")

    @app.post("/api/phrases/transform", response_model=S.Phrase)
    def phrase_transform(req: TransformRequest) -> S.Phrase:
        return apply_transform(req.phrase, req.transform, np.random.default_rng(0))

    @app.get("/api/motifs/saved", response_model=list[S.SavedMotif])
    def saved_motifs() -> list[S.SavedMotif]:
        return store.list_motifs()

    @app.post("/api/motifs/saved", response_model=list[S.SavedMotif])
    def save_motif(req: S.SavedMotif) -> list[S.SavedMotif]:
        motif = req.model_copy(
            update={
                "created_at": req.created_at or now_iso(),
                "features": req.features or symbolic_features(req.phrase, load_config().music.feature_weights),
            }
        )
        store.save_motif(motif)
        return store.list_motifs()

    @app.post("/api/experiments", response_model=S.ExperimentResult)
    def run_experiment(req: S.ExperimentRequest) -> S.ExperimentResult:
        if not req.conditions or not req.seeds:
            raise HTTPException(status_code=400, detail="need at least one condition and one seed")
        result = _guard(lambda: run_experiment_batch(req))
        store.save_experiment(result)
        return result

    @app.get("/api/experiments", response_model=list[S.ExperimentResult])
    def list_experiments() -> list[S.ExperimentResult]:
        return store.list_experiments()

    @app.get("/api/experiments/{experiment_id}", response_model=S.ExperimentResult)
    def get_experiment(experiment_id: str) -> S.ExperimentResult:
        result = store.get_experiment(experiment_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"experiment {experiment_id} not found")
        return result

    @app.get("/api/experiments/{experiment_id}/csv", response_class=Response, responses={200: {"content": {"text/csv": {}}}})
    def experiment_csv(experiment_id: str) -> Response:
        result = get_experiment(experiment_id)
        return Response(
            content=to_csv(result),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="experiment_{experiment_id}.csv"'},
        )

    @app.get("/api/model/status", response_model=S.ModelStatus)
    def model_status_route() -> S.ModelStatus:
        return model_status(load_config().model)

    dist = Path(frontend_dir) if frontend_dir else FRONTEND_DIST
    if serve_frontend and (dist / "index.html").exists():
        # Mounted AFTER all /api routes so they take precedence; SPA fallback for other paths.
        app.mount("/", SPAStaticFiles(directory=dist, html=True), name="frontend")

    return app


app = create_app()
