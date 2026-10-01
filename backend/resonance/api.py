"""HTTP API for Resonance Lab.

Route signatures here ARE the contract (see docs/CONTRACTS.md). The simulation
workstream fills in the implementations; the frontend generates types from the
OpenAPI document this app produces (`make types`).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from resonance import schemas as S

VERSION = "0.1.0"


class RunPresetRequest(BaseModel):
    seed: int | None = None


class TransformRequest(BaseModel):
    phrase: S.Phrase
    transform: S.Perturbation


def create_app() -> FastAPI:
    app = FastAPI(title="Resonance Lab API", version=VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def todo() -> None:
        raise HTTPException(status_code=501, detail="not implemented yet")

    @app.get("/api/health", response_model=S.HealthResponse)
    def health() -> S.HealthResponse:
        todo()  # type: ignore[return-value]

    @app.get("/api/config/default", response_model=S.ExperimentConfig)
    def default_config() -> S.ExperimentConfig:
        todo()  # type: ignore[return-value]

    @app.get("/api/patterns", response_model=list[S.TimingPattern])
    def patterns() -> list[S.TimingPattern]:
        todo()  # type: ignore[return-value]

    @app.get("/api/motifs", response_model=list[S.Phrase])
    def motifs() -> list[S.Phrase]:
        todo()  # type: ignore[return-value]

    @app.get("/api/presets", response_model=list[S.PresetInfo])
    def presets() -> list[S.PresetInfo]:
        todo()  # type: ignore[return-value]

    @app.post("/api/presets/{name}/run", response_model=S.PresetResult)
    def run_preset(name: str, req: RunPresetRequest) -> S.PresetResult:
        todo()  # type: ignore[return-value]

    @app.post("/api/sessions", response_model=S.SessionSnapshot)
    def create_session(req: S.CreateSessionRequest) -> S.SessionSnapshot:
        todo()  # type: ignore[return-value]

    @app.get("/api/sessions", response_model=list[S.SessionSummary])
    def list_sessions() -> list[S.SessionSummary]:
        todo()  # type: ignore[return-value]

    @app.get("/api/sessions/{session_id}", response_model=S.SessionSnapshot)
    def get_session(session_id: str) -> S.SessionSnapshot:
        todo()  # type: ignore[return-value]

    @app.post("/api/sessions/{session_id}/step", response_model=S.StepResult)
    def step(session_id: str, req: S.StepRequest) -> S.StepResult:
        todo()  # type: ignore[return-value]

    @app.post("/api/sessions/{session_id}/reset", response_model=S.SessionSnapshot)
    def reset(session_id: str) -> S.SessionSnapshot:
        todo()  # type: ignore[return-value]

    @app.post("/api/sessions/{session_id}/intervene", response_model=S.StepResult)
    def intervene(session_id: str, req: S.Intervention) -> S.StepResult:
        todo()  # type: ignore[return-value]

    @app.post("/api/sessions/{session_id}/human_phrase", response_model=S.StepResult)
    def human_phrase(session_id: str, req: S.HumanPhraseRequest) -> S.StepResult:
        todo()  # type: ignore[return-value]

    @app.get("/api/sessions/{session_id}/events", response_model=list[S.Event])
    def events(session_id: str, from_seq: int = 0, limit: int = 500) -> list[S.Event]:
        todo()  # type: ignore[return-value]

    @app.get("/api/sessions/{session_id}/agents/{agent_id}/inspect", response_model=S.AgentInspection)
    def inspect(session_id: str, agent_id: str) -> S.AgentInspection:
        todo()  # type: ignore[return-value]

    @app.get("/api/sessions/{session_id}/export", response_model=S.RunExport)
    def export(session_id: str) -> S.RunExport:
        todo()  # type: ignore[return-value]

    @app.post("/api/sessions/import", response_model=S.SessionSnapshot)
    def import_session(req: S.RunExport) -> S.SessionSnapshot:
        todo()  # type: ignore[return-value]

    @app.post("/api/sessions/{session_id}/replay/step", response_model=S.StepResult)
    def replay_step(session_id: str, req: S.StepRequest) -> S.StepResult:
        todo()  # type: ignore[return-value]

    @app.post("/api/phrases/features", response_model=S.PhraseFeatures)
    def phrase_features(phrase: S.Phrase) -> S.PhraseFeatures:
        todo()  # type: ignore[return-value]

    @app.post("/api/phrases/render", response_class=Response, responses={200: {"content": {"audio/wav": {}}}})
    def phrase_render(phrase: S.Phrase) -> Response:
        todo()  # type: ignore[return-value]

    @app.post("/api/phrases/transform", response_model=S.Phrase)
    def phrase_transform(req: TransformRequest) -> S.Phrase:
        todo()  # type: ignore[return-value]

    @app.get("/api/motifs/saved", response_model=list[S.SavedMotif])
    def saved_motifs() -> list[S.SavedMotif]:
        todo()  # type: ignore[return-value]

    @app.post("/api/motifs/saved", response_model=list[S.SavedMotif])
    def save_motif(req: S.SavedMotif) -> list[S.SavedMotif]:
        todo()  # type: ignore[return-value]

    @app.post("/api/experiments", response_model=S.ExperimentResult)
    def run_experiment(req: S.ExperimentRequest) -> S.ExperimentResult:
        todo()  # type: ignore[return-value]

    @app.get("/api/experiments", response_model=list[S.ExperimentResult])
    def list_experiments() -> list[S.ExperimentResult]:
        todo()  # type: ignore[return-value]

    @app.get("/api/experiments/{experiment_id}", response_model=S.ExperimentResult)
    def get_experiment(experiment_id: str) -> S.ExperimentResult:
        todo()  # type: ignore[return-value]

    @app.get("/api/experiments/{experiment_id}/csv", response_class=Response, responses={200: {"content": {"text/csv": {}}}})
    def experiment_csv(experiment_id: str) -> Response:
        todo()  # type: ignore[return-value]

    @app.get("/api/model/status", response_model=S.ModelStatus)
    def model_status() -> S.ModelStatus:
        todo()  # type: ignore[return-value]

    return app


app = create_app()
