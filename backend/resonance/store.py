"""SQLite persistence (stdlib sqlite3). No secrets are ever stored.

Path: env RESONANCE_DB, default <project>/data/resonance.db. The connection is opened
lazily, so importing the API never creates a database file.

Tables:
  sessions(id, created_at, mode, preset, seed, condition_json, config_json, status, step)
  events(session_id, seq, step, t, type, agent_id, visibility, payload_json)
  experiments(id, name, created_at, result_json)
  saved_motifs(name PK, created_at, phrase_json)

Sessions are stored as event logs and reload as replayable sessions (mode "replay").
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path

from resonance.config import PROJECT_ROOT
from resonance.schemas import (
    ConditionSpec,
    Event,
    ExperimentConfig,
    ExperimentResult,
    Phrase,
    SavedMotif,
    SessionSummary,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, created_at TEXT, mode TEXT, preset TEXT, seed INTEGER,
  condition_json TEXT, config_json TEXT, status TEXT, step INTEGER
);
CREATE TABLE IF NOT EXISTS events (
  session_id TEXT, seq INTEGER, step INTEGER, t TEXT, type TEXT, agent_id TEXT,
  visibility TEXT, payload_json TEXT, PRIMARY KEY (session_id, seq)
);
CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, name TEXT, created_at TEXT, result_json TEXT);
CREATE TABLE IF NOT EXISTS saved_motifs (name TEXT PRIMARY KEY, created_at TEXT, phrase_json TEXT);
"""


def default_db_path() -> Path:
    env = os.environ.get("RESONANCE_DB")
    return Path(env) if env else PROJECT_ROOT / "data" / "resonance.db"


class Store:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.executescript(SCHEMA)
        return self._conn

    # ---------------------------------------------------------------- sessions
    def save_session(self, summary: SessionSummary, config: ExperimentConfig, events: Iterable[Event]) -> None:
        """Insert or replace the session row and its full event log."""
        with self._lock:
            c = self.conn()
            c.execute("DELETE FROM events WHERE session_id = ?", (summary.id,))
            c.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    summary.id, summary.created_at, summary.mode, summary.preset, summary.seed,
                    summary.condition.model_dump_json(), config.model_dump_json(), summary.status, summary.step,
                ),
            )
            self._insert_events(c, summary.id, events)
            c.commit()

    def append_events(self, summary: SessionSummary, events: Iterable[Event]) -> None:
        with self._lock:
            c = self.conn()
            self._insert_events(c, summary.id, events)
            c.execute("UPDATE sessions SET status = ?, step = ? WHERE id = ?", (summary.status, summary.step, summary.id))
            c.commit()

    @staticmethod
    def _insert_events(c: sqlite3.Connection, session_id: str, events: Iterable[Event]) -> None:
        c.executemany(
            "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?)",
            [
                (session_id, e.seq, e.step, e.t, e.type, e.agent_id, e.visibility, json.dumps(e.payload))
                for e in events
            ],
        )

    def list_sessions(self) -> list[SessionSummary]:
        with self._lock:
            rows = self.conn().execute(
                "SELECT id, created_at, mode, preset, seed, condition_json, status, step FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [
            SessionSummary(
                id=r[0], created_at=r[1], mode=r[2], preset=r[3], seed=r[4],
                condition=ConditionSpec.model_validate_json(r[5]), status=r[6], step=r[7],
            )
            for r in rows
        ]

    def load_session_record(self, session_id: str) -> tuple[SessionSummary, ExperimentConfig, list[Event]] | None:
        with self._lock:
            c = self.conn()
            row = c.execute(
                "SELECT id, created_at, mode, preset, seed, condition_json, status, step, config_json FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            ev_rows = c.execute(
                "SELECT seq, step, t, type, agent_id, visibility, payload_json FROM events WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        summary = SessionSummary(
            id=row[0], created_at=row[1], mode=row[2], preset=row[3], seed=row[4],
            condition=ConditionSpec.model_validate_json(row[5]), status=row[6], step=row[7],
        )
        events = [
            Event(seq=r[0], step=r[1], t=r[2], type=r[3], agent_id=r[4], visibility=r[5], payload=json.loads(r[6]))
            for r in ev_rows
        ]
        return summary, ExperimentConfig.model_validate_json(row[8]), events

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            c = self.conn()
            c.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            c.commit()

    # ---------------------------------------------------------------- experiments
    def save_experiment(self, result: ExperimentResult) -> None:
        with self._lock:
            c = self.conn()
            c.execute(
                "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?)",
                (result.id, result.name, result.created_at, result.model_dump_json()),
            )
            c.commit()

    def list_experiments(self) -> list[ExperimentResult]:
        with self._lock:
            rows = self.conn().execute("SELECT result_json FROM experiments ORDER BY created_at DESC").fetchall()
        return [ExperimentResult.model_validate_json(r[0]) for r in rows]

    def get_experiment(self, experiment_id: str) -> ExperimentResult | None:
        with self._lock:
            row = self.conn().execute("SELECT result_json FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        return None if row is None else ExperimentResult.model_validate_json(row[0])

    # ---------------------------------------------------------------- saved motifs
    def save_motif(self, motif: SavedMotif) -> None:
        with self._lock:
            c = self.conn()
            c.execute(
                "INSERT OR REPLACE INTO saved_motifs VALUES (?,?,?)",
                (motif.name, motif.created_at, motif.model_dump_json()),
            )
            c.commit()

    def list_motifs(self) -> list[SavedMotif]:
        with self._lock:
            rows = self.conn().execute("SELECT phrase_json FROM saved_motifs ORDER BY created_at").fetchall()
        out = []
        for (blob,) in rows:
            data = json.loads(blob)
            # Rows hold the full SavedMotif; tolerate a bare Phrase for forward compatibility.
            out.append(SavedMotif.model_validate(data) if "phrase" in data else SavedMotif(
                name=data.get("id", "motif"), phrase=Phrase.model_validate(data), created_at=""))
        return out
