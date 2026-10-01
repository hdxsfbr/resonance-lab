import io
import wave

import pytest
from fastapi.testclient import TestClient

from resonance.api import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(db_path=tmp_path / "api.db", serve_frontend=False)
    with TestClient(app) as c:
        yield c


def test_health_and_static_data(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and "model" in h and "provider" in h["model"]
    assert client.get("/api/config/default").json()["task"]["n_patterns"] == 4
    assert len(client.get("/api/patterns").json()) == 4
    motifs = client.get("/api/motifs").json()
    assert len(motifs) == 8 and motifs[0]["motif_id"] == "m0-ascent"
    status = client.get("/api/model/status").json()
    assert {"provider", "available", "tested_in_this_environment"} <= set(status)


def test_session_lifecycle(client):
    snap = client.post("/api/sessions", json={}).json()
    sid = snap["id"]
    assert snap["status"] == "ready" and len(snap["agents"]) == 2
    res = client.post(f"/api/sessions/{sid}/step", json={"n": 5}).json()
    assert res["snapshot"]["step"] == 5 and res["events"][-1]["type"] == "episode_complete"
    first_phrase = next(e for e in res["events"] if e["type"] == "phrase_sent")["payload"]["phrase"]["id"]
    interventions = [
        {"kind": "replay_motif", "phrase_id": first_phrase},
        {"kind": "reset_memory", "agent_id": "A"},
        {"kind": "reset_state", "agent_id": "B"},
        {"kind": "freeze_state", "agent_id": "A", "value": True},
        {"kind": "set_coupling", "agent_id": "B", "value": False},
        {"kind": "swap_feature", "transform": "transpose"},
        {"kind": "set_param", "path": "state.decay.activation", "value": 0.2},
    ]
    for iv in interventions:
        r = client.post(f"/api/sessions/{sid}/intervene", json=iv)
        assert r.status_code == 200, (iv, r.text)
        ev = r.json()["events"]
        assert len(ev) == 1 and ev[0]["type"] == "intervention" and ev[0]["payload"]["effective_from_step"] == 5
    bad = client.post(f"/api/sessions/{sid}/intervene", json={"kind": "set_param", "path": "task.n_patterns", "value": 3})
    assert bad.status_code == 400
    motif = client.get("/api/motifs").json()[1]
    hp = client.post(f"/api/sessions/{sid}/human_phrase", json={"phrase": motif, "target_id": 0}).json()
    assert hp["snapshot"]["step"] == 6
    insp = client.get(f"/api/sessions/{sid}/agents/A/inspect").json()
    assert insp["agent"]["frozen_state"] is True and len(insp["information_received"]) >= 3
    assert client.get(f"/api/sessions/{sid}/agents/Z/inspect").status_code == 404
    evs = client.get(f"/api/sessions/{sid}/events", params={"from_seq": 3, "limit": 4}).json()
    assert [e["seq"] for e in evs] == [3, 4, 5, 6]
    assert any(s["id"] == sid for s in client.get("/api/sessions").json())
    # export -> import -> replay
    export = client.get(f"/api/sessions/{sid}/export").json()
    imported = client.post("/api/sessions/import", json=export).json()
    assert imported["mode"] == "replay" and imported["id"] != sid
    rep = client.post(f"/api/sessions/{imported['id']}/replay/step", json={"n": 100}).json()
    assert rep["snapshot"]["metrics"] == client.get(f"/api/sessions/{sid}").json()["metrics"]
    assert client.post(f"/api/sessions/{imported['id']}/step", json={"n": 1}).status_code == 409
    reset = client.post(f"/api/sessions/{sid}/reset").json()
    assert reset["step"] == 0 and reset["id"] == sid


def test_store_reload(tmp_path):
    db = tmp_path / "reload.db"
    with TestClient(create_app(db_path=db, serve_frontend=False)) as c1:
        sid = c1.post("/api/sessions", json={}).json()["id"]
        c1.post(f"/api/sessions/{sid}/step", json={"n": 4})
        scores = c1.get(f"/api/sessions/{sid}").json()["metrics"]["score_history"]
    with TestClient(create_app(db_path=db, serve_frontend=False)) as c2:
        snap = c2.get(f"/api/sessions/{sid}").json()
        assert snap["mode"] == "replay"
        rep = c2.post(f"/api/sessions/{sid}/replay/step", json={"n": 10}).json()
        assert rep["snapshot"]["metrics"]["score_history"] == scores


def test_phrase_endpoints(client):
    phrase = client.get("/api/motifs").json()[3]
    feats = client.post("/api/phrases/features", json=phrase).json()
    assert feats["symbolic"]["note_count"] == len(phrase["notes"]) and feats["audio"]["synth_version"] == "numpy-synth-1"
    wav = client.post("/api/phrases/render", json=phrase)
    assert wav.status_code == 200 and wav.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(wav.content)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
    t = client.post("/api/phrases/transform", json={"phrase": phrase, "transform": "transpose"}).json()
    assert t["origin"]["kind"] == "transformed" and t["notes"][0]["pitch"] == phrase["notes"][0]["pitch"] + 5
    saved = client.post("/api/motifs/saved", json={"name": "mine", "phrase": phrase, "created_at": ""}).json()
    assert saved[0]["name"] == "mine" and saved[0]["features"] is not None and saved[0]["created_at"]
    assert len(client.get("/api/motifs/saved").json()) == 1


def test_experiments_endpoints(client):
    cfg = client.get("/api/config/default").json()
    cfg["episodes"] = 20
    req = {"name": "api", "conditions": [{"name": "full"}, {"name": "no_history"}], "seeds": [1, 2], "config": cfg}
    res = client.post("/api/experiments", json=req).json()
    assert len(res["runs"]) == 4 and len(res["aggregates"]) == 2
    assert client.get("/api/experiments").json()[0]["id"] == res["id"]
    assert client.get(f"/api/experiments/{res['id']}").json()["name"] == "api"
    csv = client.get(f"/api/experiments/{res['id']}/csv")
    assert csv.status_code == 200 and csv.text.startswith("row_type")
    assert client.get("/api/experiments/nope").status_code == 404


def test_presets_endpoints(client):
    presets = client.get("/api/presets").json()
    assert {p["name"] for p in presets} == {"first_encounter", "shared_history", "same_phrase_different_history"}
    assert all(p["evidence_criterion"] and p["manipulation"] for p in presets)
    cfg = client.get("/api/config/default").json()
    cfg["episodes"] = 12
    res = client.post("/api/presets/first_encounter/run", json={"seed": 3, "config": cfg}).json()
    assert res["preset"] == "first_encounter" and res["comparison"]["episodes"] == 12
    sid = res["sessions"][0]["id"]
    assert client.get(f"/api/sessions/{sid}").json()["step"] == 12
    res2 = client.post("/api/presets/same_phrase_different_history/run", json={"seed": 3, "episodes": 10}).json()
    assert len(res2["sessions"]) == 2 and "constructed" in res2["narrative"].lower()
    assert client.post("/api/presets/nope/run", json={}).status_code == 404
    live = client.post("/api/sessions", json={"preset": "shared_history"}).json()
    assert live["preset"] == "shared_history" and live["config"]["episodes"] == 120


def test_spa_static_fallback(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>lab</html>")
    (dist / "app.js").write_text("console.log(1)")
    app = create_app(db_path=tmp_path / "s.db", frontend_dir=dist)
    with TestClient(app) as c:
        assert c.get("/").text == "<html>lab</html>"
        assert c.get("/app.js").text == "console.log(1)"
        assert c.get("/lab/session/123").text == "<html>lab</html>"  # SPA route
        assert c.get("/api/health").json()["status"] == "ok"
        assert c.get("/api/nope").status_code == 404
