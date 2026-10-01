"""Model-assisted end-to-end check with the real local llama.cpp model (agent B as receiver). Run from backend/: `.venv/bin/python scripts/model_e2e.py`."""
import json, time
from resonance import schemas as S
from resonance.config import load_config
from resonance.session import Session
cfg = load_config()
cfg.episodes = 4
cfg.model.provider = "llamacpp"
cfg.model.model_id = "qwen2.5-0.5b-instruct-q4_k_m"
cfg.model.use_for = ["receiver"]
cfg.model.call_budget = 3
cfg.agents[1].policy_kind = "model"
s = Session(cfg, S.ConditionSpec(name="full"), seed=7)
t = time.time()
events = s.step(4)
print(f"4 episodes in {time.time()-t:.1f}s; events={len(events)}")
mc = [e for e in events if e.type == "model_call"]
print("model_call events:", len(mc))
for e in mc:
    p = e.payload
    print(" step", e.step, "agent", e.agent_id, "ok", p.get("ok"), "fallback", p.get("fallback_used"), "modality", p.get("input_modality"), "err", p.get("error") or p.get("response",{}).get("error"), "budget_left", p.get("budget_remaining"))
    req = p.get("request", {})
    txt = json.dumps(req).lower()
    assert "target" not in txt.replace("target_pattern", "") or e.visibility == "sender_private", "TARGET LEAK in receiver prompt"
    resp = p.get("response", {})
    print("   parsed:", resp.get("parsed"), "latency_ms", resp.get("latency_ms"))
traces = [(e.step, e.agent_id, e.payload.get("trace", {}).get("policy_kind")) for e in events if e.type == "action_chosen"]
print("action traces (step, agent, kind):", traces)
snap = s.snapshot()
print("B policy_kind:", snap.agents[1].policy_kind, "scores:", snap.metrics.score_history)
# secrets check: no env var values in the export
exp = s.export().model_dump_json()
import os
leaks = [k for k,v in os.environ.items() if len(v) > 12 and v in exp]
print("env values present in export:", leaks)
