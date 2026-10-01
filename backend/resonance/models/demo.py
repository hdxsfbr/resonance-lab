"""Run ONE model-assisted receiver decision and print the recorded model_call
event and the resulting PolicyTrace (both redacted), as JSON.

    cd backend && . .venv/bin/activate
    python -m resonance.models.demo                 # local llama.cpp (Qwen2.5-0.5B)
    python -m resonance.models.demo --provider scripted

Uses the demo receiver context from `resonance.models.testing` (a fixed
observation, learner scores, state and two retrieved memories) and a minimal
softmax fallback policy; it does not import the simulation core.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from resonance import schemas as S
from resonance.models.policy import ModelPolicy
from resonance.models.registry import build_model_policy, last_build_error, status
from resonance.models.testing import EventSink, FakeLocalPolicy, make_receiver_ctx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--provider", default="llamacpp", choices=["llamacpp", "scripted", "anthropic", "openai_compatible"]
    )
    ap.add_argument("--model-id", default="")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default="")
    ap.add_argument("--channel", default="music", choices=["music", "symbol"])
    args = ap.parse_args(argv)

    cfg = S.ModelConfig(
        provider=args.provider,
        model_id=args.model_id,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        use_for=["receiver"],
        max_retries=0,
    )
    print("# status", json.dumps(status(cfg).model_dump(), indent=2))
    pol = build_model_policy(cfg, FakeLocalPolicy())
    if not isinstance(pol, ModelPolicy):
        print(f"# provider unavailable, local policy only: {last_build_error.get(args.provider)}")
        return 1
    sink = EventSink()
    ctx = make_receiver_ctx(emit=sink, channel=args.channel)
    t0 = time.perf_counter()
    trace = pol.choose_pattern(ctx)
    wall = time.perf_counter() - t0
    for event_type, payload in sink.events:
        print(f"# event {event_type}")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("# trace")
    print(json.dumps(trace.model_dump(), indent=2, ensure_ascii=False))
    print(f"# wall time for the decision (incl. model load on first call): {wall:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
