"""Command line entry points.

  python -m resonance.cli run --seed 7 --condition full --episodes 120 --export out.json
  python -m resonance.cli experiments --seeds 1,2,3,4,5 --episodes 120 --out ../data/exports
  python -m resonance.cli presets --seed 7 --out ../data/exports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from resonance.conditions import standard_conditions, with_description
from resonance.config import load_config
from resonance.experiments import metrics_from_session, run_experiment, to_csv
from resonance.presets import PRESETS, run_preset_sessions
from resonance.schemas import ConditionName, ConditionSpec, ExperimentRequest
from resonance.session import Session


def _seeds(text: str) -> list[int]:
    if "-" in text and "," not in text:
        lo, hi = (int(x) for x in text.split("-"))
        return list(range(lo, hi + 1))
    return [int(x) for x in text.split(",") if x.strip()]


def cmd_run(args: argparse.Namespace) -> None:
    overrides = {"episodes": args.episodes} if args.episodes else {}
    config = load_config(overrides)
    condition = with_description(ConditionSpec(name=ConditionName(args.condition), perturbation=args.perturbation))
    session = Session(config, condition, args.seed)
    session.step(config.episodes)
    m = metrics_from_session(session)
    print(f"condition={condition.name.value} perturbation={condition.perturbation} seed={args.seed} episodes={m.episodes}")
    print(f"first_block={m.first_block_score:.3f} final_block={m.final_block_score:.3f} success_rate={m.success_rate:.3f}")
    print("learning_curve=" + " ".join(f"{v:.2f}" for v in m.learning_curve))
    if args.export:
        path = Path(args.export)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(session.export().model_dump_json(indent=1))
        print(f"export -> {path}")


def cmd_experiments(args: argparse.Namespace) -> None:
    overrides = {"episodes": args.episodes} if args.episodes else {}
    config = load_config(overrides)
    req = ExperimentRequest(name=args.name, conditions=standard_conditions(), seeds=_seeds(args.seeds), config=config)
    result = run_experiment(req)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"experiment_{args.name}.json").write_text(result.model_dump_json(indent=1))
    (out / f"experiment_{args.name}.csv").write_text(to_csv(result))
    for a in result.aggregates:
        print(
            f"{a.condition:16s} {a.perturbation:15s} n={a.n_runs} final_block={a.mean_final_block_score:.3f}"
            f"±{a.sd_final_block_score:.3f} success={a.mean_success_rate:.3f}±{a.sd_success_rate:.3f} "
            f"curve={' '.join(f'{v:.2f}' for v in a.mean_learning_curve)}"
        )
    print(f"wrote {out / f'experiment_{args.name}.json'} and .csv")


def cmd_presets(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in PRESETS:
        result, sessions = run_preset_sessions(name, args.seed)
        (out / f"preset_{name}.json").write_text(result.model_dump_json(indent=1))
        for i, s in enumerate(sessions):
            (out / f"preset_{name}_session{i + 1}_export.json").write_text(s.export().model_dump_json(indent=1))
        print(f"{name}: {result.narrative}")
        print("  comparison:", json.dumps({k: v for k, v in result.comparison.items() if not str(k).endswith("probs")}))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="resonance.cli", description="Resonance Lab simulation CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one session headless")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--condition", default="full", choices=[c.value for c in ConditionName])
    r.add_argument("--perturbation", default="none")
    r.add_argument("--episodes", type=int, default=None)
    r.add_argument("--export", default=None, help="write a RunExport JSON here")
    r.set_defaults(func=cmd_run)
    e = sub.add_parser("experiments", help="all six conditions (perturbed: transpose, rhythm_shuffle) x seeds")
    e.add_argument("--seeds", default="1-10", help="comma list or range, e.g. 1,2,3 or 1-10")
    e.add_argument("--episodes", type=int, default=None)
    e.add_argument("--name", default="batch")
    e.add_argument("--out", default="../data/exports")
    e.set_defaults(func=cmd_experiments)
    p = sub.add_parser("presets", help="run all presets and write results + session exports")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="../data/exports")
    p.set_defaults(func=cmd_presets)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
