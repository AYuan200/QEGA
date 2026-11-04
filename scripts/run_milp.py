"""Compute small-instance MILP lower bounds with resumable CSV output."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qega.config import load_base_config
from qega.milp import solve_milp
from qega.model import Instance

FIELDS = ["instance_id", "N", "K", "M", "lower_bound_min", "incumbent_min",
          "solver_time_s", "status", "message", "mip_gap"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-dir", type=Path, default=ROOT / "results" / "instances")
    parser.add_argument("--pattern", default="*_small.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "milp_bounds.csv")
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.instance_dir.glob(args.pattern))
    if not paths:
        raise SystemExit("no small instances found; materialize clustering configs first")
    done = set()
    if args.output.exists() and args.resume:
        with args.output.open(newline="", encoding="utf-8") as handle:
            done = {row["instance_id"] for row in csv.DictReader(handle)}
    elif args.output.exists():
        raise SystemExit(f"{args.output} exists; pass --resume")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = not args.output.exists() or args.output.stat().st_size == 0
    service = load_base_config(ROOT / "configs" / "base.yaml").t_station_min
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if header:
            writer.writeheader()
            handle.flush()
        for path in paths:
            if path.stem in done:
                continue
            instance = Instance.from_canonical_json(path.read_text(encoding="utf-8"))
            result = solve_milp(instance, service, args.time_limit)
            writer.writerow({
                "instance_id": instance.instance_id, "N": instance.N,
                "K": instance.K, "M": instance.M,
                "lower_bound_min": result.lower_bound_min,
                "incumbent_min": result.incumbent_min,
                "solver_time_s": result.solver_time_s,
                "status": result.status, "message": result.message,
                "mip_gap": result.mip_gap,
            })
            handle.flush()
            print(f"[milp] {instance.instance_id} LB={result.lower_bound_min:.6f} "
                  f"time={result.solver_time_s:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
