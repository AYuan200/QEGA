"""Run the complete final-manuscript experiment matrix in dependency order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = {
    "calibration": ["taguchi_qega.yaml", "n_sensitivity.yaml"],
    "main": ["main_qega.yaml", "main_qsa.yaml", "main_qpso.yaml", "main_sga.yaml", "main_sweep.yaml"],
    "sorting": ["sorting_mks.yaml", "sorting_rds.yaml", "sorting_dps.yaml"],
    "uav": ["uav_ablation.yaml"],
    "clustering": ["clustering_qega.yaml", "clustering_order_search.yaml", "clustering_sga.yaml", "clustering_sweep.yaml"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stages", nargs="+", choices=[*STAGES, "milp"], default=[*STAGES, "milp"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--milp-time-limit", type=float, default=3600.0)
    args = parser.parse_args()
    for stage in args.stages:
        if stage == "milp":
            command = [sys.executable, str(ROOT / "scripts" / "run_milp.py"),
                       "--time-limit", str(args.milp_time_limit)]
            if args.resume:
                command.append("--resume")
        else:
            for filename in STAGES[stage]:
                command = [
                    sys.executable, str(ROOT / "run_experiment.py"),
                    "--config", str(ROOT / "configs" / filename),
                    "--workers", str(args.workers),
                ]
                if args.resume:
                    command.append("--resume")
                if args.limit is not None:
                    command.extend(["--limit", str(args.limit)])
                print(f"[run_all] stage={stage} config={filename}", flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
            continue
        print(f"[run_all] stage={stage}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
