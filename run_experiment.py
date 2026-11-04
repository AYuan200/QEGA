"""Configuration-driven experiment runner with checkpointing and validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from qega.baselines import (
    QPSOParams,
    QSAParams,
    SGAParams,
    SweepParams,
    run_order_search,
    run_qpso,
    run_qsa,
    run_sga,
    run_sweep,
)
from qega.config import load_base_config, load_baseline_config
from qega.generator import GeneratorParams, generate_instance
from qega.instances import iter_instance_specs
from qega.model import Instance
from qega.qega import QEGAParams, run_qega
from qega.validator import validate_solution

CSV_FIELDS = [
    "experiment", "algorithm", "params_json", "instance_id",
    "N", "K", "M", "type", "drone_ratio",
    "run", "seed", "makespan_min", "runtime_s", "feasible", "num_evals",
]
ALGORITHMS = ("qega", "qsa", "qpso", "sga", "sweep", "order_search")
ALGORITHM_PARAMS = {
    "qega": QEGAParams,
    "qsa": QSAParams,
    "qpso": QPSOParams,
    "sga": SGAParams,
    "sweep": SweepParams,
    "order_search": QSAParams,
}
ALGORITHM_RUNNERS = {
    "qega": run_qega,
    "qsa": run_qsa,
    "qpso": run_qpso,
    "sga": run_sga,
    "sweep": run_sweep,
    "order_search": run_order_search,
}
_INSTANCE_CACHE: dict[str, Instance] = {}


def _load_instance(path: str) -> Instance:
    if path not in _INSTANCE_CACHE:
        if len(_INSTANCE_CACHE) >= 8:
            _INSTANCE_CACHE.clear()
        _INSTANCE_CACHE[path] = Instance.from_canonical_json(
            Path(path).read_text(encoding="utf-8")
        )
    return _INSTANCE_CACHE[path]


def _drone_ratio(instance: Instance) -> float:
    return sum(c.vehicle_type.aerial for c in instance.couriers) / instance.M


def _canonical_params_json(params: dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _run_one(task: dict) -> dict:
    instance = _load_instance(task["instance_path"])
    algo_params = {k: v for k, v in task["params"].items() if k != "label"}
    start = time.perf_counter()
    algorithm = task["algorithm"]
    result = ALGORITHM_RUNNERS[algorithm](
        instance,
        params=ALGORITHM_PARAMS[algorithm](**algo_params),
        seed=task["seed"],
        t_station_min=task["t_station_min"],
    )
    runtime = time.perf_counter() - start
    violations = validate_solution(instance, result.best_solution)
    if violations:
        details = "\n".join(str(v) for v in violations[:10])
        raise RuntimeError(
            f"infeasible solution rejected: {task['experiment']} {algorithm} "
            f"{task['params_json']} {instance.instance_id} run={task['run']} "
            f"seed={task['seed']}\n{details}"
        )
    return {
        "experiment": task["experiment"],
        "algorithm": algorithm,
        "params_json": task["params_json"],
        "instance_id": instance.instance_id,
        "N": instance.N,
        "K": instance.K,
        "M": instance.M,
        "type": instance.type_label,
        "drone_ratio": round(_drone_ratio(instance), 6),
        "run": task["run"],
        "seed": task["seed"],
        "makespan_min": result.best_makespan_min,
        "runtime_s": round(runtime, 6),
        "feasible": True,
        "num_evals": result.num_evals,
    }


def expand_instance_entries(cfg: dict) -> list[dict]:
    if "instances" in cfg:
        return list(cfg["instances"])
    grid = cfg.get("instance_grid")
    if grid is None:
        raise ValueError("configuration needs instances or instance_grid")
    master_seed = int(grid["master_seed"])
    replicates = set(grid.get("replicates", range(1, 11)))
    filters = {
        key: set(grid[key]) if key in grid else None
        for key in ("N", "K", "M")
    }
    schemes = set(grid.get("schemes", [])) or None
    ratios = grid.get("drone_ratios", [None])
    limit = grid.get("limit")
    entries = []
    for spec in iter_instance_specs(master_seed):
        if spec.replicate not in replicates:
            continue
        if any(filters[key] is not None and getattr(spec.params, key) not in filters[key]
               for key in filters):
            continue
        if schemes is not None and spec.params.scheme_code not in schemes:
            continue
        for ratio in ratios:
            instance_id = spec.instance_id
            if ratio is not None:
                instance_id += f"_dr{float(ratio):.2f}"
            entries.append({
                "id": instance_id,
                "generate": {
                    "N": spec.params.N,
                    "K": spec.params.K,
                    "M": spec.params.M,
                    "scheme": spec.params.scheme_code,
                    "seed": spec.seed,
                    "drone_ratio": ratio,
                },
            })
            if limit is not None and len(entries) >= int(limit):
                return entries
    return entries


def materialize_instances(entries: list[dict], instance_dir: Path) -> list[Path]:
    instance_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for entry in entries:
        instance_id = entry["id"]
        path = instance_dir / f"{instance_id}.json"
        if not path.exists():
            gen = entry.get("generate")
            if gen is None:
                raise FileNotFoundError(
                    f"instance {instance_id} not found in {instance_dir} "
                    "and no generate specification was given"
                )
            scheme = gen.get("scheme", "RRR")
            params = GeneratorParams(
                N=int(gen["N"]), K=int(gen["K"]), M=int(gen["M"]),
                scheme_packages=scheme[0], scheme_weight=scheme[1],
                scheme_split=scheme[2], drone_ratio=gen.get("drone_ratio"),
            )
            instance = generate_instance(params, int(gen["seed"]), instance_id)
            path.write_text(instance.to_canonical_json() + "\n", encoding="utf-8")
            print(f"[runner] generated {instance_id} (seed={gen['seed']})", flush=True)
        paths.append(path)
    return paths


def _algorithm_params(cfg: dict, base, baselines) -> dict:
    algorithm = cfg.get("algorithm", "qega")
    anchor = dict(cfg.get("anchor") or {})
    fe_max = int(cfg.get("fe_max", anchor.pop("FE_max", base.FE_max)))
    n = anchor.pop("n", None) if algorithm != "sga" else None
    if n is not None:
        n = int(n)
    sorter = str(anchor.pop("sorter", cfg.get("sorter", "mks"))).lower()
    if algorithm == "qega":
        params = {
            "N_p": int(anchor.pop("N_p", base.N_p)),
            "p_c": float(anchor.pop("p_c", base.p_c)),
            "p_m": float(anchor.pop("p_m", base.p_m)),
            "alpha": float(anchor.pop("alpha", base.alpha)),
            "FE_max": fe_max, "n": n,
            "init_2opt_passes": anchor.pop("init_2opt_passes", base.init_2opt_passes),
            "sorter": sorter,
        }
    elif algorithm in {"qsa", "order_search"}:
        defaults = baselines["qsa"]
        params = {
            "FE_max": fe_max, "n": n,
            "initial_temperature": float(anchor.pop(
                "initial_temperature", defaults["initial_temperature"])),
            "cooling_rate": float(anchor.pop("cooling_rate", defaults["cooling_rate"])),
            "sorter": sorter,
        }
    elif algorithm == "qpso":
        defaults = baselines["qpso"]
        params = {
            "N_p": int(anchor.pop("N_p", defaults["N_p"])),
            "FE_max": fe_max, "n": n,
            "inertia": float(anchor.pop("inertia", defaults["inertia"])),
            "cognitive": float(anchor.pop("cognitive", defaults["cognitive"])),
            "social": float(anchor.pop("social", defaults["social"])),
            "sorter": sorter,
        }
    elif algorithm == "sga":
        defaults = baselines["sga"]
        params = {
            "N_p": int(anchor.pop("N_p", defaults["N_p"])),
            "p_c": float(anchor.pop("p_c", defaults["p_c"])),
            "p_m": float(anchor.pop("p_m", defaults["p_m"])),
            "FE_max": fe_max, "sorter": sorter,
        }
    elif algorithm == "sweep":
        defaults = baselines["sweep"]
        params = {
            "n": n,
            "two_opt_max_passes": anchor.pop(
                "two_opt_max_passes", defaults["two_opt_max_passes"]),
            "sorter": sorter,
        }
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    if anchor:
        raise ValueError(f"unknown anchor keys: {sorted(anchor)}")
    return params


def _build_tasks(cfg: dict, root: Path) -> tuple[list[dict], Path]:
    experiment = cfg["experiment"]
    algorithm = cfg.get("algorithm", "qega")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    base = load_base_config(root / cfg.get("base_config", "configs/base.yaml"))
    baselines = load_baseline_config(
        root / cfg.get("baseline_config", "configs/baselines.yaml"))
    base_params = _algorithm_params(cfg, base, baselines)
    runs = int(cfg["runs"])
    seeds = [int(s) for s in cfg["seeds"]]
    if len(seeds) != runs:
        raise ValueError(f"len(seeds)={len(seeds)} != runs={runs}")
    instance_dir = root / cfg.get("instance_dir", "results/instances")
    entries = expand_instance_entries(cfg)
    instance_paths = materialize_instances(entries, instance_dir)

    tasks = []
    for path in instance_paths:
        for grid_entry in cfg["param_grid"]:
            entry = dict(grid_entry)
            label = entry.pop("label", None)
            unknown = set(entry) - set(base_params)
            if unknown:
                raise ValueError(f"unknown param_grid keys: {sorted(unknown)}")
            params = {**base_params, **entry}
            if label is not None:
                params["label"] = label
            params_json = _canonical_params_json(params)
            for run_idx in range(runs):
                tasks.append({
                    "experiment": experiment, "algorithm": algorithm,
                    "params": params, "params_json": params_json,
                    "instance_path": str(path), "instance_id": path.stem,
                    "run": run_idx + 1, "seed": seeds[run_idx],
                    "t_station_min": base.t_station_min,
                })
    output = root / cfg.get("output", f"results/{experiment}.csv")
    return tasks, output


def _existing_keys(output: Path) -> set[tuple]:
    keys = set()
    if not output.exists():
        return keys
    with output.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            keys.add((row["experiment"], row["algorithm"], row["params_json"],
                      row["instance_id"], int(row["run"])))
    return keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    tasks, output = _build_tasks(cfg, root)
    if output.exists() and not args.resume:
        raise SystemExit(f"{output} already exists; pass --resume to continue")
    done = _existing_keys(output) if args.resume else set()
    pending = [t for t in tasks if (
        t["experiment"], t["algorithm"], t["params_json"], t["instance_id"], t["run"]
    ) not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    total = len(pending)
    print(f"[runner] config={args.config} tasks={len(tasks)} done={len(done)} "
          f"pending={total} workers={args.workers}", flush=True)
    if not pending:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    need_header = not output.exists() or output.stat().st_size == 0
    started = time.perf_counter()
    completed = 0
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if need_header:
            writer.writeheader()
            handle.flush()

        def write_row(row: dict) -> None:
            nonlocal completed
            writer.writerow(row)
            handle.flush()
            completed += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (total - completed)
            print(f"[runner] {completed}/{total} {row['instance_id']} "
                  f"run={row['run']} makespan={row['makespan_min']:.2f}min "
                  f"runtime={row['runtime_s']:.1f}s ETA={eta / 60:.1f}min", flush=True)

        try:
            if args.workers <= 1:
                for task in pending:
                    write_row(_run_one(task))
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(_run_one, task) for task in pending]
                    for future in as_completed(futures):
                        write_row(future.result())
        except RuntimeError as exc:
            print(f"[runner] FATAL: {exc}", file=sys.stderr)
            return 1
    print(f"[runner] finished {completed} runs in "
          f"{(time.perf_counter() - started) / 60:.1f} min -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
