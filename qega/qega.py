"""可运行的 QEGA 主程序（SPEC §3.1–3.7、§3.9）。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import load_base_config
from .decoder import Decoder, EvaluationBudgetExhausted, Solution
from .model import Instance
from .operators import (
    Chromosome,
    crossover,
    initialize_population,
    mutate,
    preserve_elite,
    select_parent,
)
from .sorters import resolve_sorter


@dataclass(frozen=True)
class QEGAParams:
    N_p: int = 50
    p_c: float = 0.8
    p_m: float = 0.1
    alpha: float = 0.4
    FE_max: int = 50_000
    n: int | None = None
    init_2opt_passes: int | None = 2
    sorter: str = "mks"

    def __post_init__(self) -> None:
        if self.N_p <= 0 or self.FE_max <= 0:
            raise ValueError("N_p and FE_max must be positive")
        if not 0.0 <= self.p_c <= 1.0 or not 0.0 <= self.p_m <= 1.0:
            raise ValueError("p_c and p_m must be in [0, 1]")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.n is not None and self.n <= 0:
            raise ValueError("n must be positive")
        if self.init_2opt_passes is not None and self.init_2opt_passes < 0:
            raise ValueError("init_2opt_passes must be non-negative or None")
        if self.sorter not in {"mks", "rds", "dps"}:
            raise ValueError("sorter must be mks, rds, or dps")

    def to_dict(self) -> dict:
        return {
            "N_p": self.N_p,
            "p_c": self.p_c,
            "p_m": self.p_m,
            "alpha": self.alpha,
            "FE_max": self.FE_max,
            "n": self.n,
            "init_2opt_passes": self.init_2opt_passes,
            "sorter": self.sorter,
        }


@dataclass(frozen=True)
class QEGAResult:
    instance_id: str
    seed: int
    params: QEGAParams
    n: int
    best_chromosome: Chromosome
    best_solution: Solution
    best_times_min: dict[int, float]
    best_makespan_min: float
    num_evals: int

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "seed": self.seed,
            "params": self.params.to_dict(),
            "n": self.n,
            "best_chromosome": list(self.best_chromosome),
            "best_makespan_min": self.best_makespan_min,
            "best_times_min": {
                str(courier_id): self.best_times_min[courier_id]
                for courier_id in sorted(self.best_times_min)
            },
            "num_evals": self.num_evals,
            "solution": self.best_solution.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def default_cluster_count(order_count: int) -> int:
    """Final-manuscript rule: n=clamp(round(N/50), 15, 25)."""
    if order_count <= 0:
        raise ValueError("order_count must be positive")
    return max(1, min(order_count, min(25, max(15, int(round(order_count / 50.0))))))


def kmeans_queues(
    instance: Instance,
    n: int,
    rng: np.random.Generator,
    max_iter: int = 100,
) -> dict[int, list]:
    """按订单站点坐标执行确定种子驱动的 K-means。"""
    if not 1 <= n <= instance.N:
        raise ValueError("n must be in [1, N]")
    station_by_id = {station.id: station for station in instance.stations}
    points = np.array(
        [[station_by_id[order.station].x, station_by_id[order.station].y] for order in instance.orders],
        dtype=float,
    )
    count = len(points)
    centroids = [points[int(rng.integers(count))].copy()]
    while len(centroids) < n:
        current = np.array(centroids)
        distances = np.min(
            np.sum((points[:, None, :] - current[None, :, :]) ** 2, axis=2), axis=1
        )
        total = float(np.sum(distances))
        if total <= 1e-15:
            index = int(rng.integers(count))
        else:
            index = int(rng.choice(count, p=distances / total))
        centroids.append(points[index].copy())

    labels = np.full(count, -1, dtype=int)
    for _ in range(max_iter):
        centroid_array = np.array(centroids)
        distances = np.sum(
            (points[:, None, :] - centroid_array[None, :, :]) ** 2, axis=2
        )
        new_labels = np.argmin(distances, axis=1)
        new_centroids = []
        for cluster in range(n):
            members = points[new_labels == cluster]
            if len(members):
                new_centroids.append(np.mean(members, axis=0))
            else:
                new_centroids.append(centroid_array[cluster])
        converged = np.array_equal(new_labels, labels) and np.allclose(
            np.array(new_centroids), centroid_array, rtol=0.0, atol=1e-12
        )
        labels = new_labels
        centroids = new_centroids
        if converged:
            break

    queues = {cluster + 1: [] for cluster in range(n)}
    for order, label in zip(instance.orders, labels):
        queues[int(label) + 1].append(order)
    return queues


def _queue_centroids(instance: Instance, queues: dict[int, list]) -> dict[int, tuple[float, float]]:
    stations = {station.id: station for station in instance.stations}
    centroids = {}
    for queue_id, orders in queues.items():
        if orders:
            centroids[queue_id] = (
                math.fsum(stations[order.station].x for order in orders) / len(orders),
                math.fsum(stations[order.station].y for order in orders) / len(orders),
            )
        else:
            centroids[queue_id] = (0.0, 0.0)
    return centroids


def run_qega(
    instance: Instance,
    params: QEGAParams | None = None,
    seed: int = 0,
    t_station_min: float | None = None,
    config_path: str | Path | None = None,
) -> QEGAResult:
    base = load_base_config(config_path)
    if params is None:
        params = QEGAParams(
            N_p=base.N_p,
            p_c=base.p_c,
            p_m=base.p_m,
            alpha=base.alpha,
            FE_max=base.FE_max,
            init_2opt_passes=base.init_2opt_passes,
        )
    service_time = base.t_station_min if t_station_min is None else float(t_station_min)
    n = params.n or default_cluster_count(instance.N)
    rng = np.random.default_rng(seed)
    queues = kmeans_queues(instance, n, rng)
    decoder = Decoder(
        instance,
        queues,
        t_station_min=service_time,
        fe_max=params.FE_max,
        sorter=resolve_sorter(params.sorter, rng),
    )

    best_chromosome: Chromosome | None = None
    best_result = None

    def evaluate(chromosome: Sequence[int]) -> float:
        nonlocal best_chromosome, best_result
        chromosome_tuple = tuple(chromosome)
        result = decoder.decode(chromosome_tuple)
        if best_result is None or result.makespan_min < best_result.makespan_min:
            best_chromosome = chromosome_tuple
            best_result = result
        return result.makespan_min

    try:
        population = initialize_population(
            _queue_centroids(instance, queues),
            params.N_p,
            params.alpha,
            rng,
            evaluate,
            two_opt_max_passes=params.init_2opt_passes,
        )
        fitnesses = [evaluate(chromosome) for chromosome in population]

        while decoder.fe_count < params.FE_max:
            new_population: list[Chromosome] = []
            while len(new_population) < params.N_p:
                parent1 = select_parent(population, fitnesses, rng)
                parent2 = select_parent(population, fitnesses, rng)
                new_population.extend(crossover(parent1, parent2, rng, params.p_c))
            new_population = new_population[:params.N_p]
            new_population = [
                mutate(chromosome, rng, params.p_m) for chromosome in new_population
            ]
            new_fitnesses = []
            for chromosome in new_population:
                if decoder.fe_count >= params.FE_max:
                    break
                new_fitnesses.append(evaluate(chromosome))
            if len(new_fitnesses) < params.N_p:
                break
            new_population, new_fitnesses = preserve_elite(
                new_population,
                new_fitnesses,
                best_chromosome,
                best_result.makespan_min,
            )
            population, fitnesses = new_population, new_fitnesses
    except EvaluationBudgetExhausted:
        pass

    if best_result is None or best_chromosome is None:
        raise RuntimeError("QEGA finished without any fitness evaluation")
    return QEGAResult(
        instance_id=instance.instance_id,
        seed=int(seed),
        params=params,
        n=n,
        best_chromosome=best_chromosome,
        best_solution=best_result.solution,
        best_times_min=best_result.times_min,
        best_makespan_min=best_result.makespan_min,
        num_evals=decoder.fe_count,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run QEGA on a canonical instance JSON")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fe-max", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    base = load_base_config(args.config)
    params = QEGAParams(
        N_p=base.N_p,
        p_c=base.p_c,
        p_m=base.p_m,
        alpha=base.alpha,
        FE_max=args.fe_max or base.FE_max,
        n=args.n,
        init_2opt_passes=base.init_2opt_passes,
    )
    instance = Instance.from_canonical_json(args.instance.read_text(encoding="utf-8"))
    result = run_qega(instance, params=params, seed=args.seed, config_path=args.config)
    payload = result.to_canonical_json() + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "instance_id": result.instance_id,
        "makespan_min": result.best_makespan_min,
        "num_evals": result.num_evals,
        "output": str(args.output) if args.output else None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
