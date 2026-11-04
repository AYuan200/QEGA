"""SPEC §3.8 的 QSA、QPSO、SGA 与 sweep 基线。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import load_base_config
from .decoder import DecodeResult, Decoder, EvaluationBudgetExhausted, Solution
from .model import Instance
from .operators import (
    Chromosome,
    mutate,
    preserve_elite,
    select_parent,
    sweep_permutation,
    two_opt,
)
from .qega import _queue_centroids, default_cluster_count, kmeans_queues
from .sorters import resolve_sorter


@dataclass(frozen=True)
class QSAParams:
    FE_max: int = 50_000
    n: int | None = None
    initial_temperature: float = 100.0
    cooling_rate: float = 0.995
    sorter: str = "mks"

    def __post_init__(self) -> None:
        if self.FE_max <= 0:
            raise ValueError("FE_max must be positive")
        if self.n is not None and self.n <= 0:
            raise ValueError("n must be positive")
        if self.initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive")
        if not 0.0 < self.cooling_rate < 1.0:
            raise ValueError("cooling_rate must be in (0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QPSOParams:
    N_p: int = 50
    FE_max: int = 50_000
    n: int | None = None
    inertia: float = 0.9
    cognitive: float = 2.0
    social: float = 2.0
    sorter: str = "mks"

    def __post_init__(self) -> None:
        if self.N_p <= 0 or self.FE_max <= 0:
            raise ValueError("N_p and FE_max must be positive")
        if self.n is not None and self.n <= 0:
            raise ValueError("n must be positive")
        if not 0.0 <= self.inertia <= 1.0:
            raise ValueError("inertia must be in [0, 1]")
        if self.cognitive < 0.0 or self.social < 0.0:
            raise ValueError("learning factors must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SGAParams:
    N_p: int = 50
    p_c: float = 0.9
    p_m: float = 0.1
    FE_max: int = 50_000
    sorter: str = "mks"

    def __post_init__(self) -> None:
        if self.N_p <= 0 or self.FE_max <= 0:
            raise ValueError("N_p and FE_max must be positive")
        if not 0.0 <= self.p_c <= 1.0 or not 0.0 <= self.p_m <= 1.0:
            raise ValueError("p_c and p_m must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SweepParams:
    n: int | None = None
    two_opt_max_passes: int | None = 2
    sorter: str = "mks"

    def __post_init__(self) -> None:
        if self.n is not None and self.n <= 0:
            raise ValueError("n must be positive")
        if self.two_opt_max_passes is not None and self.two_opt_max_passes < 0:
            raise ValueError("two_opt_max_passes must be non-negative or None")

    def to_dict(self) -> dict:
        return asdict(self)


BaselineParams = QSAParams | QPSOParams | SGAParams | SweepParams


@dataclass(frozen=True)
class BaselineResult:
    algorithm: str
    instance_id: str
    seed: int
    params: BaselineParams
    n: int
    best_chromosome: Chromosome
    best_solution: Solution
    best_times_min: dict[int, float]
    best_makespan_min: float
    num_evals: int

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
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


class _Evaluation:
    def __init__(self, decoder: Decoder) -> None:
        self.decoder = decoder
        self.best_chromosome: Chromosome | None = None
        self.best_result: DecodeResult | None = None

    def __call__(self, chromosome: Sequence[int]) -> float:
        chromosome_tuple = tuple(chromosome)
        result = self.decoder.decode(chromosome_tuple)
        if self.best_result is None or result.makespan_min < self.best_result.makespan_min:
            self.best_chromosome = chromosome_tuple
            self.best_result = result
        return result.makespan_min


def _service_time(
    t_station_min: float | None, config_path: str | Path | None
) -> float:
    if t_station_min is not None:
        return float(t_station_min)
    return load_base_config(config_path).t_station_min


def _queue_evaluation(
    instance: Instance,
    n: int,
    sorter: str,
    seed: int,
    service_time: float,
    fe_max: int | None,
) -> tuple[np.random.Generator, dict[int, list], _Evaluation]:
    rng = np.random.default_rng(seed)
    queues = kmeans_queues(instance, n, rng)
    decoder = Decoder(
        instance,
        queues,
        t_station_min=service_time,
        fe_max=fe_max,
        sorter=resolve_sorter(sorter, rng),
    )
    return rng, queues, _Evaluation(decoder)


def _result(
    algorithm: str,
    instance: Instance,
    seed: int,
    params: BaselineParams,
    n: int,
    evaluation: _Evaluation,
) -> BaselineResult:
    if evaluation.best_result is None or evaluation.best_chromosome is None:
        raise RuntimeError(f"{algorithm} finished without any fitness evaluation")
    return BaselineResult(
        algorithm=algorithm,
        instance_id=instance.instance_id,
        seed=int(seed),
        params=params,
        n=n,
        best_chromosome=evaluation.best_chromosome,
        best_solution=evaluation.best_result.solution,
        best_times_min=evaluation.best_result.times_min,
        best_makespan_min=evaluation.best_result.makespan_min,
        num_evals=evaluation.decoder.fe_count,
    )


def run_qsa(
    instance: Instance,
    params: QSAParams | None = None,
    seed: int = 0,
    t_station_min: float | None = None,
    config_path: str | Path | None = None,
) -> BaselineResult:
    params = params or QSAParams()
    n = params.n or default_cluster_count(instance.N)
    rng, queues, evaluation = _queue_evaluation(
        instance,
        n,
        params.sorter,
        seed,
        _service_time(t_station_min, config_path),
        params.FE_max,
    )
    decoder = evaluation.decoder
    genes = tuple(sorted(queues))
    current = tuple(int(x) for x in rng.permutation(genes))
    temperature = params.initial_temperature
    try:
        current_fitness = evaluation(current)
        while decoder.fe_count < params.FE_max:
            candidate_list = list(current)
            if n > 1:
                a, b = (int(x) for x in rng.choice(n, size=2, replace=False))
                candidate_list[a], candidate_list[b] = (
                    candidate_list[b],
                    candidate_list[a],
                )
            candidate = tuple(candidate_list)
            candidate_fitness = evaluation(candidate)
            delta = candidate_fitness - current_fitness
            if delta <= 0.0 or float(rng.random()) < math.exp(-delta / temperature):
                current, current_fitness = candidate, candidate_fitness
            temperature *= params.cooling_rate
    except EvaluationBudgetExhausted:
        pass
    return _result("qsa", instance, seed, params, n, evaluation)


def _swaps_between(source: Chromosome, target: Chromosome) -> tuple[tuple[int, int], ...]:
    current = list(source)
    positions = {gene: index for index, gene in enumerate(current)}
    swaps = []
    for index, gene in enumerate(target):
        other_index = positions[gene]
        if other_index == index:
            continue
        displaced = current[index]
        current[index], current[other_index] = current[other_index], current[index]
        positions[gene] = index
        positions[displaced] = other_index
        swaps.append((index, other_index))
    return tuple(swaps)


def _apply_swaps(
    chromosome: Chromosome, swaps: Sequence[tuple[int, int]]
) -> Chromosome:
    position = list(chromosome)
    for a, b in swaps:
        position[a], position[b] = position[b], position[a]
    return tuple(position)


def _next_velocity(
    current: Chromosome,
    personal_best: Chromosome,
    global_best: Chromosome,
    velocity: tuple[tuple[int, int], ...],
    params: QPSOParams,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], ...]:
    updated = [swap for swap in velocity if float(rng.random()) < params.inertia]
    for target, factor in (
        (personal_best, params.cognitive),
        (global_best, params.social),
    ):
        probability = min(1.0, factor * float(rng.random()))
        for swap in _swaps_between(current, target):
            if float(rng.random()) < probability:
                updated.append(swap)
    if len(updated) > len(current):
        indices = rng.choice(len(updated), size=len(current), replace=False)
        updated = [updated[int(index)] for index in indices]
    return tuple(updated)


def run_qpso(
    instance: Instance,
    params: QPSOParams | None = None,
    seed: int = 0,
    t_station_min: float | None = None,
    config_path: str | Path | None = None,
) -> BaselineResult:
    params = params or QPSOParams()
    n = params.n or default_cluster_count(instance.N)
    rng, queues, evaluation = _queue_evaluation(
        instance,
        n,
        params.sorter,
        seed,
        _service_time(t_station_min, config_path),
        params.FE_max,
    )
    decoder = evaluation.decoder
    genes = tuple(sorted(queues))
    positions: list[Chromosome] = []
    velocities: list[tuple[tuple[int, int], ...]] = []
    personal_best_fitness = []
    for _ in range(min(params.N_p, params.FE_max)):
        position = tuple(int(x) for x in rng.permutation(genes))
        if n > 1:
            a, b = (int(x) for x in rng.choice(n, size=2, replace=False))
            velocity = ((a, b),)
        else:
            velocity = ()
        positions.append(position)
        velocities.append(velocity)
        personal_best_fitness.append(evaluation(position))

    personal_best = list(positions)
    best_index = min(
        range(len(positions)), key=lambda index: personal_best_fitness[index]
    )
    global_best = personal_best[best_index]
    global_best_fitness = personal_best_fitness[best_index]

    while decoder.fe_count < params.FE_max:
        for index in range(len(positions)):
            if decoder.fe_count >= params.FE_max:
                break
            velocity = _next_velocity(
                positions[index],
                personal_best[index],
                global_best,
                velocities[index],
                params,
                rng,
            )
            candidate = _apply_swaps(positions[index], velocity)
            fitness = evaluation(candidate)
            positions[index] = candidate
            velocities[index] = velocity
            if fitness < personal_best_fitness[index]:
                personal_best[index] = candidate
                personal_best_fitness[index] = fitness
            if fitness < global_best_fitness:
                global_best = candidate
                global_best_fitness = fitness
    return _result("qpso", instance, seed, params, n, evaluation)


def _order_crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    rng: np.random.Generator,
    p_c: float,
) -> tuple[Chromosome, Chromosome]:
    if len(parent1) < 2 or float(rng.random()) >= p_c:
        return tuple(parent1), tuple(parent2)
    n = len(parent1)
    a, b = sorted(int(x) for x in rng.choice(n, size=2, replace=False))
    b += 1

    def child(source: Chromosome, other: Chromosome) -> Chromosome:
        result: list[int | None] = [None] * n
        result[a:b] = source[a:b]
        used = set(source[a:b])
        fill = [
            gene
            for gene in (*other[b:], *other[:b])
            if gene not in used
        ]
        positions = list(range(b, n)) + list(range(0, a))
        for index, gene in zip(positions, fill):
            result[index] = gene
        return tuple(result)

    return child(parent1, parent2), child(parent2, parent1)


def run_sga(
    instance: Instance,
    params: SGAParams | None = None,
    seed: int = 0,
    t_station_min: float | None = None,
    config_path: str | Path | None = None,
) -> BaselineResult:
    """Yuan et al. (2021) 风格的站点排列传统 GA，DM/AM 复用队列解码器。"""
    params = params or SGAParams()
    rng = np.random.default_rng(seed)
    queues = {
        station.id: [order for order in instance.orders if order.station == station.id]
        for station in instance.stations
        if station.id != 0
    }
    decoder = Decoder(
        instance,
        queues,
        t_station_min=_service_time(t_station_min, config_path),
        fe_max=params.FE_max,
        sorter=resolve_sorter(params.sorter, rng),
    )
    evaluation = _Evaluation(decoder)
    genes = tuple(sorted(queues))
    population = [
        tuple(int(x) for x in rng.permutation(genes)) for _ in range(params.N_p)
    ]
    fitnesses = []
    try:
        for chromosome in population:
            if decoder.fe_count >= params.FE_max:
                break
            fitnesses.append(evaluation(chromosome))
        population = population[: len(fitnesses)]

        while decoder.fe_count < params.FE_max:
            new_population: list[Chromosome] = []
            while len(new_population) < len(population):
                parent1 = select_parent(population, fitnesses, rng)
                parent2 = select_parent(population, fitnesses, rng)
                new_population.extend(
                    _order_crossover(parent1, parent2, rng, params.p_c)
                )
            new_population = new_population[: len(population)]
            new_population = [
                mutate(chromosome, rng, params.p_m) for chromosome in new_population
            ]
            new_fitnesses = []
            for chromosome in new_population:
                if decoder.fe_count >= params.FE_max:
                    break
                new_fitnesses.append(evaluation(chromosome))
            if len(new_fitnesses) < len(population):
                break
            new_population, new_fitnesses = preserve_elite(
                new_population,
                new_fitnesses,
                evaluation.best_chromosome,
                evaluation.best_result.makespan_min,
            )
            population, fitnesses = new_population, new_fitnesses
    except EvaluationBudgetExhausted:
        pass
    return _result("sga", instance, seed, params, instance.K, evaluation)


def run_sweep(
    instance: Instance,
    params: SweepParams | None = None,
    seed: int = 0,
    t_station_min: float | None = None,
    config_path: str | Path | None = None,
) -> BaselineResult:
    params = params or SweepParams()
    n = params.n or default_cluster_count(instance.N)
    rng, queues, evaluation = _queue_evaluation(
        instance,
        n,
        params.sorter,
        seed,
        _service_time(t_station_min, config_path),
        None,
    )
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    clockwise = bool(int(rng.integers(0, 2)))
    initial = sweep_permutation(_queue_centroids(instance, queues), theta, clockwise)
    two_opt(initial, evaluation, max_passes=params.two_opt_max_passes)
    return _result("sweep", instance, seed, params, n, evaluation)


def run_order_search(
    instance: Instance,
    params: QSAParams | None = None,
    seed: int = 0,
    t_station_min: float | None = None,
    config_path: str | Path | None = None,
) -> BaselineResult:
    """Order-level permutation search used for the clustering-loss experiment."""
    base = params or QSAParams()
    if base.n is not None and base.n != instance.N:
        raise ValueError("order_search requires n=None or n=N")
    params = QSAParams(
        FE_max=base.FE_max,
        n=instance.N,
        initial_temperature=base.initial_temperature,
        cooling_rate=base.cooling_rate,
        sorter=base.sorter,
    )
    result = run_qsa(instance, params, seed, t_station_min, config_path)
    return BaselineResult(
        algorithm="order_search",
        instance_id=result.instance_id,
        seed=result.seed,
        params=result.params,
        n=result.n,
        best_chromosome=result.best_chromosome,
        best_solution=result.best_solution,
        best_times_min=result.best_times_min,
        best_makespan_min=result.best_makespan_min,
        num_evals=result.num_evals,
    )
