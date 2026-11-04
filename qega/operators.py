"""QEGA 初始化与遗传算子（SPEC §3.5–3.6）。"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Mapping, Sequence

import numpy as np

Chromosome = tuple[int, ...]


def _copy(chromosome: Sequence[int]) -> Chromosome:
    return tuple(list(chromosome))


def _repair_after_swap(original: Sequence[int], swapped: list[int]) -> Chromosome:
    missing = [gene for gene in original if gene not in swapped]
    counts = Counter(swapped)
    last_position = {
        gene: max(i for i, value in enumerate(swapped) if value == gene)
        for gene, count in counts.items()
        if count > 1
    }
    missing_index = 0
    for i, gene in enumerate(tuple(swapped)):
        if counts[gene] > 1 and i != last_position[gene]:
            swapped[i] = missing[missing_index]
            missing_index += 1
    return tuple(swapped)


def crossover_at(
    parent1: Sequence[int],
    parent2: Sequence[int],
    start1: int,
    start2: int,
    length: int,
) -> tuple[Chromosome, Chromosome]:
    """互换指定等长子序列，再按原父代缺失基因顺序修复。"""
    n = len(parent1)
    if len(parent2) != n or len(set(parent1)) != n or set(parent1) != set(parent2):
        raise ValueError("parents must be permutations of the same genes")
    if length <= 0:
        raise ValueError("length must be positive")
    if not 0 <= start1 <= n - length or not 0 <= start2 <= n - length:
        raise ValueError("crossover segment is out of bounds")

    child1 = list(parent1)
    child2 = list(parent2)
    child1[start1:start1 + length] = parent2[start2:start2 + length]
    child2[start2:start2 + length] = parent1[start1:start1 + length]
    return _repair_after_swap(parent1, child1), _repair_after_swap(parent2, child2)


def crossover(
    parent1: Sequence[int],
    parent2: Sequence[int],
    rng: np.random.Generator,
    p_c: float = 1.0,
) -> tuple[Chromosome, Chromosome]:
    if not 0.0 <= p_c <= 1.0:
        raise ValueError("p_c must be in [0, 1]")
    if len(parent1) < 2 or float(rng.random()) >= p_c:
        return _copy(parent1), _copy(parent2)
    n = len(parent1)
    length = int(rng.integers(1, n + 1))
    start1 = int(rng.integers(0, n - length + 1))
    start2 = int(rng.integers(0, n - length + 1))
    return crossover_at(parent1, parent2, start1, start2, length)


def selection_probabilities(fitnesses: Sequence[float]) -> list[float]:
    """式 17；适应度全部相同时退化为均匀分布。"""
    if not fitnesses:
        raise ValueError("fitnesses must not be empty")
    worst = max(fitnesses)
    weights = [worst - fitness for fitness in fitnesses]
    denominator = sum(weights)
    if denominator <= 0.0:
        return [1.0 / len(fitnesses)] * len(fitnesses)
    return [weight / denominator for weight in weights]


def select_parent(
    population: Sequence[Chromosome],
    fitnesses: Sequence[float],
    rng: np.random.Generator,
) -> Chromosome:
    probabilities = selection_probabilities(fitnesses)
    index = int(rng.choice(len(population), p=probabilities))
    return population[index]


def mutate(
    chromosome: Sequence[int], rng: np.random.Generator, p_m: float
) -> Chromosome:
    if not 0.0 <= p_m <= 1.0:
        raise ValueError("p_m must be in [0, 1]")
    child = list(chromosome)
    if len(child) >= 2 and float(rng.random()) < p_m:
        a, b = (int(x) for x in rng.choice(len(child), size=2, replace=False))
        child[a], child[b] = child[b], child[a]
    return tuple(child)


def preserve_elite(
    population: Sequence[Chromosome],
    fitnesses: Sequence[float],
    elite: Chromosome,
    elite_fitness: float,
) -> tuple[list[Chromosome], list[float]]:
    new_population = list(population)
    new_fitnesses = list(fitnesses)
    if not new_population:
        raise ValueError("population must not be empty")
    if min(new_fitnesses) > elite_fitness:
        worst = max(range(len(new_population)), key=lambda i: new_fitnesses[i])
        new_population[worst] = _copy(elite)
        new_fitnesses[worst] = elite_fitness
    return new_population, new_fitnesses


def sweep_permutation(
    centroids: Mapping[int, tuple[float, float]],
    theta: float,
    clockwise: bool,
) -> Chromosome:
    def key(item: tuple[int, tuple[float, float]]) -> tuple[float, int]:
        queue_id, (x, y) = item
        angle = math.atan2(y, x) % (2.0 * math.pi)
        relative = (
            (theta - angle) % (2.0 * math.pi)
            if clockwise
            else (angle - theta) % (2.0 * math.pi)
        )
        return relative, queue_id

    return tuple(queue_id for queue_id, _ in sorted(centroids.items(), key=key))


def two_opt(
    chromosome: Sequence[int],
    evaluate: Callable[[Chromosome], float],
    max_passes: int | None = None,
) -> Chromosome:
    """两位置交换的 best-improvement 局部搜索。

    max_passes：完整扫描轮数上限（一轮 = 全部 C(n,2) 对偶各评估一次）；
    None 表示不限制（论文算法 5 原版行为）。限制轮数是修订行为（SPEC §6-M12），
    防止初始化阶段在统一 FE 预算下耗尽全部评估次数。
    """
    if max_passes is not None and max_passes < 0:
        raise ValueError("max_passes must be non-negative or None")
    current = _copy(chromosome)
    best_fitness = evaluate(current)
    passes = 0
    while max_passes is None or passes < max_passes:
        passes += 1
        best_pair: tuple[int, int] | None = None
        for a in range(len(current) - 1):
            for b in range(a + 1, len(current)):
                candidate = list(current)
                candidate[a], candidate[b] = candidate[b], candidate[a]
                candidate_tuple = tuple(candidate)
                fitness = evaluate(candidate_tuple)
                if fitness < best_fitness - 1e-12:
                    best_fitness = fitness
                    best_pair = (a, b)
        if best_pair is None:
            return current
        current_list = list(current)
        a, b = best_pair
        current_list[a], current_list[b] = current_list[b], current_list[a]
        current = tuple(current_list)
    return current


def initialize_population(
    centroids: Mapping[int, tuple[float, float]],
    population_size: int,
    alpha: float,
    rng: np.random.Generator,
    evaluate: Callable[[Chromosome], float],
    two_opt_max_passes: int | None = None,
) -> list[Chromosome]:
    """Generate ``alpha`` sweep/2-opt individuals and the rest randomly."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    genes = tuple(sorted(centroids))
    population: list[Chromosome] = []
    sweep_count = min(population_size, math.ceil(alpha * population_size))
    for _ in range(sweep_count):
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        clockwise = bool(int(rng.integers(0, 2)))
        initial = sweep_permutation(centroids, theta, clockwise)
        population.append(two_opt(initial, evaluate, max_passes=two_opt_max_passes))
    while len(population) < population_size:
        population.append(tuple(int(x) for x in rng.permutation(genes)))
    return population
