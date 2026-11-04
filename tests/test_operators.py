"""遗传算子单元测试（SPEC §3.5–3.6）。"""

import numpy as np
import pytest

from qega.operators import (
    crossover,
    crossover_at,
    mutate,
    preserve_elite,
    selection_probabilities,
)


def test_crossover_spec_vector_position_by_position():
    p1 = (3, 4, 5, 1, 2, 6)
    p2 = (6, 4, 2, 1, 3, 5)

    c1, c2 = crossover_at(p1, p2, start1=1, start2=3, length=3)

    assert c1 == (4, 1, 3, 5, 2, 6)
    assert c2 == (6, 3, 2, 4, 5, 1)
    assert list(c1) == [4, 1, 3, 5, 2, 6]
    assert list(c2) == [6, 3, 2, 4, 5, 1]


def test_random_crossover_returns_permutations_and_copies_parents():
    p1 = tuple(range(1, 9))
    p2 = tuple(reversed(p1))
    c1, c2 = crossover(p1, p2, np.random.default_rng(7), p_c=1.0)
    assert sorted(c1) == sorted(p1)
    assert sorted(c2) == sorted(p1)

    c1, c2 = crossover(p1, p2, np.random.default_rng(7), p_c=0.0)
    assert c1 == p1 and c2 == p2
    assert c1 is not p1 and c2 is not p2


def test_selection_probabilities_follow_equation_17():
    probs = selection_probabilities([1.0, 2.0, 3.0])
    assert probs == pytest.approx([2.0 / 3.0, 1.0 / 3.0, 0.0])
    assert selection_probabilities([2.0, 2.0, 2.0]) == pytest.approx([1 / 3] * 3)


def test_mutation_and_elite_preservation():
    original = (1, 2, 3, 4)
    unchanged = mutate(original, np.random.default_rng(1), p_m=0.0)
    changed = mutate(original, np.random.default_rng(1), p_m=1.0)
    assert unchanged == original and unchanged is not original
    assert sorted(changed) == sorted(original)
    assert sum(a != b for a, b in zip(changed, original)) == 2

    population = [(1, 2, 3), (3, 2, 1)]
    fitness = [5.0, 6.0]
    elite = (2, 1, 3)
    new_population, new_fitness = preserve_elite(population, fitness, elite, 4.0)
    assert new_population[1] == elite
    assert new_fitness[1] == 4.0


def test_crossover_rejects_invalid_segments():
    with pytest.raises(ValueError):
        crossover_at((1, 2), (1, 2), 0, 0, 0)
    with pytest.raises(ValueError):
        crossover_at((1, 2), (1, 3), 0, 0, 1)
