"""步骤 3 基线与排序变体的 N=50 冒烟、可行性和可复现性测试。"""

import pytest

from qega.baselines import (
    QPSOParams,
    QSAParams,
    SGAParams,
    SweepParams,
    run_qpso,
    run_qsa,
    run_sga,
    run_sweep,
)
from qega.generator import GeneratorParams, generate_instance
from qega.qega import QEGAParams, run_qega
from qega.validator import validate_solution


def _instance():
    return generate_instance(
        GeneratorParams(
            N=50,
            K=10,
            M=8,
            scheme_packages="R",
            scheme_weight="R",
            scheme_split="R",
        ),
        seed=20261017,
    )


@pytest.mark.parametrize(
    ("algorithm", "runner", "params", "expected_evals"),
    [
        ("qsa", run_qsa, QSAParams(FE_max=500), 500),
        ("qpso", run_qpso, QPSOParams(FE_max=500), 500),
        ("sga", run_sga, SGAParams(FE_max=500), 500),
        ("sweep", run_sweep, SweepParams(), None),
    ],
)
def test_baseline_smoke_feasible_and_byte_identical(
    algorithm, runner, params, expected_evals
):
    first = runner(_instance(), params=params, seed=12345, t_station_min=5.0)
    second = runner(_instance(), params=params, seed=12345, t_station_min=5.0)

    assert first.algorithm == algorithm
    assert first.best_makespan_min > 0.0
    assert validate_solution(_instance(), first.best_solution) == []
    assert sorted(first.best_chromosome) == list(range(1, first.n + 1))
    if expected_evals is not None:
        assert first.num_evals == expected_evals
    else:
        assert first.num_evals > 0
    assert first.to_canonical_bytes() == second.to_canonical_bytes()


@pytest.mark.parametrize("sorter", ["mks", "rds", "dps"])
def test_sorting_variant_smoke_feasible_and_byte_identical(sorter):
    params = QEGAParams(FE_max=500, sorter=sorter)
    first = run_qega(_instance(), params=params, seed=12345, t_station_min=5.0)
    second = run_qega(_instance(), params=params, seed=12345, t_station_min=5.0)

    assert first.best_makespan_min > 0.0
    assert first.num_evals == 500
    assert validate_solution(_instance(), first.best_solution) == []
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
