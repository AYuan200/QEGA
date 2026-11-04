"""QEGA 主程序冒烟与可复现性测试。"""

from qega.generator import GeneratorParams, generate_instance
from qega.qega import QEGAParams, default_cluster_count, kmeans_queues, run_qega
from qega.validator import validate_solution

import numpy as np


def test_default_cluster_count_scales_with_order_count():
    # Final manuscript: n = clamp(round(N/50), 15, 25)
    assert default_cluster_count(100) == 15
    assert default_cluster_count(500) == 15
    assert default_cluster_count(1000) == 20
    assert default_cluster_count(2000) == 25
    assert default_cluster_count(5000) == 25
    assert default_cluster_count(1) == 1


def test_kmeans_queues_cover_orders_once():
    inst = generate_instance(GeneratorParams(N=100, K=10, M=18), seed=901)
    queues = kmeans_queues(inst, 4, np.random.default_rng(7))
    assert sorted(queues) == [1, 2, 3, 4]
    assigned = [o.id for queue in queues.values() for o in queue]
    assert sorted(assigned) == list(range(1, 101))


def test_n100_smoke_feasible_and_same_seed_byte_identical():
    params = GeneratorParams(N=100, K=10, M=18, scheme_packages="R", scheme_weight="R", scheme_split="R")
    inst1 = generate_instance(params, seed=20261016)
    inst2 = generate_instance(params, seed=20261016)
    algorithm = QEGAParams(FE_max=2000)

    result1 = run_qega(inst1, params=algorithm, seed=12345)
    result2 = run_qega(inst2, params=algorithm, seed=12345)

    assert result1.num_evals == 2000
    assert result1.best_makespan_min > 0.0
    assert validate_solution(inst1, result1.best_solution) == []
    assert result1.to_canonical_bytes() == result2.to_canonical_bytes()


def test_init_2opt_passes_bounds_initialization_cost():
    """SPEC §6-M12：限扫轮数后初始化不得耗尽 FE 预算，GA 主循环必须执行。"""
    from qega.operators import initialize_population, two_opt

    # two_opt 单元级：max_passes=1 时评估次数 ≤ 1 + C(n,2)
    evals = 0
    n = 8

    def counting_evaluate(chromosome):
        nonlocal evals
        evals += 1
        return float(sum(chromosome))

    two_opt(tuple(range(n)), counting_evaluate, max_passes=1)
    assert evals <= 1 + n * (n - 1) // 2

    # 端到端：N=100（n=2）+ alpha=1.0 时初始化成本有界，预算大头留给 GA
    inst = generate_instance(GeneratorParams(N=100, K=10, M=18), seed=777)
    result = run_qega(
        inst,
        params=QEGAParams(FE_max=2000, alpha=1.0, init_2opt_passes=2),
        seed=5,
    )
    assert result.num_evals == 2000
    assert validate_solution(inst, result.best_solution) == []

    # init_2opt_passes 校验
    import pytest

    with pytest.raises(ValueError):
        QEGAParams(init_2opt_passes=-1)
