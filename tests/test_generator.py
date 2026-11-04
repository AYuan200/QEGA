"""算例生成器单元测试（SPEC §5 + §6-M1）。

核心验收：同一种子两次生成逐字节一致；规模与参数正确；可行性假设全部满足。
"""

import math

import pytest

from qega import constants as C
from qega.generator import (
    GeneratorParams,
    build_fleet,
    fleet_type_counts,
    generate_instance,
    order_feasible,
)
from qega.model import KIND_DELIVERY, KIND_PICKUP, Order, Package, fleet_type_set

SEED = 20261016

# 覆盖 8 种方案组合与规模极端的代表性子集（全量 1000 见 test_instances.py 与冒烟脚本）
REPRESENTATIVE_PARAMS = [
    GeneratorParams(N=200, K=10, M=8, scheme_packages="R", scheme_weight="R", scheme_split="R"),
    GeneratorParams(N=200, K=10, M=8, scheme_packages="U", scheme_weight="U", scheme_split="U"),
    GeneratorParams(N=500, K=15, M=18, scheme_packages="R", scheme_weight="U", scheme_split="R"),
    GeneratorParams(N=500, K=15, M=18, scheme_packages="U", scheme_weight="R", scheme_split="U"),
    GeneratorParams(N=1000, K=20, M=35, scheme_packages="R", scheme_weight="R", scheme_split="U"),
    GeneratorParams(N=1000, K=20, M=35, scheme_packages="U", scheme_weight="U", scheme_split="R"),
    GeneratorParams(N=2000, K=25, M=65, scheme_packages="R", scheme_weight="U", scheme_split="U"),
    GeneratorParams(N=2000, K=30, M=150, scheme_packages="U", scheme_weight="R", scheme_split="R"),
    GeneratorParams(N=200, K=30, M=150, scheme_packages="R", scheme_weight="R", scheme_split="R"),  # 退化组合
]


@pytest.fixture(scope="module", params=REPRESENTATIVE_PARAMS, ids=lambda p: p.instance_id)
def instance(request):
    return generate_instance(request.param, seed=SEED)


class TestReproducibility:
    @pytest.mark.parametrize("params", REPRESENTATIVE_PARAMS[:4], ids=lambda p: p.instance_id)
    def test_same_seed_byte_identical(self, params):
        a = generate_instance(params, seed=SEED)
        b = generate_instance(params, seed=SEED)
        assert a.to_canonical_bytes() == b.to_canonical_bytes()

    def test_different_seed_differs(self):
        params = REPRESENTATIVE_PARAMS[0]
        a = generate_instance(params, seed=SEED)
        b = generate_instance(params, seed=SEED + 1)
        assert a.to_canonical_bytes() != b.to_canonical_bytes()

    def test_serialization_roundtrip(self, instance):
        from qega.model import Instance

        clone = Instance.from_canonical_json(instance.to_canonical_json())
        assert clone.to_canonical_bytes() == instance.to_canonical_bytes()


class TestDimensions:
    def test_order_station_courier_counts(self, instance):
        assert instance.N == _param(instance).N
        assert len(instance.stations) == _param(instance).K + 1
        assert instance.M == _param(instance).M
        assert instance.stations[0].is_depot

    def test_fleet_type_counts(self, instance):
        params = _param(instance)
        expected = dict(fleet_type_counts(params.M))
        actual: dict[str, int] = {}
        for c in instance.couriers:
            actual[c.vehicle_type] = actual.get(c.vehicle_type, 0) + 1
        assert actual == {t: n for t, n in expected.items()}
        # 表 5 默认占比：机型计数与 20/30/50% 一致（最大余数法）
        counts = {t.name: n for t, n in fleet_type_counts(params.M)}
        if params.M == 8:
            assert counts == {"drone": 2, "mini_van": 2, "electric_tricycle": 4}
        if params.M == 150:
            assert counts == {"drone": 30, "mini_van": 45, "electric_tricycle": 75}

    def test_degenerate_flag(self, instance):
        params = _param(instance)
        expected = params.N == C.DEGENERATE_N and params.M == C.DEGENERATE_M
        assert instance.degenerate == expected


class TestStations:
    def test_within_radius(self, instance):
        for s in instance.stations[1:]:
            assert 0.0 < math.hypot(s.x, s.y) <= C.MAP_RADIUS_KM + 1e-12

    def test_min_interstation_distance(self, instance):
        assert instance.min_interstation_distance() >= C.MIN_STATION_DISTANCE_KM - 1e-12

    def test_distance_matrices(self, instance):
        n = len(instance.stations)
        for vt in fleet_type_set(instance.couriers):
            d = instance.distance_matrix(vt)
            assert len(d) == n and all(len(row) == n for row in d)
            # 无人机读 D′，地面车辆读 D，同一来源
            expected = instance.dist_drone if vt.aerial else instance.dist_ground
            assert d is expected


class TestOrders:
    def test_package_bounds(self, instance):
        for o in instance.orders:
            assert C.PACKAGES_PER_ORDER_MIN <= o.num_packages <= C.PACKAGES_PER_ORDER_MAX
            for p in o.packages:
                assert C.PACKAGE_WEIGHT_MIN_KG - 1e-12 <= p.weight_kg <= C.PACKAGE_WEIGHT_MAX_KG + 1e-12

    def test_split_consistency(self, instance):
        for o in instance.orders:
            assert o.n_delivery + o.n_pickup == o.num_packages
            for p in o.packages:
                if p.kind == KIND_DELIVERY:
                    assert p.source == 0 and p.dest == o.station
                else:
                    assert p.source == o.station and p.dest == 0

    def test_feasibility_m1_all_orders(self, instance):
        """SPEC §6-M1：每个订单至少一种车型载重+航程双重可行（按实际 D/D′）。"""
        types = fleet_type_set(instance.couriers)
        for o in instance.orders:
            assert order_feasible(o, instance.dist_ground, instance.dist_drone, types), (
                f"order {o.id} infeasible in {instance.instance_id}"
            )

    def test_sampling_schemes_applied(self):
        # Type-R 包裹数来自截断 Poisson(1.4)：固定种子下 N=2000 中 5 包裹订单应稀少
        r = generate_instance(
            GeneratorParams(2000, 10, 8, "R", "R", "R"), seed=SEED
        )
        five_r = sum(1 for o in r.orders if o.num_packages == 5)
        assert five_r < 0.10 * r.N  # Poisson(1.4) 截断后 P(X=5)≈0.3%
        # Type-U 离散均匀：五个取值应都大量出现
        u = generate_instance(
            GeneratorParams(2000, 10, 8, "U", "U", "U"), seed=SEED
        )
        counts = [0] * 6
        for o in u.orders:
            counts[o.num_packages] += 1
        assert all(counts[k] > 0.10 * u.N for k in range(1, 6))


class TestFleet:
    @pytest.mark.parametrize(
        "M,exp",
        [
            (8, (2, 2, 4)),
            (18, (4, 5, 9)),
            (35, (7, 11, 17)),
            (65, (13, 20, 32)),
            (150, (30, 45, 75)),
        ],
    )
    def test_table5_proportions(self, M, exp):
        counts = tuple(n for _, n in fleet_type_counts(M))
        assert counts == exp
        assert len(build_fleet(M)) == M

    def test_drone_ratio_ablation(self):
        # SPEC M8 消融接口：0% 无人机
        counts = dict(fleet_type_counts(8, drone_ratio=0.0))
        assert counts[C.DRONE] == 0
        assert sum(counts.values()) == 8
        # 40% 无人机
        counts = dict(fleet_type_counts(8, drone_ratio=0.4))
        assert counts[C.DRONE] == 3
        assert sum(counts.values()) == 8

    def test_courier_ids_consecutive(self):
        fleet = build_fleet(18)
        assert [c.id for c in fleet] == list(range(1, 19))


class TestRoadFactorMatrix:
    """SPEC §2 修订：ρ~U[1.2,1.4] 对称、D′<D 对所有弧成立、同种子重生成逐字节一致。"""

    def test_rho_symmetric_and_in_range(self, instance):
        rho = instance.road_factor_matrix()
        n = len(instance.stations)
        for i in range(n):
            assert rho[i][i] == 1.0
            for j in range(i + 1, n):
                assert rho[i][j] == rho[j][i]
                assert C.ROAD_FACTOR_MIN - 1e-9 <= rho[i][j] <= C.ROAD_FACTOR_MAX + 1e-9

    def test_drone_distance_strictly_less_than_ground(self, instance):
        n = len(instance.stations)
        for i in range(n):
            for j in range(i + 1, n):
                assert instance.dist_drone[i][j] < instance.dist_ground[i][j]

    def test_matrices_byte_identical_on_regeneration(self):
        params = REPRESENTATIVE_PARAMS[0]
        a = generate_instance(params, seed=SEED)
        b = generate_instance(params, seed=SEED)
        assert a.dist_ground == b.dist_ground
        assert a.dist_drone == b.dist_drone
        assert a.road_factor_matrix() == b.road_factor_matrix()
        assert a.to_canonical_bytes() == b.to_canonical_bytes()

    def test_matrices_persisted_in_canonical_json(self, instance):
        from qega.model import Instance

        clone = Instance.from_canonical_json(instance.to_canonical_json())
        assert clone.dist_ground == instance.dist_ground
        assert clone.dist_drone == instance.dist_drone


class TestFeasibilityUsesActualMatrix:
    """M1 航程判定必须读算例实际距离矩阵 D/D′，禁止名义 1.3·E 近似。"""

    @staticmethod
    def _order(weight=1.0):
        return Order(1, 1, (Package("p", KIND_DELIVERY, weight, 0, 1),))

    def test_ground_actual_matrix_decides(self):
        # E=29.5 km：名义 1.3·E=38.35（往返 76.7 ≤ 80，名义可行）；
        # 实际 ρ=1.39 → D=41.005（往返 82.01 > 80，实际不可行）
        order = self._order()
        dist_drone = [[0.0, 29.5], [29.5, 0.0]]
        dist_ground = [[0.0, 29.5 * 1.39], [29.5 * 1.39, 0.0]]
        assert not order_feasible(order, dist_ground, dist_drone, [C.ELECTRIC_TRICYCLE])
        assert order_feasible(order, dist_ground, dist_drone, [C.MINI_VAN])

    def test_drone_range_uses_euclidean(self):
        order = self._order()
        dist_ground = [[0.0, 99.0], [99.0, 0.0]]  # 地面距离不影响无人机判定
        assert not order_feasible(order, dist_ground, [[0.0, 10.4], [10.4, 0.0]], [C.DRONE])
        assert order_feasible(order, dist_ground, [[0.0, 9.9], [9.9, 0.0]], [C.DRONE])


def _param(instance):
    return next(p for p in REPRESENTATIVE_PARAMS if p.instance_id == instance.instance_id)
