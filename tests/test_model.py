"""数据结构单元测试：Station / Package / Order / Courier / Instance。"""

import pytest

from qega.model import (
    KIND_DELIVERY,
    KIND_PICKUP,
    Courier,
    Instance,
    Order,
    Package,
    Station,
    VehicleType,
    euclidean_matrix,
    fleet_type_set,
)

VT_A = VehicleType("a", capacity_kg=10.0, speed_kmh=30.0, range_km=100.0, aerial=False)
VT_B = VehicleType("b", capacity_kg=5.0, speed_kmh=60.0, range_km=20.0, aerial=True)


def _pkg(pid="p1_1", kind=KIND_DELIVERY, w=1.0, station=1):
    return Package(
        id=pid,
        kind=kind,
        weight_kg=w,
        source=0 if kind == KIND_DELIVERY else station,
        dest=station if kind == KIND_DELIVERY else 0,
    )


def _instance():
    stations = [Station(0, 0.0, 0.0), Station(1, 3.0, 4.0), Station(2, 0.0, 8.0)]
    dist_drone = euclidean_matrix(stations)
    dist_ground = [[x * 1.3 for x in row] for row in dist_drone]
    orders = [Order(1, 1, (_pkg(), _pkg("p1_2", KIND_PICKUP, 2.0)))]
    couriers = [Courier(1, VT_A), Courier(2, VT_B), Courier(3, VT_A)]
    return Instance(
        "t0", stations, orders, couriers, dist_ground, dist_drone,
        scheme_code="RUU", seed=7,
    )


class TestPackage:
    def test_delivery_must_originate_at_depot(self):
        with pytest.raises(ValueError):
            Package("x", KIND_DELIVERY, 1.0, source=2, dest=1)

    def test_pickup_must_return_to_depot(self):
        with pytest.raises(ValueError):
            Package("x", KIND_PICKUP, 1.0, source=1, dest=2)

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError):
            Package("x", "express", 1.0, source=0, dest=1)


class TestOrder:
    def test_station_consistency_enforced(self):
        with pytest.raises(ValueError):
            Order(1, 2, (_pkg(station=1),))  # 包裹目的站 ≠ 订单站点

    def test_depot_station_rejected(self):
        with pytest.raises(ValueError):
            Order(1, 0, (_pkg(),))

    def test_empty_order_rejected(self):
        with pytest.raises(ValueError):
            Order(1, 1, ())

    def test_weight_aggregation(self):
        o = Order(1, 1, (_pkg(w=3.0), _pkg("p2", KIND_DELIVERY, 4.0), _pkg("p3", KIND_PICKUP, 2.5)))
        assert o.n_delivery == 2 and o.n_pickup == 1 and o.num_packages == 3
        assert o.delivery_weight_kg == 7.0
        assert o.pickup_weight_kg == 2.5
        assert o.total_weight_kg == 9.5


class TestInstance:
    def test_dimensions(self):
        inst = _instance()
        assert (inst.N, inst.K, inst.M) == (1, 2, 3)

    def test_depot_must_be_first(self):
        with pytest.raises(ValueError):
            Instance("bad", [Station(1, 0, 0), Station(0, 0, 0)], [], [], [], [])

    def test_asymmetric_matrix_rejected(self):
        inst = _instance()
        bad = [row[:] for row in inst.dist_ground]
        bad[0][1] += 1.0
        with pytest.raises(ValueError):
            Instance("bad", inst.stations, inst.orders, inst.couriers, bad, inst.dist_drone)

    def test_wrong_shape_matrix_rejected(self):
        inst = _instance()
        with pytest.raises(ValueError):
            Instance("bad", inst.stations, inst.orders, inst.couriers, [[0.0]], inst.dist_drone)

    def test_type_label(self):
        assert _instance().type_label == "Mixed-RUU"
        inst = _instance()
        inst.scheme_code = "RRR"
        assert inst.type_label == "Type-R"
        inst.scheme_code = "UUU"
        assert inst.type_label == "Type-U"

    def test_serialization_roundtrip_byte_identical(self):
        inst = _instance()
        clone = Instance.from_canonical_json(inst.to_canonical_json())
        assert clone.to_canonical_bytes() == inst.to_canonical_bytes()
        assert clone.N == inst.N and clone.K == inst.K and clone.M == inst.M
        assert clone.degenerate == inst.degenerate and clone.seed == inst.seed
        assert clone.dist_ground == inst.dist_ground
        assert clone.dist_drone == inst.dist_drone

    def test_fleet_type_set_dedup_preserves_order(self):
        types = fleet_type_set(_instance().couriers)
        assert [t.name for t in types] == ["a", "b"]
