"""队内排序测试（SPEC §3.3）。"""

import numpy as np

from qega.model import (
    KIND_DELIVERY,
    Courier,
    Instance,
    Order,
    Package,
    Station,
    VehicleType,
    euclidean_matrix,
)
from qega.sorters import dps_sort, mks_sort, rds_sort


def _order(order_id, station, weight):
    return Order(
        order_id,
        station,
        (Package(f"p{order_id}", KIND_DELIVERY, weight, 0, station),),
    )


def _instance(capacity=100.0):
    stations = [Station(0, 0, 0), Station(1, 5, 0), Station(2, 2, 0)]
    dist = euclidean_matrix(stations)
    vt = VehicleType("v", capacity, 30.0, 100.0, False)
    orders = [_order(1, 1, 1.0), _order(2, 2, 2.0), _order(3, 2, 3.0)]
    return Instance("sort", stations, orders, [Courier(1, vt)], dist, dist)


def test_mks_distance_ascending_then_total_weight_descending():
    inst = _instance()
    ordered = mks_sort(inst.orders, inst)
    assert [o.id for o in ordered] == [3, 2, 1]


def test_rds_is_a_seeded_permutation():
    inst = _instance()
    first = rds_sort(inst.orders, inst, np.random.default_rng(1))
    second = rds_sort(inst.orders, inst, np.random.default_rng(1))
    assert [order.id for order in first] == [order.id for order in second]
    assert sorted(order.id for order in first) == [1, 2, 3]


def test_dps_places_a_fully_loaded_knapsack_batch_first():
    inst = _instance(capacity=5.0)
    ordered = dps_sort(inst.orders, inst)
    assert [order.id for order in ordered] == [2, 3, 1]
    assert sum(order.total_weight_kg for order in ordered[:2]) == 5.0
    assert [order.id for order in dps_sort(inst.orders, inst)] == [2, 3, 1]
