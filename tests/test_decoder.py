"""DM/AM 解码测试（SPEC §3.2–3.4，重点验证修订版 skip）。"""

from collections import deque

import pytest

from qega.decoder import Decoder, EvaluationBudgetExhausted, am
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


def _instance(range_km=10.0):
    stations = [Station(0, 0, 0), Station(1, 6, 0), Station(2, 0, 4)]
    dist = euclidean_matrix(stations)
    vt = VehicleType("limited", 20.0, 60.0, range_km, False)
    orders = [
        Order(1, 1, (Package("p1", KIND_DELIVERY, 1.0, 0, 1),)),
        Order(2, 2, (Package("p2", KIND_DELIVERY, 1.0, 0, 2),)),
    ]
    return Instance("am_skip", stations, orders, [Courier(1, vt)], dist, dist)


def test_am_skips_over_range_order_instead_of_breaking():
    inst = _instance(range_km=10.0)
    courier = inst.couriers[0]
    queues = {1: deque(inst.orders)}

    trip = am(inst, courier, queues, start_queue=1)

    # 首单往返 12 km 超航程；若 break 会空趟，skip 则继续装入第二单（8 km）。
    assert trip.station_ids == (2,)
    assert trip.order_ids == (2,)
    assert [o.id for o in queues[1]] == [1]


def test_am_checks_both_range_and_payload_and_preserves_rejected_order():
    inst = _instance(range_km=100.0)
    courier = inst.couriers[0]
    heavy = Order(3, 2, (Package("p3", KIND_DELIVERY, 21.0, 0, 2),))
    queues = {1: deque([heavy, inst.orders[1]])}

    trip = am(inst, courier, queues, start_queue=1)

    assert trip.order_ids == (2,)
    assert [o.id for o in queues[1]] == [3]


def test_fe_is_counted_once_at_decoder_entry_and_budget_is_enforced():
    inst = _instance(range_km=100.0)
    decoder = Decoder(inst, {1: list(inst.orders)}, t_station_min=0.0, fe_max=1)

    result = decoder.decode((1,))
    assert result.makespan_min > 0.0
    assert decoder.fe_count == 1
    with pytest.raises(EvaluationBudgetExhausted):
        decoder.decode((1,))
    assert decoder.fe_count == 1


def test_decoder_solution_contains_every_order_exactly_once():
    inst = _instance(range_km=100.0)
    result = Decoder(inst, {1: list(inst.orders)}, t_station_min=0.0).decode((1,))
    assigned = [
        order_id
        for trips in result.solution.trips.values()
        for trip in trips
        for stop in trip.stops
        for order_id in stop.order_ids
    ]
    assert sorted(assigned) == [1, 2]


def test_am_accumulates_later_deliveries_in_earlier_post_stop_loads():
    inst = _instance(range_km=100.0)
    orders = [
        Order(3, 1, (Package("p3", "pickup", 12.0, 1, 0),)),
        Order(4, 2, (Package("p4", KIND_DELIVERY, 6.0, 0, 2),)),
        Order(5, 2, (Package("p5", KIND_DELIVERY, 6.0, 0, 2),)),
    ]
    inst.orders = orders
    queues = {1: deque(orders)}

    trip = am(inst, inst.couriers[0], queues, start_queue=1)

    # 第二站首单到达前，第一站后的载重为 12+6=18；再装第二单会变成 24>20。
    assert trip.order_ids == (3, 4)
    assert [order.id for order in queues[1]] == [5]


def test_dm_follows_chromosome_queue_order():
    inst = _instance(range_km=100.0)
    queues = {1: [inst.orders[0]], 2: [inst.orders[1]]}
    decoder = Decoder(inst, queues, t_station_min=0.0)

    forward = decoder.decode((1, 2))
    reverse = decoder.decode((2, 1))

    assert forward.solution.trips[1][0].order_ids == (1, 2)
    assert reverse.solution.trips[1][0].order_ids == (2, 1)
    assert forward.solution.to_dict() != reverse.solution.to_dict()


def test_dm_retries_a_nonempty_queue_with_the_next_courier():
    import numpy as np

    from qega.generator import GeneratorParams, generate_instance
    from qega.qega import default_cluster_count, kmeans_queues
    from qega.validator import validate_solution

    inst = generate_instance(GeneratorParams(N=20, K=5, M=5), seed=7)
    rng = np.random.default_rng(11)
    queues = kmeans_queues(inst, default_cluster_count(inst.N), rng)
    result = Decoder(inst, queues, t_station_min=5.0, fe_max=1).decode(
        tuple(sorted(queues))
    )
    assert validate_solution(inst, result.solution) == []
