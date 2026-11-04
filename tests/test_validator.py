"""约束校验器测试（SPEC §2 式 6/8/9/10/11–13）。"""

from qega.decoder import Solution, Stop, Trip
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
)
from qega.validator import assert_valid, validate_solution

import pytest


def _package(pid, kind, weight, station):
    return Package(
        pid,
        kind,
        weight,
        0 if kind == KIND_DELIVERY else station,
        station if kind == KIND_DELIVERY else 0,
    )


def _instance(capacity=10.0, range_km=20.0):
    stations = [Station(0, 0, 0), Station(1, 5, 0), Station(2, -4, 0)]
    dist = euclidean_matrix(stations)
    vt = VehicleType("v", capacity, 30.0, range_km, False)
    orders = [
        Order(1, 1, (_package("p1d", KIND_DELIVERY, 6.0, 1), _package("p1p", KIND_PICKUP, 4.0, 1))),
        Order(2, 2, (_package("p2p", KIND_PICKUP, 3.0, 2),)),
    ]
    return Instance("validation", stations, orders, [Courier(1, vt)], dist, dist)


def _valid_solution():
    return Solution({1: [Trip((Stop(1, (1,)), Stop(2, (2,))))]})


def test_valid_solution_has_no_violations():
    inst = _instance()
    assert validate_solution(inst, _valid_solution()) == []
    assert_valid(inst, _valid_solution())


def test_range_violation_reports_trip_station_and_numeric_values():
    inst = _instance(range_km=8.0)
    violations = validate_solution(inst, _valid_solution())
    range_hits = [v for v in violations if v.equation == "8"]
    assert range_hits
    hit = range_hits[0]
    assert hit.courier_id == 1 and hit.trip == 1 and hit.station == 2
    assert hit.value == 18.0 and hit.limit == 8.0
    text = str(hit)
    assert "trip=1" in text and "station=2" in text
    assert "value=18" in text and "limit=8" in text


def test_trip_origin_and_departure_payload_are_checked():
    inst = _instance(capacity=5.0)
    solution = Solution({1: [Trip((Stop(1, (1,)),), start=1)]})
    violations = validate_solution(inst, solution)
    eq6 = next(v for v in violations if v.equation == "6")
    eq9 = next(v for v in violations if v.equation == "9")
    assert (eq6.trip, eq6.station, eq6.value, eq6.limit) == (1, 1, 1.0, 0.0)
    assert (eq9.trip, eq9.station, eq9.value, eq9.limit) == (1, 0, 6.0, 5.0)


def test_post_stop_payload_is_checked_after_unload_then_load():
    inst = _instance(capacity=10.0, range_km=100.0)
    # 第一站装 8 kg，第二站再装 5 kg：卸货后装货得到 13 kg > 10 kg。
    o1 = Order(3, 1, (_package("p3", KIND_PICKUP, 8.0, 1),))
    o2 = Order(4, 2, (_package("p4", KIND_PICKUP, 5.0, 2),))
    inst.orders = [o1, o2]
    solution = Solution({1: [Trip((Stop(1, (3,)), Stop(2, (4,))))]})
    violations = validate_solution(inst, solution)
    hit = next(v for v in violations if v.equation == "10")
    assert (hit.trip, hit.station, hit.value, hit.limit) == (1, 2, 13.0, 10.0)


def test_delivery_pickup_locations_and_exactly_once_coverage():
    inst = _instance()
    # 订单 1 被放在错误站点并重复，订单 2 缺失。
    solution = Solution({1: [Trip((Stop(2, (1, 1)),))]})
    violations = validate_solution(inst, solution)
    assert any(v.equation == "11" for v in violations)  # 配送目的站
    assert any(v.equation == "12" for v in violations)  # 取件源站
    counts = {v.message: v for v in violations if v.equation == "13"}
    assert any(v.value == 2.0 and v.limit == 1.0 for v in counts.values())
    assert any(v.value == 0.0 and v.limit == 1.0 for v in counts.values())


def test_assert_valid_raises_with_all_location_fields():
    inst = _instance(range_km=1.0)
    with pytest.raises(ValueError, match=r"equation=8.*trip=1.*station=2.*value="):
        assert_valid(inst, _valid_solution())
