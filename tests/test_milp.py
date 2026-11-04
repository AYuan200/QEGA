"""Tiny exact-model regression test (not a large-scale benchmark)."""

import pytest

from qega.milp import solve_milp
from qega.model import KIND_DELIVERY, Courier, Instance, Order, Package, Station, VehicleType, euclidean_matrix


def test_single_order_lower_bound_matches_closed_form():
    stations = [Station(0, 0, 0), Station(1, 2, 0)]
    distance = euclidean_matrix(stations)
    vehicle = VehicleType("van", 10, 60, 100, False)
    order = Order(1, 1, (Package("p1", KIND_DELIVERY, 1, 0, 1),))
    instance = Instance("tiny", stations, [order], [Courier(1, vehicle)], distance, distance)
    result = solve_milp(instance, t_station_min=5.0, time_limit_s=30.0)
    assert result.lower_bound_min == pytest.approx(9.0, abs=1e-6)
    assert result.incumbent_min == pytest.approx(9.0, abs=1e-6)
