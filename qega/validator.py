"""方案约束校验器（SPEC §2 式 6/8/9/10/11–13）。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .decoder import Solution
from .model import DEPOT_ID, Instance, KIND_DELIVERY, KIND_PICKUP

_EPS = 1e-9


@dataclass(frozen=True)
class Violation:
    equation: str
    courier_id: int
    trip: int
    station: int
    value: float | None
    limit: float | None
    message: str

    def __str__(self) -> str:
        value = "None" if self.value is None else f"{self.value:.12g}"
        limit = "None" if self.limit is None else f"{self.limit:.12g}"
        return (
            f"equation={self.equation} courier={self.courier_id} trip={self.trip} "
            f"station={self.station} value={value} limit={limit}: {self.message}"
        )


def validate_solution(instance: Instance, solution: Solution) -> list[Violation]:
    violations: list[Violation] = []
    orders = {order.id: order for order in instance.orders}
    couriers = {courier.id: courier for courier in instance.couriers}
    assigned: Counter[int] = Counter()
    locations: dict[int, list[tuple[int, int, int]]] = {}

    def add(
        equation: str,
        courier_id: int,
        trip: int,
        station: int,
        value: float | None,
        limit: float | None,
        message: str,
    ) -> None:
        violations.append(
            Violation(equation, courier_id, trip, station, value, limit, message)
        )

    for courier_id in sorted(solution.trips):
        courier = couriers.get(courier_id)
        if courier is None:
            add("13", courier_id, 0, 0, 1.0, 0.0, "unknown courier")
            continue
        distance = instance.distance_matrix(courier.vehicle_type)
        for trip_number, trip in enumerate(solution.trips[courier_id], start=1):
            if trip.start != DEPOT_ID:
                add(
                    "6", courier_id, trip_number, trip.start,
                    float(trip.start), float(DEPOT_ID), "trip must start at depot v0",
                )
            station_ids = trip.station_ids
            known_stations = all(0 <= station < len(instance.stations) for station in station_ids)
            if known_stations and 0 <= trip.start < len(instance.stations) and 0 <= trip.end < len(instance.stations):
                legs = [trip.start, *station_ids, trip.end]
                route_distance = sum(
                    distance[a][b] for a, b in zip(legs, legs[1:])
                )
                if route_distance > courier.range_km + _EPS:
                    add(
                        "8", courier_id, trip_number,
                        station_ids[-1] if station_ids else trip.end,
                        route_distance, courier.range_km,
                        "trip distance including depot return exceeds vehicle range",
                    )
            else:
                invalid = next(
                    (s for s in [trip.start, *station_ids, trip.end] if not 0 <= s < len(instance.stations)),
                    -1,
                )
                add("8", courier_id, trip_number, invalid, float(invalid), float(instance.K), "unknown station in trip")

            trip_orders = [orders.get(order_id) for order_id in trip.order_ids]
            known_trip_orders = [order for order in trip_orders if order is not None]
            initial_delivery = sum(order.delivery_weight_kg for order in known_trip_orders)
            if initial_delivery > courier.capacity_kg + _EPS:
                add(
                    "9", courier_id, trip_number, DEPOT_ID,
                    initial_delivery, courier.capacity_kg,
                    "delivery payload at depot departure exceeds capacity",
                )

            load = initial_delivery
            for stop in trip.stops:
                stop_orders = []
                for order_id in stop.order_ids:
                    assigned[order_id] += 1
                    locations.setdefault(order_id, []).append(
                        (courier_id, trip_number, stop.station)
                    )
                    order = orders.get(order_id)
                    if order is None:
                        add(
                            "13", courier_id, trip_number, stop.station,
                            1.0, 0.0, f"unknown order id {order_id}",
                        )
                        continue
                    stop_orders.append(order)
                    for package in order.packages:
                        if package.kind == KIND_DELIVERY and package.dest != stop.station:
                            add(
                                "11", courier_id, trip_number, stop.station,
                                float(stop.station), float(package.dest),
                                f"package {package.id} is not delivered to its destination",
                            )
                        if package.kind == KIND_PICKUP and package.source != stop.station:
                            add(
                                "12", courier_id, trip_number, stop.station,
                                float(stop.station), float(package.source),
                                f"package {package.id} is not picked up at its source",
                            )
                load -= sum(order.delivery_weight_kg for order in stop_orders)
                load += sum(order.pickup_weight_kg for order in stop_orders)
                if load > courier.capacity_kg + _EPS:
                    add(
                        "10", courier_id, trip_number, stop.station,
                        load, courier.capacity_kg,
                        "payload after unload-then-load exceeds capacity",
                    )
            if trip.end != DEPOT_ID and any(
                package.kind == KIND_PICKUP
                for order in known_trip_orders
                for package in order.packages
            ):
                add(
                    "12", courier_id, trip_number, trip.end,
                    float(trip.end), float(DEPOT_ID),
                    "pickup packages must return to depot v0",
                )

    for order_id in sorted(orders):
        count = assigned[order_id]
        if count != 1:
            courier_id, trip_number, station = locations.get(
                order_id, [(0, 0, 0)]
            )[-1]
            add(
                "13", courier_id, trip_number, station,
                float(count), 1.0,
                f"order {order_id} must be completed exactly once",
            )
    return violations


def assert_valid(instance: Instance, solution: Solution) -> None:
    violations = validate_solution(instance, solution)
    if violations:
        raise ValueError("\n".join(str(violation) for violation in violations))
