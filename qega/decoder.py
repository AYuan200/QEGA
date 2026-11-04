"""队列染色体的 DM/AM 解码器（SPEC §3.2–3.4 修订版）。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Mapping, Sequence

from .evaluator import evaluate_skeleton
from .model import DEPOT_ID, Courier, Instance, Order
from .sorters import mks_sort

_EPS = 1e-9


@dataclass(frozen=True)
class Stop:
    station: int
    order_ids: tuple[int, ...]

    def to_dict(self) -> dict:
        return {"station": self.station, "order_ids": list(self.order_ids)}

    @staticmethod
    def from_dict(data: dict) -> "Stop":
        return Stop(int(data["station"]), tuple(int(x) for x in data["order_ids"]))


@dataclass(frozen=True)
class Trip:
    stops: tuple[Stop, ...]
    start: int = DEPOT_ID
    end: int = DEPOT_ID

    @property
    def station_ids(self) -> tuple[int, ...]:
        return tuple(stop.station for stop in self.stops)

    @property
    def order_ids(self) -> tuple[int, ...]:
        return tuple(order_id for stop in self.stops for order_id in stop.order_ids)

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "stops": [stop.to_dict() for stop in self.stops],
        }

    @staticmethod
    def from_dict(data: dict) -> "Trip":
        return Trip(
            stops=tuple(Stop.from_dict(stop) for stop in data["stops"]),
            start=int(data.get("start", DEPOT_ID)),
            end=int(data.get("end", DEPOT_ID)),
        )


@dataclass
class Solution:
    trips: dict[int, list[Trip]]

    def skeleton(self) -> dict[int, list[tuple[int, ...]]]:
        return {
            courier_id: [trip.station_ids for trip in trips]
            for courier_id, trips in self.trips.items()
        }

    def to_dict(self) -> dict:
        return {
            str(courier_id): [trip.to_dict() for trip in self.trips[courier_id]]
            for courier_id in sorted(self.trips)
        }

    @staticmethod
    def from_dict(data: dict) -> "Solution":
        return Solution(
            {
                int(courier_id): [Trip.from_dict(trip) for trip in trips]
                for courier_id, trips in data.items()
            }
        )


@dataclass(frozen=True)
class DecodeResult:
    solution: Solution
    times_min: dict[int, float]
    makespan_min: float

    @property
    def fitness(self) -> float:
        return self.makespan_min


class EvaluationBudgetExhausted(RuntimeError):
    pass


def am(
    instance: Instance,
    courier: Courier,
    queues: Mapping[int, Deque[Order]],
    start_queue: int,
    queue_order: Sequence[int] | None = None,
) -> Trip:
    """Scan every queue once, skipping orders that violate load or range."""
    if start_queue not in queues:
        raise ValueError(f"unknown queue: {start_queue}")
    order_of_queues = list(queue_order or sorted(queues))
    if sorted(order_of_queues) != sorted(queues):
        raise ValueError("queue_order must contain every queue exactly once")
    start = order_of_queues.index(start_queue)
    scan_order = order_of_queues[start:] + order_of_queues[:start]
    distance = instance.distance_matrix(courier.vehicle_type)
    capacity = courier.capacity_kg
    max_range = courier.range_km

    accepted: list[Order] = []
    post_loads: list[float] = []
    initial_delivery = 0.0
    route_distance = 0.0
    last_station: int | None = None

    for queue_id in scan_order:
        queue = queues[queue_id]
        for _ in range(len(queue)):
            order = queue.popleft()
            station = order.station
            delivery = order.delivery_weight_kg
            pickup = order.pickup_weight_kg
            new_initial = initial_delivery + delivery

            if last_station == station:
                earlier_load = max(post_loads[:-1], default=0.0)
                new_last_load = post_loads[-1] + pickup
                new_distance = route_distance
                capacity_ok = (
                    new_initial <= capacity + _EPS
                    and earlier_load + delivery <= capacity + _EPS
                    and new_last_load <= capacity + _EPS
                )
                new_post_load = new_last_load
            else:
                previous_max = max(post_loads, default=0.0)
                base_load = post_loads[-1] if post_loads else initial_delivery
                new_post_load = base_load + pickup
                if last_station is None:
                    new_distance = 2.0 * distance[DEPOT_ID][station]
                else:
                    new_distance = (
                        route_distance
                        - distance[last_station][DEPOT_ID]
                        + distance[last_station][station]
                        + distance[station][DEPOT_ID]
                    )
                capacity_ok = (
                    new_initial <= capacity + _EPS
                    and previous_max + delivery <= capacity + _EPS
                    and new_post_load <= capacity + _EPS
                )

            if capacity_ok and new_distance <= max_range + _EPS:
                accepted.append(order)
                initial_delivery = new_initial
                route_distance = new_distance
                if last_station == station:
                    for i in range(len(post_loads) - 1):
                        post_loads[i] += delivery
                    post_loads[-1] = new_post_load
                else:
                    for i in range(len(post_loads)):
                        post_loads[i] += delivery
                    post_loads.append(new_post_load)
                last_station = station
            else:
                queue.append(order)

    mutable_stops: list[list] = []
    for order in accepted:
        if mutable_stops and mutable_stops[-1][0] == order.station:
            mutable_stops[-1][1].append(order.id)
        else:
            mutable_stops.append([order.station, [order.id]])
    return Trip(
        tuple(Stop(station, tuple(order_ids)) for station, order_ids in mutable_stops)
    )


class Decoder:
    """DM 入口；每次实际解码恰好计一次 FE。"""

    def __init__(
        self,
        instance: Instance,
        queues: Mapping[int, Sequence[Order]],
        t_station_min: float,
        fe_max: int | None = None,
        sorter: Callable[[Sequence[Order], Instance], list[Order]] = mks_sort,
    ) -> None:
        if fe_max is not None and fe_max <= 0:
            raise ValueError("fe_max must be positive")
        if not instance.couriers:
            raise ValueError("instance must contain at least one courier")
        if not queues:
            raise ValueError("at least one queue is required")
        self.instance = instance
        self.t_station_min = float(t_station_min)
        self.fe_max = fe_max
        self._queue_ids = tuple(sorted(queues))
        self._sorted_queues = {
            queue_id: tuple(sorter(tuple(orders), instance))
            for queue_id, orders in queues.items()
        }
        self.fe_count = 0

    @property
    def remaining_budget(self) -> int | None:
        if self.fe_max is None:
            return None
        return self.fe_max - self.fe_count

    def decode(self, chromosome: Sequence[int]) -> DecodeResult:
        if self.fe_max is not None and self.fe_count >= self.fe_max:
            raise EvaluationBudgetExhausted("fitness evaluation budget exhausted")
        self.fe_count += 1
        if sorted(chromosome) != list(self._queue_ids):
            raise ValueError("chromosome must be a permutation of the queue ids")

        queues = {
            queue_id: deque(orders)
            for queue_id, orders in self._sorted_queues.items()
        }
        remaining = sum(len(queue) for queue in queues.values())
        trips = {courier.id: [] for courier in self.instance.couriers}
        courier_index = 0
        queue_index = 0
        unsuccessful = 0

        while remaining:
            courier = self.instance.couriers[courier_index]
            queue_id = chromosome[queue_index]
            trip = am(
                self.instance, courier, queues, queue_id, queue_order=chromosome
            )
            assigned = len(trip.order_ids)
            if assigned:
                trips[courier.id].append(trip)
                remaining -= assigned
                unsuccessful = 0
            else:
                unsuccessful += 1
                if unsuccessful >= self.instance.M:
                    raise RuntimeError(
                        "the remaining orders cannot be feasibly assigned to any courier"
                    )
            courier_index = (courier_index + 1) % self.instance.M
            queue_index = (queue_index + 1) % len(self._queue_ids)

        solution = Solution(trips)
        evaluation = evaluate_skeleton(
            self.instance, solution.skeleton(), self.t_station_min
        )
        return DecodeResult(
            solution=solution,
            times_min=evaluation.times_min,
            makespan_min=evaluation.makespan_min,
        )
