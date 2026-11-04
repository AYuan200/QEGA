"""Small-instance MILP lower bounds using SciPy/HiGHS.

The formulation follows the final manuscript: order assignment, depot-based
trips, subtour elimination, unload-before-load capacity, and range including
the return to the depot. It is intended only for N <= 50 clustering-loss
experiments, not for the large-scale benchmark.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from .model import DEPOT_ID, Instance


@dataclass(frozen=True)
class MILPResult:
    instance_id: str
    lower_bound_min: float
    incumbent_min: float | None
    solver_time_s: float
    status: int
    message: str
    mip_gap: float | None


def solve_milp(
    instance: Instance,
    t_station_min: float = 5.0,
    time_limit_s: float | None = None,
    mip_rel_gap: float = 0.0,
) -> MILPResult:
    started = time.perf_counter()
    N, K, M = instance.N, instance.K, instance.M
    orders = list(instance.orders)
    order_index = {order.id: i for i, order in enumerate(orders)}
    names: list[tuple] = []
    lower: list[float] = []
    upper: list[float] = []
    integrality: list[int] = []
    objective: list[float] = []

    def variable(name: tuple, lb: float = 0.0, ub: float = math.inf,
                 integer: bool = False, cost: float = 0.0) -> int:
        index = len(names)
        names.append(name)
        lower.append(lb)
        upper.append(ub)
        integrality.append(int(integer))
        objective.append(cost)
        return index

    cmax = variable(("cmax",), cost=1.0)
    x = {
        (order.id, courier.id, trip): variable(("x", order.id, courier.id, trip), 0, 1, True)
        for order in orders for courier in instance.couriers for trip in range(1, N + 1)
    }
    y = {}
    for courier in instance.couriers:
        for trip in range(1, N + 1):
            for a in range(K + 1):
                for b in range(K + 1):
                    if a != b:
                        y[courier.id, trip, a, b] = variable(
                            ("y", courier.id, trip, a, b), 0, 1, True)
    arrival = {}
    load = {}
    for courier in instance.couriers:
        for trip in range(1, N + 1):
            for station in range(K + 1):
                arrival[courier.id, trip, station] = variable(
                    ("arrival", courier.id, trip, station),
                    0.0, 0.0 if station == DEPOT_ID else math.inf)
                load[courier.id, trip, station] = variable(
                    ("load", courier.id, trip, station), 0.0, courier.capacity_kg)

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []

    def constraint(coefficients: dict[int, float], lb: float = -math.inf,
                   ub: float = math.inf) -> None:
        row = len(constraint_lower)
        for column, value in coefficients.items():
            if value:
                rows.append(row)
                cols.append(column)
                values.append(float(value))
        constraint_lower.append(lb)
        constraint_upper.append(ub)

    for order in orders:
        constraint({x[order.id, c.id, t]: 1.0
                    for c in instance.couriers for t in range(1, N + 1)}, 1.0, 1.0)

    total_weight = sum(order.total_weight_kg for order in orders)
    for courier in instance.couriers:
        vt = courier.vehicle_type
        distance = instance.distance_matrix(vt)
        speed = vt.speed_km_per_min
        max_leg_time = max(distance[a][b] for a in range(K + 1) for b in range(K + 1)) / speed
        time_big = 2.0 * (vt.range_km / speed + (K + 1) * t_station_min + max_leg_time)
        load_big = courier.capacity_kg + total_weight
        for trip in range(1, N + 1):
            for station in range(K + 1):
                flow = {}
                for other in range(K + 1):
                    if other == station:
                        continue
                    flow[y[courier.id, trip, other, station]] = 1.0
                    flow[y[courier.id, trip, station, other]] = -1.0
                constraint(flow, 0.0, 0.0)
            constraint({y[courier.id, trip, DEPOT_ID, b]: 1.0
                        for b in range(1, K + 1)}, ub=1.0)
            for order in orders:
                station = order.station
                coefficients = {x[order.id, courier.id, trip]: 1.0}
                coefficients.update({y[courier.id, trip, station, b]: -1.0
                                     for b in range(K + 1) if b != station})
                constraint(coefficients, ub=0.0)

            for a in range(1, K + 1):
                for b in range(1, K + 1):
                    if a == b:
                        continue
                    arc = y[courier.id, trip, a, b]
                    travel = distance[a][b] / speed
                    constraint({
                        arrival[courier.id, trip, b]: 1.0,
                        arrival[courier.id, trip, a]: -1.0,
                        arc: -time_big,
                    }, lb=travel + t_station_min - time_big)
                    delta = {
                        x[order.id, courier.id, trip]: order.pickup_weight_kg - order.delivery_weight_kg
                        for order in orders if order.station == a
                    }
                    coefficients = {
                        load[courier.id, trip, b]: 1.0,
                        load[courier.id, trip, a]: -1.0,
                        arc: -load_big,
                    }
                    coefficients.update({index: -value for index, value in delta.items()})
                    constraint(coefficients, lb=-load_big)
            for station in range(1, K + 1):
                arc = y[courier.id, trip, DEPOT_ID, station]
                travel = distance[DEPOT_ID][station] / speed
                constraint({arrival[courier.id, trip, station]: 1.0, arc: -time_big},
                           lb=travel - time_big)
                delta = {
                    x[order.id, courier.id, trip]: order.pickup_weight_kg - order.delivery_weight_kg
                    for order in orders if order.station == station
                }
                coefficients = {load[courier.id, trip, station]: 1.0}
                coefficients.update(delta)
                constraint(coefficients, ub=courier.capacity_kg)
            constraint({
                load[courier.id, trip, DEPOT_ID]: 1.0,
                **{x[order.id, courier.id, trip]: -order.delivery_weight_kg
                   for order in orders},
            }, 0.0, 0.0)
            constraint({
                y[courier.id, trip, a, b]: distance[a][b]
                for a in range(K + 1) for b in range(K + 1) if a != b
            }, ub=vt.range_km)

    for courier in instance.couriers:
        vt = courier.vehicle_type
        distance = instance.distance_matrix(vt)
        speed = vt.speed_km_per_min
        coefficients = {cmax: 1.0}
        for trip in range(1, N + 1):
            for a in range(K + 1):
                for b in range(K + 1):
                    if a == b:
                        continue
                    stop = t_station_min if b >= 1 else 0.0
                    coefficients[y[courier.id, trip, a, b]] = -(distance[a][b] / speed + stop)
        constraint(coefficients, lb=0.0)

    matrix = coo_array(
        (np.asarray(values), (np.asarray(rows), np.asarray(cols))),
        shape=(len(constraint_lower), len(names)),
    ).tocsc()
    options = {"mip_rel_gap": float(mip_rel_gap)}
    if time_limit_s is not None:
        options["time_limit"] = float(time_limit_s)
    result = milp(
        c=np.asarray(objective), integrality=np.asarray(integrality),
        bounds=Bounds(np.asarray(lower), np.asarray(upper)),
        constraints=LinearConstraint(
            matrix, np.asarray(constraint_lower), np.asarray(constraint_upper)),
        options=options,
    )
    bound = float(getattr(result, "mip_dual_bound", math.nan))
    incumbent = float(result.fun) if result.fun is not None else None
    gap = getattr(result, "mip_gap", None)
    return MILPResult(
        instance_id=instance.instance_id,
        lower_bound_min=bound,
        incumbent_min=incumbent,
        solver_time_s=time.perf_counter() - started,
        status=int(result.status),
        message=str(result.message),
        mip_gap=None if gap is None else float(gap),
    )
