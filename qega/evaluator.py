"""行驶骨架时间评估器（SPEC §2 式 3–5 的纯时间计算，不含任何调度/分配逻辑）。

输入：算例 + 显式方案（每个快递员若干趟，每趟为站点访问序列）+ 服务时间 T。
输出：每个快递员的完工时间（min）与 makespan（min）。

时间模型：每趟从 v0 出发，按序访问站点后返回 v0；站间行驶时间 =
d(l_{τ-1}, l_τ)/V_j（距离按车型读算例落盘的 D/D′ 矩阵）；每停靠一站加 T；
同一快递员的各趟顺序执行、时间累加。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .model import DEPOT_ID, Instance

# 方案的数据形态：{courier_id: [trip, ...]}，trip = [station_id, ...]（不含 v0，
# 起止 v0 由评估器隐含补全）
SkeletonPlan = Mapping[int, Sequence[Sequence[int]]]


@dataclass(frozen=True)
class SkeletonEvaluation:
    times_min: dict[int, float]  # courier_id -> 完工时间（min）
    makespan_min: float


def evaluate_skeleton(
    instance: Instance, plan: SkeletonPlan, t_station_min: float
) -> SkeletonEvaluation:
    """计算骨架方案的各快递员完工时间与 makespan（单位 min）。"""
    courier_by_id = {c.id: c for c in instance.couriers}
    times: dict[int, float] = {}
    for cid, courier in courier_by_id.items():
        vt = courier.vehicle_type
        dist = instance.distance_matrix(vt)
        speed_km_min = vt.speed_km_per_min
        total = 0.0
        for trip in plan.get(cid, []):
            if not trip:
                raise ValueError(f"courier {cid}: empty trip")
            for s in trip:
                if not 0 <= s < len(instance.stations):
                    raise ValueError(f"courier {cid}: unknown station {s}")
            legs = [DEPOT_ID, *trip, DEPOT_ID]
            travel_km = sum(
                dist[a][b] for a, b in zip(legs, legs[1:])
            )
            total += travel_km / speed_km_min + len(trip) * t_station_min
        times[cid] = total
    unknown = set(plan) - set(courier_by_id)
    if unknown:
        raise ValueError(f"plan references unknown couriers: {sorted(unknown)}")
    return SkeletonEvaluation(
        times_min=times, makespan_min=max(times.values(), default=0.0)
    )
