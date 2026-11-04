"""SPEC §3.3 的队内订单排序接口。"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from .model import DEPOT_ID, Instance, Order

Sorter = Callable[[Sequence[Order], Instance], list[Order]]
_WEIGHT_SCALE = 10


def mks_sort(orders: Sequence[Order], instance: Instance) -> list[Order]:
    """按 depot 欧氏距离升序、订单总重降序排列。"""
    distance = instance.dist_drone
    return sorted(
        orders,
        key=lambda order: (
            distance[DEPOT_ID][order.station],
            -order.total_weight_kg,
            order.id,
        ),
    )


def rds_sort(
    orders: Sequence[Order], instance: Instance, rng: np.random.Generator
) -> list[Order]:
    """用给定 RNG 对队内订单作随机排列。"""
    indices = rng.permutation(len(orders))
    return [orders[int(index)] for index in indices]


def _loaded_batch(weights: Sequence[int], capacity: int) -> list[int]:
    """返回一个最大整数装载批次的订单下标；位图同时承担 DP 回溯。"""
    states = [-1] * (capacity + 1)
    states[0] = 0
    for index, weight in enumerate(weights):
        if weight > capacity:
            continue
        for load in range(capacity, weight - 1, -1):
            previous = load - weight
            if states[previous] >= 0 and states[load] < 0:
                states[load] = states[previous] | (1 << index)
    best_load = max(load for load, mask in enumerate(states) if mask >= 0)
    if best_load == 0:
        return []
    mask = states[best_load]
    return [index for index in range(len(weights)) if mask & (1 << index)]


def dps_sort(orders: Sequence[Order], instance: Instance) -> list[Order]:
    """按重复 0/1 背包得到的最大装载批次排列订单。

    排序阶段没有特定快递员上下文，因此容量取车队最大载重。重量向上、
    容量向下离散到 0.1 kg；最终可行性仍由 AM 按实际车型和连续重量判定。
    """
    if not orders:
        return []
    if not instance.couriers:
        raise ValueError("DPS requires at least one courier")
    capacity = int(
        math.floor(
            max(courier.capacity_kg for courier in instance.couriers)
            * _WEIGHT_SCALE
            + 1e-9
        )
    )
    remaining = list(orders)
    ordered: list[Order] = []
    while remaining:
        weights = [
            max(1, int(math.ceil(order.total_weight_kg * _WEIGHT_SCALE - 1e-9)))
            for order in remaining
        ]
        batch = _loaded_batch(weights, capacity)
        if not batch:
            ordered.extend(remaining)
            break
        selected = set(batch)
        ordered.extend(remaining[index] for index in batch)
        remaining = [
            order for index, order in enumerate(remaining) if index not in selected
        ]
    return ordered


def resolve_sorter(name: str, rng: np.random.Generator | None = None) -> Sorter:
    """把配置中的排序器名称转换为 Decoder 的二参数接口。"""
    key = name.lower()
    if key == "mks":
        return mks_sort
    if key == "dps":
        return dps_sort
    if key == "rds":
        if rng is None:
            raise ValueError("RDS requires a seeded random generator")
        return lambda orders, instance: rds_sort(orders, instance, rng)
    raise ValueError(f"unknown sorter: {name!r}; expected mks, rds, or dps")
