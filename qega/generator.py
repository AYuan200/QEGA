"""算例生成器（SPEC §5）。

- 服务区域：半径 30 km 圆形，v0 在圆心；任意两站（含 v0）欧氏间距 ≥ 500 m（拒绝采样）。
- 车队：三种车型按表 5 比例配置（最大余数法取整，确定性平局规则）。
- 订单：三个参数各自独立采用 Type-R / Type-U 采样方案（scheme_code 三位字母）。
- 可行性（SPEC §6-M1，扩展式 7）：每个订单必须至少存在一种车型
  在载重（max(配送总重, 取件总重) ≤ W）与航程（单独一趟 2·d(v0,s) ≤ r_j）上
  均可行，航程按算例实际距离矩阵 D/D′ 判定；不满足则整单重采样。
- 退化组合 N=200 且 M=150 打 degenerate 标记。

随机性：单个 numpy Generator 驱动，种子由调用方给定；同种子逐字节可复现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import constants as C
from .model import (
    DEPOT_ID,
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

# 三个采样方案因子在 scheme_code 中的位置
SCHEME_FIELDS = ("packages_per_order", "package_weight", "delivery_split")


@dataclass(frozen=True)
class GeneratorParams:
    """单个算例的生成参数（表 4 全因子的一个组合）。"""

    N: int
    K: int
    M: int
    scheme_packages: str = "R"  # 每单包裹数：R=截断Poisson(1.4) / U=离散均匀[1,5]
    scheme_weight: str = "R"    # 包裹重量：R=截断对数正态(0.5,1) / U=连续均匀[0.1,20]
    scheme_split: str = "R"     # 配送/取件划分：R=B(n,0.5) / U=离散均匀[0,n]
    drone_ratio: float | None = None  # None=表5默认20%；车队消融接口（SPEC M8）

    def __post_init__(self) -> None:
        for name, v in (
            ("scheme_packages", self.scheme_packages),
            ("scheme_weight", self.scheme_weight),
            ("scheme_split", self.scheme_split),
        ):
            if v not in C.SCHEME_LEVELS:
                raise ValueError(f"{name} must be one of {C.SCHEME_LEVELS}, got {v!r}")
        if self.drone_ratio is not None and not 0.0 <= self.drone_ratio <= 1.0:
            raise ValueError("drone_ratio must be in [0, 1]")

    @property
    def scheme_code(self) -> str:
        return self.scheme_packages + self.scheme_weight + self.scheme_split

    @property
    def degenerate(self) -> bool:
        return self.N == C.DEGENERATE_N and self.M == C.DEGENERATE_M

    @property
    def instance_id(self) -> str:
        base = f"N{self.N}_K{self.K}_M{self.M}_{self.scheme_code}"
        if self.drone_ratio is not None and self.drone_ratio != C.DRONE.proportion:
            base += f"_dr{self.drone_ratio:.2f}"
        return base


# ---------------------------------------------------------------- 采样原语

def _trunc_poisson(rng: np.random.Generator) -> int:
    """截断 Poisson(λ=1.4) on [1,5]（拒绝采样）。"""
    while True:
        x = int(rng.poisson(C.POISSON_LAMBDA))
        if C.PACKAGES_PER_ORDER_MIN <= x <= C.PACKAGES_PER_ORDER_MAX:
            return x


def _trunc_lognormal(rng: np.random.Generator) -> float:
    """对数正态(μ=0.5, σ=1) 截断至 [0.1, 20]（拒绝采样）。"""
    while True:
        x = float(rng.lognormal(C.LOGNORMAL_MU, C.LOGNORMAL_SIGMA))
        if C.PACKAGE_WEIGHT_MIN_KG <= x <= C.PACKAGE_WEIGHT_MAX_KG:
            return x


def _sample_num_packages(rng: np.random.Generator, scheme: str) -> int:
    if scheme == "R":
        return _trunc_poisson(rng)
    return int(rng.integers(C.PACKAGES_PER_ORDER_MIN, C.PACKAGES_PER_ORDER_MAX + 1))


def _sample_weight(rng: np.random.Generator, scheme: str) -> float:
    if scheme == "R":
        return _trunc_lognormal(rng)
    return float(
        rng.uniform(C.PACKAGE_WEIGHT_MIN_KG, C.PACKAGE_WEIGHT_MAX_KG)
    )


def _sample_num_delivery(rng: np.random.Generator, total: int, scheme: str) -> int:
    if scheme == "R":
        return int(rng.binomial(total, C.SPLIT_BINOMIAL_P))
    return int(rng.integers(0, total + 1))


# ---------------------------------------------------------------- 站点与车队

def _sample_stations(
    rng: np.random.Generator,
    K: int,
    radius_km: float = C.MAP_RADIUS_KM,
    min_dist_km: float = C.MIN_STATION_DISTANCE_KM,
    max_tries_per_station: int = 100_000,
) -> list[Station]:
    """v0 固定于圆心；v1..vK 圆盘均匀采样，两两（含 v0）欧氏间距 ≥ min_dist。"""
    stations = [Station(id=DEPOT_ID, x=0.0, y=0.0)]
    for sid in range(1, K + 1):
        for _ in range(max_tries_per_station):
            r = radius_km * math.sqrt(float(rng.random()))
            theta = 2.0 * math.pi * float(rng.random())
            x, y = r * math.cos(theta), r * math.sin(theta)
            if all(
                math.hypot(x - s.x, y - s.y) >= min_dist_km for s in stations
            ):
                stations.append(Station(id=sid, x=x, y=y))
                break
        else:
            raise RuntimeError(
                f"cannot place station {sid} with min distance {min_dist_km} km"
            )
    return stations


def fleet_type_counts(
    M: int,
    drone_ratio: float | None = None,
) -> tuple[tuple[VehicleType, int], ...]:
    """按占比把 M 名快递员分配到三种车型。

    最大余数法取整；小数部分并列时按 VEHICLE_TYPES 固定顺序优先（确定性）。
    drone_ratio=None 用表 5 默认 20%；给定值时小型货车:三轮车按 3:5 分剩余。
    """
    if drone_ratio is None:
        ratios = [t.proportion for t in C.VEHICLE_TYPES]
    else:
        rest = 1.0 - drone_ratio
        ratios = [drone_ratio, rest * 0.375, rest * 0.625]  # 0.3:0.5 = 3:5
    raw = [M * r for r in ratios]
    counts = [math.floor(x) for x in raw]
    remainder = M - sum(counts)
    fracs = [x - math.floor(x) for x in raw]
    # 小数部分降序；并列按类型索引升序（VEHICLE_TYPES 顺序优先）
    order = sorted(range(len(raw)), key=lambda i: (-fracs[i], i))
    for i in order[:remainder]:
        counts[i] += 1
    return tuple(zip(C.VEHICLE_TYPES, counts))


def _sample_road_factors(
    rng: np.random.Generator,
    n_stations: int,
    lo: float = C.ROAD_FACTOR_MIN,
    hi: float = C.ROAD_FACTOR_MAX,
) -> list[list[float]]:
    """道路系数矩阵 ρ：每对无序站点独立采样 U[lo,hi]，对称，对角线 1.0。"""
    rho = [[1.0] * n_stations for _ in range(n_stations)]
    for i in range(n_stations):
        for j in range(i + 1, n_stations):
            r = float(rng.uniform(lo, hi))
            rho[i][j] = rho[j][i] = r
    return rho


def build_fleet(M: int, drone_ratio: float | None = None) -> list[Courier]:
    """构造车队：按车型顺序连续编号 c1..cM。"""
    couriers: list[Courier] = []
    cid = 1
    for vtype, count in fleet_type_counts(M, drone_ratio):
        for _ in range(count):
            couriers.append(Courier(id=cid, vehicle_type=vtype))
            cid += 1
    return couriers


# ---------------------------------------------------------------- 订单与可行性

def _sample_order(
    rng: np.random.Generator, order_id: int, K: int, params: GeneratorParams
) -> Order:
    station = int(rng.integers(1, K + 1))
    total = _sample_num_packages(rng, params.scheme_packages)
    n_delivery = _sample_num_delivery(rng, total, params.scheme_split)
    packages = []
    for f in range(total):
        kind = KIND_DELIVERY if f < n_delivery else KIND_PICKUP
        packages.append(
            Package(
                id=f"p{order_id}_{f + 1}",
                kind=kind,
                weight_kg=_sample_weight(rng, params.scheme_weight),
                source=DEPOT_ID if kind == KIND_DELIVERY else station,
                dest=station if kind == KIND_DELIVERY else DEPOT_ID,
            )
        )
    return Order(id=order_id, station=station, packages=tuple(packages))


def order_feasible(
    order: Order,
    dist_ground: list[list[float]],
    dist_drone: list[list[float]],
    vehicle_types: list[VehicleType],
) -> bool:
    """SPEC §6-M1（扩展式 7）：存在一种车型载重与航程双重可行。

    载重：该订单单独一趟时，v0 装载的配送总重 ≤ W，且站内先卸后装后的
    取件总重 ≤ W（式 9/10 的最紧情形）→ max(配送总重, 取件总重) ≤ W。
    航程：单独一趟 v0→s→v0 的行驶距离 ≤ r_j（式 8）。
    距离一律读取算例落盘的实际矩阵 D/D′（无人机读 D′=E，地面车辆读 D=ρ·E），
    禁止用名义 1.3·E 近似。
    """
    required = max(order.delivery_weight_kg, order.pickup_weight_kg)
    for t in vehicle_types:
        if t.capacity_kg < required:
            continue
        d = dist_drone[DEPOT_ID][order.station] if t.aerial else dist_ground[DEPOT_ID][order.station]
        if 2.0 * d <= t.range_km:
            return True
    return False


def _sample_feasible_order(
    rng: np.random.Generator,
    order_id: int,
    dist_ground: list[list[float]],
    dist_drone: list[list[float]],
    vehicle_types: list[VehicleType],
    params: GeneratorParams,
    max_resamples: int = 10_000,
) -> Order:
    """采样订单并强制执行 M1 可行性假设；不满足则整单重采样。"""
    for _ in range(max_resamples):
        order = _sample_order(rng, order_id, params.K, params)
        if order_feasible(order, dist_ground, dist_drone, vehicle_types):
            return order
    raise RuntimeError(
        f"cannot sample a feasible order (id={order_id}) after {max_resamples} tries"
    )


# ---------------------------------------------------------------- 入口

def generate_instance(
    params: GeneratorParams, seed: int, instance_id: str | None = None
) -> Instance:
    """生成一个算例。同 (params, seed) 逐字节可复现。

    RNG 消耗顺序（固定）：站点坐标 → 道路系数 ρ → 订单（含可行性重采样）。
    """
    rng = np.random.default_rng(seed)
    stations = _sample_stations(rng, params.K)
    dist_drone = euclidean_matrix(stations)  # D′ = E
    rho = _sample_road_factors(rng, len(stations))
    dist_ground = [  # D = ρ·E
        [rho[i][j] * dist_drone[i][j] for j in range(len(stations))]
        for i in range(len(stations))
    ]
    couriers = build_fleet(params.M, params.drone_ratio)
    types = fleet_type_set(couriers)
    orders = [
        _sample_feasible_order(rng, oid, dist_ground, dist_drone, types, params)
        for oid in range(1, params.N + 1)
    ]
    return Instance(
        instance_id=instance_id or params.instance_id,
        stations=stations,
        orders=orders,
        couriers=couriers,
        dist_ground=dist_ground,
        dist_drone=dist_drone,
        scheme_code=params.scheme_code,
        seed=seed,
        degenerate=params.degenerate,
        meta={
            "schemes": dict(zip(SCHEME_FIELDS, params.scheme_code)),
            "map_radius_km": C.MAP_RADIUS_KM,
            "min_station_distance_km": C.MIN_STATION_DISTANCE_KM,
            "road_factor_range": [C.ROAD_FACTOR_MIN, C.ROAD_FACTOR_MAX],
        },
    )
