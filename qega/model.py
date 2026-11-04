"""核心数据结构（SPEC §2）：Station、Package、Order、Courier/VehicleType、Instance。

距离规则（SPEC §2 修订版）：
- 对每对无序站点 (k,k′) 采样道路系数 ρ(k,k′) ~ U[1.2,1.4]（对称，算例种子驱动）；
- 地面车辆距离 D = ρ·E（E 为欧氏距离）；无人机距离 D′ = E；
- D 与 D′ 随算例落盘固化于 Instance，所有模块读取同一来源；
- 任意两站间距离 ≥ 500 m 的约束施加在欧氏距离 E 上（由生成器保证）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

DEPOT_ID = 0  # v0 转运中心

KIND_DELIVERY = "delivery"  # 配送：s=v0, d∈{v1..vK}
KIND_PICKUP = "pickup"      # 取件：s∈{v1..vK}, d=v0


@dataclass(frozen=True)
class Station:
    """站点。id=0 为转运中心 v0，1..K 为配送柜。"""

    id: int
    x: float
    y: float

    @property
    def is_depot(self) -> bool:
        return self.id == DEPOT_ID

    def to_dict(self) -> dict:
        return {"id": self.id, "x": self.x, "y": self.y}

    @staticmethod
    def from_dict(d: dict) -> "Station":
        return Station(id=int(d["id"]), x=float(d["x"]), y=float(d["y"]))


@dataclass(frozen=True)
class Package:
    """包裹 p_{i,f}：重量 w、源站 s、目的站 d。"""

    id: str
    kind: str  # KIND_DELIVERY / KIND_PICKUP
    weight_kg: float
    source: int
    dest: int

    def __post_init__(self) -> None:
        if self.kind not in (KIND_DELIVERY, KIND_PICKUP):
            raise ValueError(f"unknown package kind: {self.kind!r}")
        if self.kind == KIND_DELIVERY and self.source != DEPOT_ID:
            raise ValueError("delivery package must originate at depot v0")
        if self.kind == KIND_PICKUP and self.dest != DEPOT_ID:
            raise ValueError("pickup package must be destined to depot v0")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "weight_kg": self.weight_kg,
            "source": self.source,
            "dest": self.dest,
        }

    @staticmethod
    def from_dict(d: dict) -> "Package":
        return Package(
            id=str(d["id"]),
            kind=str(d["kind"]),
            weight_kg=float(d["weight_kg"]),
            source=int(d["source"]),
            dest=int(d["dest"]),
        )


@dataclass(frozen=True)
class Order:
    """订单 E_i：不可拆分；对应唯一客户站点。

    同一订单内配送包裹的目的站 = 取件包裹的源站 = station（SPEC §2）。
    """

    id: int
    station: int  # 客户站点 v_k (k >= 1)
    packages: tuple[Package, ...]

    def __post_init__(self) -> None:
        if self.station == DEPOT_ID:
            raise ValueError("order station must be a customer station (not v0)")
        if not self.packages:
            raise ValueError("order must contain at least one package")
        for p in self.packages:
            if p.kind == KIND_DELIVERY and p.dest != self.station:
                raise ValueError("delivery package dest must equal order station")
            if p.kind == KIND_PICKUP and p.source != self.station:
                raise ValueError("pickup package source must equal order station")

    @property
    def n_delivery(self) -> int:
        return sum(1 for p in self.packages if p.kind == KIND_DELIVERY)

    @property
    def n_pickup(self) -> int:
        return sum(1 for p in self.packages if p.kind == KIND_PICKUP)

    @property
    def num_packages(self) -> int:
        return len(self.packages)

    @property
    def delivery_weight_kg(self) -> float:
        return sum(p.weight_kg for p in self.packages if p.kind == KIND_DELIVERY)

    @property
    def pickup_weight_kg(self) -> float:
        return sum(p.weight_kg for p in self.packages if p.kind == KIND_PICKUP)

    @property
    def total_weight_kg(self) -> float:
        return sum(p.weight_kg for p in self.packages)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "station": self.station,
            "packages": [p.to_dict() for p in self.packages],
        }

    @staticmethod
    def from_dict(d: dict) -> "Order":
        return Order(
            id=int(d["id"]),
            station=int(d["station"]),
            packages=tuple(Package.from_dict(p) for p in d["packages"]),
        )


@dataclass(frozen=True)
class VehicleType:
    """车型（SPEC 表 5）：载重 W、速度 V、航程 r、是否空中（无人机）、车队占比。

    aerial=True（无人机）使用直线距离矩阵 D′=E；aerial=False（地面车辆）
    使用道路距离矩阵 D=ρ·E（ρ 为算例级每弧采样系数）。
    """

    name: str
    capacity_kg: float
    speed_kmh: float
    range_km: float
    aerial: bool
    proportion: float = 0.0

    @property
    def is_drone(self) -> bool:
        return self.aerial

    @property
    def speed_km_per_min(self) -> float:
        return self.speed_kmh / 60.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "capacity_kg": self.capacity_kg,
            "speed_kmh": self.speed_kmh,
            "range_km": self.range_km,
            "aerial": self.aerial,
            "proportion": self.proportion,
        }

    @staticmethod
    def from_dict(d: dict) -> "VehicleType":
        return VehicleType(
            name=str(d["name"]),
            capacity_kg=float(d["capacity_kg"]),
            speed_kmh=float(d["speed_kmh"]),
            range_km=float(d["range_km"]),
            aerial=bool(d["aerial"]),
            proportion=float(d.get("proportion", 0.0)),
        )


@dataclass(frozen=True)
class Courier:
    """快递员 c_j，配备某一车型。"""

    id: int
    vehicle_type: VehicleType

    @property
    def capacity_kg(self) -> float:
        return self.vehicle_type.capacity_kg

    @property
    def speed_kmh(self) -> float:
        return self.vehicle_type.speed_kmh

    @property
    def range_km(self) -> float:
        return self.vehicle_type.range_km

    def to_dict(self) -> dict:
        return {"id": self.id, "vehicle_type": self.vehicle_type.to_dict()}

    @staticmethod
    def from_dict(d: dict) -> "Courier":
        return Courier(id=int(d["id"]), vehicle_type=VehicleType.from_dict(d["vehicle_type"]))


@dataclass
class Instance:
    """一个算例：站点、订单、车队、距离矩阵、规模因子与采样方案元数据。

    距离矩阵随算例落盘固化（SPEC §2），所有模块读取同一来源：
    - dist_ground（D）：地面车辆距离，D(k,k′)=ρ(k,k′)·E(k,k′)，ρ~U[1.2,1.4] 对称；
    - dist_drone（D′）：无人机距离 = 欧氏距离 E。
    """

    instance_id: str
    stations: list[Station]          # stations[0] 必须是 v0
    orders: list[Order]
    couriers: list[Courier]
    dist_ground: list[list[float]]   # D = ρ·E
    dist_drone: list[list[float]]    # D′ = E
    scheme_code: str = ""            # 三位 R/U 代码（包裹数/重量/划分），如 "RUU"
    seed: Optional[int] = None
    degenerate: bool = False         # N=200 且 M=150 的退化组合标记（SPEC §5）
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stations or self.stations[0].id != DEPOT_ID:
            raise ValueError("stations[0] must be the depot v0 (id=0)")
        ids = [s.id for s in self.stations]
        if ids != list(range(len(ids))):
            raise ValueError("station ids must be consecutive 0..K")
        n = len(self.stations)
        for name, m in (("dist_ground", self.dist_ground), ("dist_drone", self.dist_drone)):
            if len(m) != n or any(len(row) != n for row in m):
                raise ValueError(f"{name} must be a {n}x{n} matrix")
            for i in range(n):
                if m[i][i] != 0.0:
                    raise ValueError(f"{name} diagonal must be zero")
                for j in range(i + 1, n):
                    if m[i][j] != m[j][i]:
                        raise ValueError(f"{name} must be symmetric")

    # ---- 规模 ----
    @property
    def N(self) -> int:
        return len(self.orders)

    @property
    def K(self) -> int:
        return len(self.stations) - 1

    @property
    def M(self) -> int:
        return len(self.couriers)

    @property
    def type_label(self) -> str:
        if self.scheme_code == "RRR":
            return "Type-R"
        if self.scheme_code == "UUU":
            return "Type-U"
        return f"Mixed-{self.scheme_code}" if self.scheme_code else "Fixed"

    # ---- 距离矩阵 ----
    @property
    def dist_euclidean(self) -> list[list[float]]:
        """欧氏距离矩阵 E（即无人机矩阵 D′）。"""
        return self.dist_drone

    def distance_matrix(self, vehicle_type: VehicleType) -> list[list[float]]:
        """该车型的行驶距离矩阵：无人机读 D′，地面车辆读 D。"""
        return self.dist_drone if vehicle_type.aerial else self.dist_ground

    def travel_distance(self, from_id: int, to_id: int, vehicle_type: VehicleType) -> float:
        return self.distance_matrix(vehicle_type)[from_id][to_id]

    def road_factor_matrix(self) -> list[list[float]]:
        """道路系数矩阵 ρ = D/E（对角线约定为 1.0）。"""
        n = len(self.stations)
        rho = [[1.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                r = self.dist_ground[i][j] / self.dist_drone[i][j]
                rho[i][j] = rho[j][i] = r
        return rho

    def min_interstation_distance(self) -> float:
        """任意两站（含 v0）最小间距（欧氏 E 口径，SPEC §2）。"""
        d = self.dist_drone
        n = len(d)
        return min(d[i][j] for i in range(n) for j in range(i + 1, n))

    # ---- 序列化（canonical JSON：同种子两次生成逐字节一致） ----
    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "seed": self.seed,
            "scheme_code": self.scheme_code,
            "type_label": self.type_label,
            "degenerate": self.degenerate,
            "N": self.N,
            "K": self.K,
            "M": self.M,
            "meta": self.meta,
            "stations": [s.to_dict() for s in self.stations],
            "couriers": [c.to_dict() for c in self.couriers],
            "orders": [o.to_dict() for o in self.orders],
            "dist_ground": self.dist_ground,
            "dist_drone": self.dist_drone,
        }

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        return Instance(
            instance_id=str(d["instance_id"]),
            seed=d["seed"],
            scheme_code=str(d["scheme_code"]),
            degenerate=bool(d["degenerate"]),
            meta=dict(d.get("meta", {})),
            stations=[Station.from_dict(s) for s in d["stations"]],
            couriers=[Courier.from_dict(c) for c in d["couriers"]],
            orders=[Order.from_dict(o) for o in d["orders"]],
            dist_ground=[[float(x) for x in row] for row in d["dist_ground"]],
            dist_drone=[[float(x) for x in row] for row in d["dist_drone"]],
        )

    def to_canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    @staticmethod
    def from_canonical_json(text: str) -> "Instance":
        return Instance.from_dict(json.loads(text))


def euclidean_matrix(stations: Sequence[Station]) -> list[list[float]]:
    """由站点坐标计算欧氏距离矩阵 E（对称、对角线 0）。"""
    n = len(stations)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dij = math.hypot(
                stations[i].x - stations[j].x, stations[i].y - stations[j].y
            )
            d[i][j] = d[j][i] = dij
    return d


def fleet_type_set(couriers: Sequence[Courier]) -> list[VehicleType]:
    """车队中出现的车型集合（按首次出现顺序，确定性）。"""
    seen: dict[str, VehicleType] = {}
    for c in couriers:
        seen.setdefault(c.vehicle_type.name, c.vehicle_type)
    return list(seen.values())
