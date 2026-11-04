"""SPEC 表 3/4/5 的冻结常量与单位约定。

单位约定（全包统一）：
- 距离：km
- 重量：kg
- 速度：km/h（存储）；min 仅在显示时使用
- 服务时间 T：min/站，配置项，唯一来源 configs/base.yaml（不在本模块）
"""

from __future__ import annotations

from .model import VehicleType

# ---- SPEC 表 3：算例生成参数 ----
MAP_RADIUS_KM = 30.0            # 圆形服务区域半径，v0 在圆心
PACKAGE_WEIGHT_MIN_KG = 0.1
PACKAGE_WEIGHT_MAX_KG = 20.0
PACKAGES_PER_ORDER_MIN = 1
PACKAGES_PER_ORDER_MAX = 5
MIN_STATION_DISTANCE_KM = 0.5   # 任意两站（含 v0）最小间距 500 m（施加在欧氏距离 E 上）
ROAD_FACTOR_MIN = 1.2           # 道路系数 ρ(k,k′) ~ U[1.2, 1.4]，对称，每弧独立采样
ROAD_FACTOR_MAX = 1.4           # 地面距离 D = ρ·E；无人机距离 D′ = E（SPEC §2 修订）

# 注：站点服务时间 T 不在此处硬编码——它是配置项，唯一来源为 configs/base.yaml
# （SPEC §5 表 3 / §9-1；当前 5 min/站），由 qega.config.load_base_config() 读取。

# ---- SPEC 表 5：异构车队（三种车型） ----
DRONE = VehicleType(
    name="drone",
    capacity_kg=5.0,
    speed_kmh=60.0,
    range_km=20.0,
    aerial=True,
    proportion=0.20,
)
MINI_VAN = VehicleType(
    name="mini_van",
    capacity_kg=150.0,
    speed_kmh=35.0,
    range_km=300.0,
    aerial=False,
    proportion=0.30,
)
ELECTRIC_TRICYCLE = VehicleType(
    name="electric_tricycle",
    capacity_kg=50.0,
    speed_kmh=25.0,
    range_km=80.0,
    aerial=False,
    proportion=0.50,
)

# 车队配置的固定类型顺序（最大余数法平局时也按此顺序优先）
VEHICLE_TYPES = (DRONE, MINI_VAN, ELECTRIC_TRICYCLE)

# ---- SPEC 表 4：规模因子水平 ----
N_LEVELS = (200, 500, 1000, 2000, 5000)
K_LEVELS = (10, 15, 20, 25, 30)
M_LEVELS = (8, 18, 35, 65, 150)
SCHEME_LEVELS = ("R", "U")  # 三个订单参数各自独立取 R/U → 2×2×2

# 退化组合（SPEC §5 / 审稿意见 8）：一人一单负载均衡，生成时打标记
DEGENERATE_N = 200
DEGENERATE_M = 150

# ---- 订单采样分布参数（SPEC §5） ----
POISSON_LAMBDA = 1.4        # Type-R 每单包裹数：截断 Poisson on [1,5]
LOGNORMAL_MU = 0.5          # Type-R 包裹重量：截断对数正态
LOGNORMAL_SIGMA = 1.0
SPLIT_BINOMIAL_P = 0.5      # Type-R 配送/取件划分：B(n_i+m_i, 0.5)
