"""手工算例 fixture（SPEC §9-1）：论文图 2/3 use case 的纯行驶骨架测试。

口径（最终决策）：
- 论文 use case 按 T=0 绘制，仅作定性示例，不承载 T 标定功能；
- 距离直接用论文给定值（D01=1.5, D02=2, D03=1, D12=3 km，对称补全 D13/D23），
  不套任何道路系数（dist_ground 与 dist_drone 均为论文矩阵）；
- 速度为论文原值：c1 V=1/2 km/min（=30 km/h），c2 V=2/3 km/min（=40 km/h）；
- S1/S2 为显式编码的方案数据，仅用于验证评估器的时间计算，不用于解码器回归。

站点坐标嵌入（与论文距离严格一致）：v0=(0,0), v1=(1.5,0),
v2=(-11/12, √455/12)（D02=2、D12=3 精确成立）, v3=(1,0)（D13=0.5、D23=√(41/6)，
满足对称补全且 ≥500 m）；D13/D23 不出现于 S1/S2 的任何路径。
"""

import math

import pytest

from qega.config import load_base_config
from qega.evaluator import evaluate_skeleton
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

# 论文给定的距离值（km）
D01, D02, D03, D12 = 1.5, 2.0, 1.0, 3.0
D13 = 0.5                      # 对称补全（≥500 m 下限）
D23 = math.sqrt(41.0 / 6.0)    # 对称补全 ≈ 2.6141 km

STATIONS = [
    Station(0, 0.0, 0.0),
    Station(1, 1.5, 0.0),
    Station(2, -11.0 / 12.0, math.sqrt(455.0) / 12.0),
    Station(3, 1.0, 0.0),
]

VT_C1 = VehicleType("manual_c1", capacity_kg=20.0, speed_kmh=30.0, range_km=math.inf, aerial=False)
VT_C2 = VehicleType("manual_c2", capacity_kg=15.0, speed_kmh=40.0, range_km=math.inf, aerial=False)

# 显式方案数据（论文图 3）：courier_id -> [trip, ...]，trip = 站点访问序列
S1 = {1: [[1, 2]], 2: [[2], [3]]}
S2 = {1: [[2], [3]], 2: [[1], [2]]}

# T=0 骨架完工时间（min），即图 3 甘特图的时间轴读数
S1_SKELETON = {1: 13.0, 2: 9.0}    # makespan 13
S2_SKELETON = {1: 12.0, 2: 10.5}   # makespan 12


@pytest.fixture(scope="module")
def manual_instance():
    p1 = Package("p1", KIND_DELIVERY, 8.0, 0, 1)
    p2 = Package("p2", KIND_DELIVERY, 10.0, 0, 2)
    p3 = Package("p3", KIND_DELIVERY, 13.0, 0, 2)
    p4 = Package("p4", KIND_DELIVERY, 12.0, 0, 3)
    p5 = Package("p5", KIND_PICKUP, 10.0, 1, 0)
    p6 = Package("p6", KIND_PICKUP, 14.0, 3, 0)
    orders = [
        Order(1, 1, (p1, p5)),
        Order(2, 2, (p2,)),
        Order(3, 2, (p3,)),
        Order(4, 3, (p4, p6)),
    ]
    couriers = [Courier(1, VT_C1), Courier(2, VT_C2)]
    dist = euclidean_matrix(STATIONS)  # 与论文 D 严格一致，不套道路系数
    return Instance(
        instance_id="manual_fig2",
        stations=list(STATIONS),
        orders=orders,
        couriers=couriers,
        dist_ground=[row[:] for row in dist],
        dist_drone=[row[:] for row in dist],
        scheme_code="",
        seed=None,
        degenerate=False,
        meta={"source": "paper Fig.2/Fig.3 use case; travel-skeleton test only"},
    )


class TestManualInstanceData:
    def test_distances_are_paper_values_no_road_factor(self, manual_instance):
        d = manual_instance.dist_drone
        assert d[0][1] == pytest.approx(D01, abs=1e-12)
        assert d[0][2] == pytest.approx(D02, abs=1e-12)
        assert d[0][3] == pytest.approx(D03, abs=1e-12)
        assert d[1][2] == pytest.approx(D12, abs=1e-12)
        assert d[1][3] == pytest.approx(D13, abs=1e-12)
        assert d[2][3] == pytest.approx(D23, abs=1e-12)
        # 不套道路系数：地面矩阵与无人机矩阵均为论文值
        assert manual_instance.dist_ground == manual_instance.dist_drone

    def test_courier_speeds_are_paper_fractions(self, manual_instance):
        c1, c2 = manual_instance.couriers
        assert c1.capacity_kg == 20.0
        assert c1.vehicle_type.speed_km_per_min == pytest.approx(0.5)
        assert c2.capacity_kg == 15.0
        assert c2.vehicle_type.speed_km_per_min == pytest.approx(2.0 / 3.0)


class TestSkeletonAtT0:
    def test_s1(self, manual_instance):
        ev = evaluate_skeleton(manual_instance, S1, t_station_min=0.0)
        assert ev.times_min[1] == pytest.approx(S1_SKELETON[1], abs=1e-9)
        assert ev.times_min[2] == pytest.approx(S1_SKELETON[2], abs=1e-9)
        assert ev.makespan_min == pytest.approx(13.0, abs=1e-9)

    def test_s2(self, manual_instance):
        ev = evaluate_skeleton(manual_instance, S2, t_station_min=0.0)
        assert ev.times_min[1] == pytest.approx(S2_SKELETON[1], abs=1e-9)
        assert ev.times_min[2] == pytest.approx(S2_SKELETON[2], abs=1e-9)
        assert ev.makespan_min == pytest.approx(12.0, abs=1e-9)


class TestServiceTimeOverlay:
    """T>0 时：完工时间 = 骨架 + 停靠站数×T（每个方案每快递员均停靠 2 站）。"""

    STOPS_PER_COURIER = {1: 2, 2: 2}

    @pytest.mark.parametrize("t", [1.5, 3.0, 5.0])
    def test_completion_equals_skeleton_plus_stops_times_t(self, manual_instance, t):
        for plan, skeleton in ((S1, S1_SKELETON), (S2, S2_SKELETON)):
            ev = evaluate_skeleton(manual_instance, plan, t_station_min=t)
            for cid, base in skeleton.items():
                expected = base + self.STOPS_PER_COURIER[cid] * t
                assert ev.times_min[cid] == pytest.approx(expected, abs=1e-9)

    def test_config_t_is_5_and_applies(self, manual_instance):
        cfg = load_base_config()
        assert cfg.t_station_min == 5.0  # configs/base.yaml 是 T 的唯一来源
        ev = evaluate_skeleton(manual_instance, S1, t_station_min=cfg.t_station_min)
        assert ev.times_min[1] == pytest.approx(13.0 + 2 * 5.0, abs=1e-9)
        assert ev.times_min[2] == pytest.approx(9.0 + 2 * 5.0, abs=1e-9)
        assert ev.makespan_min == pytest.approx(23.0, abs=1e-9)

    def test_evaluator_input_validation(self, manual_instance):
        with pytest.raises(ValueError):
            evaluate_skeleton(manual_instance, {1: [[99]]}, 0.0)  # 未知站点
        with pytest.raises(ValueError):
            evaluate_skeleton(manual_instance, {9: [[1]]}, 0.0)   # 未知快递员
        with pytest.raises(ValueError):
            evaluate_skeleton(manual_instance, {1: [[]]}, 0.0)    # 空趟
