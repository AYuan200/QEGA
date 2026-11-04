"""距离矩阵单元测试（SPEC §2 修订：D=ρ·E 地面，D′=E 无人机，随算例落盘）。"""

import pytest

from qega import constants as C
from qega.model import Courier, Instance, Order, Package, Station, VehicleType

VT_GROUND = VehicleType("ground", capacity_kg=10.0, speed_kmh=30.0, range_km=100.0, aerial=False)
VT_DRONE = VehicleType("uav", capacity_kg=5.0, speed_kmh=60.0, range_km=20.0, aerial=True)

# v0=(0,0), v1=(3,4) → 5 km, v2=(0,8) → 8 km, d(v1,v2)=5 km
_E = [[0.0, 5.0, 8.0], [5.0, 0.0, 5.0], [8.0, 5.0, 0.0]]
_RHO = 1.3
_D = [[x * _RHO for x in row] for row in _E]


def _geom_instance():
    stations = [Station(0, 0.0, 0.0), Station(1, 3.0, 4.0), Station(2, 0.0, 8.0)]
    orders = [Order(1, 1, (Package("p", "delivery", 1.0, 0, 1),))]
    couriers = [Courier(1, VT_GROUND), Courier(2, VT_DRONE)]
    return Instance("geom", stations, orders, couriers,
                    dist_ground=[row[:] for row in _D],
                    dist_drone=[row[:] for row in _E])


class TestDistanceMatrix:
    def test_euclidean_baseline(self):
        d = _geom_instance().dist_euclidean
        assert d[0][1] == 5.0
        assert d[0][2] == 8.0
        assert d[1][2] == 5.0

    def test_symmetric_zero_diagonal(self):
        inst = _geom_instance()
        for m in (inst.dist_ground, inst.dist_drone):
            for i in range(3):
                assert m[i][i] == 0.0
                for j in range(3):
                    assert m[i][j] == m[j][i]

    def test_ground_reads_d_matrix(self):
        inst = _geom_instance()
        assert inst.distance_matrix(VT_GROUND) is inst.dist_ground
        assert inst.travel_distance(0, 1, VT_GROUND) == pytest.approx(5.0 * _RHO)

    def test_drone_reads_d_prime_matrix(self):
        inst = _geom_instance()
        assert inst.distance_matrix(VT_DRONE) is inst.dist_drone
        assert inst.travel_distance(0, 1, VT_DRONE) == 5.0

    def test_road_factor_matrix_recovered(self):
        rho = _geom_instance().road_factor_matrix()
        assert rho[0][1] == pytest.approx(_RHO)
        assert rho[1][2] == pytest.approx(_RHO)
        assert rho[0][0] == 1.0

    def test_table5_vehicle_types(self):
        assert C.DRONE.aerial and C.DRONE.is_drone
        assert not C.MINI_VAN.aerial and not C.MINI_VAN.is_drone
        assert not C.ELECTRIC_TRICYCLE.aerial
        assert (C.DRONE.capacity_kg, C.DRONE.speed_kmh, C.DRONE.range_km) == (5.0, 60.0, 20.0)
        assert (C.MINI_VAN.capacity_kg, C.MINI_VAN.speed_kmh, C.MINI_VAN.range_km) == (150.0, 35.0, 300.0)
        assert (C.ELECTRIC_TRICYCLE.capacity_kg, C.ELECTRIC_TRICYCLE.speed_kmh, C.ELECTRIC_TRICYCLE.range_km) == (50.0, 25.0, 80.0)

    def test_road_factor_range_constants(self):
        assert (C.ROAD_FACTOR_MIN, C.ROAD_FACTOR_MAX) == (1.2, 1.4)

    def test_min_interstation_distance_uses_euclidean(self):
        assert _geom_instance().min_interstation_distance() == 5.0
