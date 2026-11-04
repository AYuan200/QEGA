"""Regression tests for behavior explicit in the final manuscript."""

import math
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from qega.decoder import am
from qega.generator import GeneratorParams, generate_instance
from qega.instances import iter_instance_specs
from qega.model import KIND_DELIVERY, Courier, Instance, Order, Package, Station, VehicleType, euclidean_matrix
from qega.operators import initialize_population
from qega.qega import QEGAParams, default_cluster_count


def test_final_qega_defaults_and_cluster_rule():
    params = QEGAParams()
    assert params.N_p == 50 and params.p_c == 0.8 and params.p_m == 0.1
    assert params.alpha == 0.4 and params.FE_max == 50000
    assert params.init_2opt_passes == 2
    assert [default_cluster_count(N) for N in [500, 1000, 2000, 5000]] == [15, 20, 25, 25]


def test_initial_population_has_forty_percent_enhanced_individuals():
    centroids = {i: (float(i), 0.0) for i in range(1, 6)}
    evaluations = []

    def evaluate(chromosome):
        evaluations.append(chromosome)
        return float(sum(chromosome))

    population = initialize_population(
        centroids, 10, 0.4, np.random.default_rng(3), evaluate, two_opt_max_passes=0)
    assert len(population) == 10
    assert len(evaluations) == 4


def test_am_scans_later_queues_when_start_queue_cannot_supply_order():
    stations = [Station(0, 0, 0), Station(1, 6, 0), Station(2, 0, 4)]
    distance = euclidean_matrix(stations)
    vehicle = VehicleType("limited", 20, 60, 10, False)
    orders = [
        Order(1, 1, (Package("p1", KIND_DELIVERY, 1, 0, 1),)),
        Order(2, 2, (Package("p2", KIND_DELIVERY, 1, 0, 2),)),
    ]
    instance = Instance("all_queues", stations, orders, [Courier(1, vehicle)], distance, distance)
    queues = {1: deque([orders[0]]), 2: deque([orders[1]])}
    trip = am(instance, vehicle, queues, 1, queue_order=(1, 2))
    assert trip.order_ids == (2,)
    assert queues[1][0].id == 1 and not queues[2]


def test_full_main_configs_expand_to_10000_instances_each():
    root = Path(__file__).parents[1]
    for algorithm in ["qega", "qsa", "qpso", "sga", "sweep"]:
        config = yaml.safe_load((root / "configs" / f"main_{algorithm}.yaml").read_text())
        assert config["runs"] == 1
        grid = config["instance_grid"]
        specs = list(iter_instance_specs(grid["master_seed"]))
        assert len(specs) == 10000
        assert config.get("fe_max", 50000) == 50000 or algorithm == "sweep"


def test_taguchi_array_matches_l16_levels():
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "configs" / "taguchi_qega.yaml").read_text())
    grid = config["param_grid"]
    assert len(grid) == 16 and config["runs"] == 10
    expected = {
        "n": {10, 15, 20, 25}, "N_p": {50, 100, 150, 200},
        "p_c": {0.6, 0.7, 0.8, 0.9}, "p_m": {0.05, 0.1, 0.15, 0.2},
        "alpha": {0.2, 0.4, 0.6, 0.8},
    }
    for factor, levels in expected.items():
        assert {row[factor] for row in grid} == levels
        assert all(sum(row[factor] == level for row in grid) == 4 for level in levels)
