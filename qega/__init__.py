"""QEGA — 算例生成、队列遗传调度与约束校验代码包。"""

from . import constants
from .model import Courier, Instance, Order, Package, Station, VehicleType
from .generator import GeneratorParams, generate_instance
from .config import BaseConfig, load_base_config
from .evaluator import SkeletonEvaluation, evaluate_skeleton

__version__ = "0.1.0"

__all__ = [
    "constants",
    "Courier",
    "Instance",
    "Order",
    "Package",
    "Station",
    "VehicleType",
    "GeneratorParams",
    "generate_instance",
    "BaseConfig",
    "load_base_config",
    "SkeletonEvaluation",
    "evaluate_skeleton",
]
