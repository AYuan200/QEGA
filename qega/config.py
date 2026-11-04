"""Load the frozen algorithm settings used by the final manuscript."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_BASE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
DEFAULT_BASELINE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "baselines.yaml"


@dataclass(frozen=True)
class BaseConfig:
    t_station_min: float = 5.0
    N_p: int = 50
    p_c: float = 0.8
    p_m: float = 0.1
    alpha: float = 0.4
    FE_max: int = 50_000
    init_2opt_passes: int | None = 2


def load_base_config(path: str | Path | None = None) -> BaseConfig:
    p = Path(path) if path is not None else DEFAULT_BASE_CONFIG
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    qega = data.get("qega", {})
    passes = qega.get("init_2opt_passes", 2)
    return BaseConfig(
        t_station_min=float(data["service"]["t_station_min"]),
        N_p=int(qega.get("N_p", 50)),
        p_c=float(qega.get("p_c", 0.8)),
        p_m=float(qega.get("p_m", 0.1)),
        alpha=float(qega.get("alpha", 0.4)),
        FE_max=int(qega.get("FE_max", 50_000)),
        init_2opt_passes=None if passes is None else int(passes),
    )


def load_baseline_config(path: str | Path | None = None) -> dict[str, dict]:
    p = Path(path) if path is not None else DEFAULT_BASELINE_CONFIG
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {
        algorithm: dict(data.get(algorithm, {}))
        for algorithm in ("qsa", "qpso", "sga", "sweep")
    }
