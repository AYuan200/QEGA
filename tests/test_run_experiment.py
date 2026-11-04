"""run_experiment.py 单元测试：任务展开、CSV schema、断点续跑、可行性拒绝入库。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

import run_experiment
from run_experiment import _build_tasks, main

ROOT = Path(__file__).resolve().parent.parent


def _make_config(tmp_path: Path, **overrides) -> Path:
    cfg = {
        "experiment": "testexp",
        "algorithm": "qega",
        "output": str(tmp_path / "out.csv"),
        "instance_dir": str(tmp_path / "instances"),
        "base_config": "configs/base.yaml",
        "fe_max": 400,
        "runs": 2,
        "seeds": [101, 102],
        "instances": [
            {
                "id": "N100_K10_M8_RRR_t1",
                "generate": {"N": 100, "K": 10, "M": 8, "scheme": "RRR", "seed": 9001},
            }
        ],
        "anchor": {"N_p": 10, "p_c": 0.9, "p_m": 0.1, "alpha": 0.4},
        "param_grid": [
            {"label": "anchor"},
            {"label": "pm0.2", "p_m": 0.2},
        ],
        **overrides,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_build_tasks_expands_grid(tmp_path):
    cfg_path = _make_config(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    tasks, output = _build_tasks(cfg, ROOT)
    assert output == ROOT / str(tmp_path / "out.csv")
    assert len(tasks) == 1 * 2 * 2  # 1 算例 x 2 组合 x 2 运行
    by_label = {}
    for t in tasks:
        params = json.loads(t["params_json"])
        by_label.setdefault(params["label"], params)
    assert by_label["anchor"]["p_m"] == 0.1
    assert by_label["pm0.2"]["p_m"] == 0.2
    assert all(t["seed"] in (101, 102) for t in tasks)
    # 算例已按 generate 落盘
    assert (tmp_path / "instances" / "N100_K10_M8_RRR_t1.json").exists()


def test_run_and_resume(tmp_path):
    cfg_path = _make_config(tmp_path)
    out_csv = tmp_path / "out.csv"

    assert main(["--config", str(cfg_path)]) == 0
    with out_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert list(rows[0].keys()) == run_experiment.CSV_FIELDS
    for row in rows:
        assert row["feasible"] == "True"
        assert int(row["num_evals"]) == 400
        assert row["type"] == "Type-R"
        assert float(row["drone_ratio"]) > 0

    # 未加 --resume 时拒绝覆盖
    with pytest.raises(SystemExit):
        main(["--config", str(cfg_path)])

    # --resume 幂等：不追加重复行
    assert main(["--config", str(cfg_path), "--resume"]) == 0
    with out_csv.open(newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 4


def test_infeasible_rejected(tmp_path, monkeypatch):
    cfg_path = _make_config(tmp_path)
    monkeypatch.setattr(
        run_experiment, "validate_solution", lambda inst, sol: ["fake violation"]
    )
    assert main(["--config", str(cfg_path), "--limit", "1"]) == 1
    out_csv = tmp_path / "out.csv"
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8") as f:
            assert list(csv.DictReader(f)) == []  # 只有表头，无可行行入库
