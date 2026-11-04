"""Tests for the final 1000-combination, 10000-instance design."""

import json
from collections import Counter

from qega import constants as C
from qega.instances import (
    COMBINATIONS, REPLICATES, TOTAL_COMBINATIONS, TOTAL_INSTANCES,
    derive_instance_seeds, generate_all, generate_one, iter_instance_specs, main,
)
from qega.model import Instance

MASTER_SEED = 42


def test_full_factorial_combinations_and_replicates():
    assert len(COMBINATIONS) == TOTAL_COMBINATIONS == 1000
    assert TOTAL_INSTANCES == 10000 and REPLICATES == 10
    assert len({p.instance_id for p in COMBINATIONS}) == 1000
    specs = list(iter_instance_specs(MASTER_SEED))
    assert len(specs) == 10000
    assert len({s.instance_id for s in specs}) == 10000
    assert Counter(s.replicate for s in specs) == Counter({r: 1000 for r in range(1, 11)})


def test_factor_level_counts():
    assert Counter(p.N for p in COMBINATIONS) == Counter({n: 200 for n in C.N_LEVELS})
    assert Counter(p.K for p in COMBINATIONS) == Counter({k: 200 for k in C.K_LEVELS})
    assert Counter(p.M for p in COMBINATIONS) == Counter({m: 200 for m in C.M_LEVELS})
    for field in ("scheme_packages", "scheme_weight", "scheme_split"):
        assert Counter(getattr(p, field) for p in COMBINATIONS) == Counter({"R": 500, "U": 500})


def test_seeds_and_generate_one_are_deterministic():
    first = derive_instance_seeds(MASTER_SEED)
    assert first == derive_instance_seeds(MASTER_SEED)
    assert len(first) == len(set(first)) == TOTAL_INSTANCES
    single = generate_one(MASTER_SEED, 0)
    batch = generate_all(MASTER_SEED, limit=1)
    assert single.instance_id.endswith("_r01")
    assert single.to_canonical_bytes() == batch[0].to_canonical_bytes()


def test_subset_is_byte_identical_and_reloadable(tmp_path):
    subset = ["N200_K10_M8_RRR_r01", "N200_K10_M8_UUU_r02",
              "N500_K15_M18_RUU_r03", "N200_K30_M150_UUR_r04"]
    run1 = generate_all(MASTER_SEED, out_dir=tmp_path / "r1", only=subset)
    run2 = generate_all(MASTER_SEED, out_dir=tmp_path / "r2", only=subset)
    assert len(run1) == len(run2) == len(subset)
    for first, second in zip(run1, run2):
        assert first.to_canonical_bytes() == second.to_canonical_bytes()
        payload = (tmp_path / "r1" / f"{first.instance_id}.json").read_bytes()
        clone = Instance.from_canonical_json(payload.decode("utf-8"))
        assert clone.to_canonical_bytes() == first.to_canonical_bytes()
        assert json.loads(payload)["instance_id"] == first.instance_id


def test_cli_limit(tmp_path, capsys):
    assert main(["--seed", str(MASTER_SEED), "--out", str(tmp_path), "--limit", "3"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["generated"] == 3
    assert len(list(tmp_path.glob("*.json"))) == 3
