"""Final-manuscript full-factorial instance design and generation CLI.

There are 5*5*5*2*2*2 = 1000 factor combinations. Ten independent
instances per combination give 10000 instances. Instance, rather than run,
is the independent sample in the main comparisons.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from . import constants as C
from .generator import GeneratorParams, generate_instance
from .model import Instance

TOTAL_COMBINATIONS = 1000
REPLICATES = 10
TOTAL_INSTANCES = TOTAL_COMBINATIONS * REPLICATES


def iter_combinations() -> list[GeneratorParams]:
    combos = []
    for N, K, M in itertools.product(C.N_LEVELS, C.K_LEVELS, C.M_LEVELS):
        for sp, sw, ss in itertools.product(C.SCHEME_LEVELS, repeat=3):
            combos.append(GeneratorParams(
                N=N, K=K, M=M,
                scheme_packages=sp, scheme_weight=sw, scheme_split=ss,
            ))
    return combos


COMBINATIONS: list[GeneratorParams] = iter_combinations()
assert len(COMBINATIONS) == TOTAL_COMBINATIONS
assert len({p.instance_id for p in COMBINATIONS}) == TOTAL_COMBINATIONS


@dataclass(frozen=True)
class InstanceSpec:
    index: int
    replicate: int
    params: GeneratorParams
    seed: int

    @property
    def instance_id(self) -> str:
        return f"{self.params.instance_id}_r{self.replicate:02d}"


def derive_instance_seeds(master_seed: int) -> list[int]:
    ss = np.random.SeedSequence(master_seed)
    return [int(s.generate_state(1)[0]) for s in ss.spawn(TOTAL_INSTANCES)]


def iter_instance_specs(master_seed: int) -> Iterable[InstanceSpec]:
    seeds = derive_instance_seeds(master_seed)
    index = 0
    for params in COMBINATIONS:
        for replicate in range(1, REPLICATES + 1):
            yield InstanceSpec(index, replicate, params, seeds[index])
            index += 1


def generate_spec(spec: InstanceSpec) -> Instance:
    return generate_instance(spec.params, seed=spec.seed, instance_id=spec.instance_id)


def generate_one(master_seed: int, index: int) -> Instance:
    if not 0 <= index < TOTAL_INSTANCES:
        raise IndexError(f"index must be in [0, {TOTAL_INSTANCES - 1}]")
    spec = next(s for s in iter_instance_specs(master_seed) if s.index == index)
    return generate_spec(spec)


def write_instance(instance: Instance, out_dir: Path) -> Path:
    path = out_dir / f"{instance.instance_id}.json"
    path.write_text(instance.to_canonical_json() + "\n", encoding="utf-8")
    return path


def generate_all(
    master_seed: int,
    out_dir: Path | None = None,
    limit: int | None = None,
    only: list[str] | None = None,
) -> list[Instance]:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    only_set = set(only) if only else None
    instances = []
    for spec in iter_instance_specs(master_seed):
        if limit is not None and spec.index >= limit:
            break
        if only_set and spec.instance_id not in only_set:
            continue
        instance = generate_spec(spec)
        if out_dir is not None:
            write_instance(instance, out_dir)
        instances.append(instance)
    return instances


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qega.instances",
        description="Generate the final manuscript's 10000-instance design",
    )
    parser.add_argument("--seed", type=int, required=True, help="master seed")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args(argv)

    instances = generate_all(args.seed, out_dir=args.out, limit=args.limit, only=args.only)
    print(json.dumps({
        "master_seed": args.seed,
        "generated": len(instances),
        "out": str(args.out) if args.out else None,
        "degenerate_marked": sum(i.degenerate for i in instances),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
