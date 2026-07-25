"""Experiment orchestration.

One command runs the whole study; `make all` then regenerates every table and
figure from the raw results it writes. Nothing downstream re-runs an
experiment, so a report can never silently disagree with the data it claims to
describe.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from . import data as data_mod
from .nested_cv import run_nested_cv
from .selectors.baselines import (
    LassoSelector,
    MRMRSelector,
    OracleSelector,
    RandomSubsetSelector,
    UnivariateFilterSelector,
)
from .selectors.evolutionary import NSGA2Selector, NSGA3Selector
from .selectors.knockoff import KnockoffSelector
from .stats.multiobj import REFERENCE_POINT, backend


def build_selector_factories(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map selector names to zero-argument factories.

    The evolutionary budget and the random-subset budget are derived from the
    *same* config values, so they cannot drift apart. That matching is what
    makes the random-subset ablation mean anything.
    """
    pop = int(cfg["pop_size"])
    gens = int(cfg["n_gen"])
    budget = pop * gens
    max_feat = int(cfg["max_features"])
    k_init = (int(cfg["k_init_min"]), int(cfg["k_init_max"]))

    return {
        "NSGA-II": lambda: NSGA2Selector(pop, gens, max_feat, k_init),
        "NSGA-III": lambda: NSGA3Selector(pop, gens, max_feat, k_init),
        "Random subsets": lambda: RandomSubsetSelector(budget, k_init),
        "t-test + BH": lambda: UnivariateFilterSelector(q=float(cfg["fdr_q"])),
        "mRMR": lambda: MRMRSelector(k_max=min(50, max_feat)),
        "LASSO path": lambda: LassoSelector(),
        # Both offsets are run. Knockoff+ (offset=1) cannot make fewer than
        # ceil(1/q) discoveries by construction, so at q=0.1 it is structurally
        # unable to select anything unless 10+ features clear the threshold.
        # Running only the "+" variant would report that floor as zero power.
        "Knockoff+": lambda: KnockoffSelector(
            q=float(cfg["fdr_q"]), screen_to=int(cfg["knockoff_screen_to"]), offset=1
        ),
        "Knockoff": lambda: KnockoffSelector(
            q=float(cfg["fdr_q"]), screen_to=int(cfg["knockoff_screen_to"]), offset=0
        ),
        "Oracle (true support)": OracleSelector,
    }


#: Selectors that are only defined where the ground-truth support is known.
SYNTHETIC_ONLY = {"Oracle (true support)"}


def _task(dataset, factory, name, seed, cfg, permute):
    rows = run_nested_cv(
        dataset,
        factory,
        name,
        seed=seed,
        n_outer=int(cfg["n_outer"]),
        inner_folds=int(cfg["inner_folds"]),
        permute_labels=permute,
        size_scale=int(cfg["max_features"]),
    )
    return rows


def run(config_path: Path, out_dir: Path, permute: bool = False, quick: bool = False) -> Path:
    # Convergence warnings from thousands of tiny logistic fits are expected and
    # would otherwise bury the progress output. Errors are untouched.
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    cfg = yaml.safe_load(config_path.read_text())
    if quick:
        cfg.update(cfg.get("quick_overrides", {}))

    out_dir.mkdir(parents=True, exist_ok=True)
    factories = build_selector_factories(cfg)
    selected = [s for s in cfg["selectors"] if s in factories]
    unknown = set(cfg["selectors"]) - set(factories)
    if unknown:
        raise KeyError(f"unknown selectors in config: {sorted(unknown)}")

    datasets = []
    for spec in cfg["datasets"]:
        try:
            datasets.append(data_mod.load(spec))
        except Exception as exc:  # network failure on OpenML must not kill the run
            print(f"  ! skipping dataset {spec}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if not datasets:
        raise RuntimeError("no datasets could be loaded")

    for ds in datasets:
        print(f"  {ds.describe()}")

    seeds = [int(cfg["seed"]) + i for i in range(int(cfg["n_seeds"]))]
    # The oracle is undefined on real data, and permuting labels destroys the
    # correspondence between y and the planted support, so it is dropped there too.
    jobs = [
        (ds, factories[name], name, seed)
        for ds in datasets
        for name in selected
        for seed in seeds
        if not (name in SYNTHETIC_ONLY and (ds.true_support is None or permute))
    ]
    print(f"\n  {len(jobs)} runs over {len(datasets)} datasets x {len(selected)} selectors x {len(seeds)} seeds")
    print(f"  each over {cfg['n_outer']} outer folds -> {len(jobs) * int(cfg['n_outer'])} fold results")
    if permute:
        print("  LABELS PERMUTED -- expecting chance-level outer error")

    started = time.time()
    batches = Parallel(n_jobs=int(cfg.get("n_jobs", -1)), verbose=5)(
        delayed(_task)(ds, fac, name, seed, cfg, permute) for ds, fac, name, seed in jobs
    )
    elapsed = time.time() - started

    fold_results = [r for batch in batches for r in batch]
    frame = pd.DataFrame([r.as_row() for r in fold_results])

    supports = {
        f"{r.dataset}|{r.selector}|{r.seed}|{r.fold}": r.support for r in fold_results
    }
    fronts = {
        f"{r.dataset}|{r.selector}|{r.seed}|{r.fold}": r.front
        for r in fold_results
        if r.front is not None and len(r.front)
    }
    metas = {
        f"{r.dataset}|{r.selector}|{r.seed}|{r.fold}": r.meta for r in fold_results
    }

    tag = "permuted" if permute else "main"
    frame.to_csv(out_dir / f"folds_{tag}.csv", index=False)
    np.savez_compressed(out_dir / f"supports_{tag}.npz", **supports)
    np.savez_compressed(out_dir / f"fronts_{tag}.npz", **fronts)

    manifest = {
        "config": cfg,
        "config_path": str(config_path),
        "permuted": permute,
        "quick": quick,
        "reference_point": list(REFERENCE_POINT),
        "size_scale": int(cfg["max_features"]),
        "indicator_backend": backend(),
        "elapsed_seconds": round(elapsed, 1),
        "n_fold_results": len(fold_results),
        "datasets": [
            {
                "name": d.name,
                "n": d.n,
                "p": d.p,
                "fingerprint": d.fingerprint(),
                "provenance": d.provenance.as_dict(),
                "true_support_known": d.true_support is not None,
            }
            for d in datasets
        ],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "selector_meta_sample": {
            k: {kk: vv for kk, vv in v.items() if kk != "order"}
            for k, v in list(metas.items())[:3]
        },
    }
    (out_dir / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\n  wrote {len(frame)} fold results to {out_dir}/folds_{tag}.csv in {elapsed:.1f}s")
    return out_dir / f"folds_{tag}.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the feature-selection study.")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--permute",
        action="store_true",
        help="Shuffle labels before splitting. Outer error must land at chance.",
    )
    parser.add_argument("--quick", action="store_true", help="Apply quick_overrides from the config.")
    args = parser.parse_args(argv)
    run(args.config, args.out, permute=args.permute, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
