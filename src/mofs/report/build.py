"""Regenerate every table and figure from raw results.

Nothing here re-runs an experiment. `make all` runs this against whatever is in
`results/`, so a report can never quietly disagree with the data it describes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..selectors.base import jaccard
from ..stats.compare import (
    cliffs_delta,
    cliffs_delta_label,
    friedman_ranks,
    posthoc_holm,
)
from ..stats.multiobj import REFERENCE_POINT, attainment_surface

#: A block is one (dataset, seed, outer fold). Every selector saw exactly the
#: same training rows and the same held-out fold within a block, so the
#: comparison is properly paired -- which is what a signed-rank test requires.
BLOCK = ["dataset", "seed", "fold"]

PALETTE = {
    "Oracle (true support)": "#111111",
    "NSGA-II": "#1f77b4",
    "NSGA-III": "#17becf",
    "Random subsets": "#d62728",
    "t-test + BH": "#2ca02c",
    "mRMR": "#9467bd",
    "LASSO path": "#ff7f0e",
    "Knockoff+": "#8c564b",
    "Knockoff": "#e377c2",
}


def _fmt_iqr(series: pd.Series) -> str:
    q1, med, q3 = series.quantile([0.25, 0.5, 0.75])
    return f"{med:.3f} [{q1:.3f}, {q3:.3f}]"


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median [IQR] of the headline quantities, per dataset and selector."""
    rows = []
    for (ds, sel), g in df.groupby(["dataset", "selector"], sort=False):
        rows.append(
            {
                "dataset": ds,
                "selector": sel,
                "balanced error": _fmt_iqr(g["outer_balanced_error"]),
                "features": _fmt_iqr(g["n_selected"].astype(float)),
                "hypervolume": _fmt_iqr(g["hypervolume"]),
                "precision": _fmt_iqr(g["selection_precision"])
                if g["selection_precision"].notna().any()
                else "-",
                "recall": _fmt_iqr(g["selection_recall"])
                if g["selection_recall"].notna().any()
                else "-",
            }
        )
    return pd.DataFrame(rows)


def pivot_blocks(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Blocks x selectors matrix, dropping blocks a selector is missing from."""
    wide = df.pivot_table(index=BLOCK, columns="selector", values=metric, aggfunc="median")
    return wide.dropna(axis=0, how="any")


def statistical_comparison(df: pd.DataFrame, metric: str, lower_is_better: bool) -> dict:
    """Friedman -> Holm-corrected pairwise Wilcoxon -> Cliff's delta.

    The post-hoc is only computed when Friedman is significant; running it
    otherwise is precisely the multiplicity error the correction exists to
    prevent, so we record that it was skipped rather than reporting it anyway.
    """
    wide = pivot_blocks(df, metric)
    if wide.shape[1] < 3 or wide.shape[0] < 2:
        return {"skipped": "need >=3 selectors and >=2 blocks"}

    # friedman_ranks ranks ascending (rank 1 = smallest). For a
    # higher-is-better metric, negate so rank 1 still means best.
    oriented = wide if lower_is_better else -wide
    stat, p, ranks = friedman_ranks(oriented)

    out = {
        "metric": metric,
        "lower_is_better": lower_is_better,
        "n_blocks": int(wide.shape[0]),
        "friedman_stat": stat,
        "friedman_p": p,
        "mean_ranks": ranks.round(3).to_dict(),
        "posthoc": None,
        "cliffs": None,
    }
    if p >= 0.05:
        out["posthoc_note"] = "Friedman not significant at 0.05; no post-hoc performed."
        return out

    out["posthoc"] = posthoc_holm(oriented).round(5)

    # Cliff's delta on the RAW per-block values, not on the ranks -- the
    # distinction the shared contract insists on.
    best = ranks.index[0]
    deltas = {}
    for other in wide.columns:
        if other == best:
            continue
        d = cliffs_delta(wide[best].to_numpy(), wide[other].to_numpy())
        deltas[other] = {"delta": round(d, 3), "magnitude": cliffs_delta_label(d)}
    out["cliffs"] = {"reference": best, "vs": deltas}
    return out


def budget_table(df: pd.DataFrame) -> pd.DataFrame:
    """Actual unique objective evaluations per fold, per selector.

    The random-subset ablation only means something if random genuinely got the
    same budget as the evolutionary search. Asserting that in the config is not
    the same as observing it: NSGA-II re-visits solutions and the objective
    cache absorbs the duplicates, so its *unique* evaluation count can fall well
    below pop x gens. This table reports what each method actually spent, so a
    reader can check the comparison rather than take it on trust.
    """
    out = (
        df.groupby(["dataset", "selector"])["n_evaluations"]
        .agg(median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75))
        .round(0)
        .astype(int)
        .reset_index()
    )
    return out


def stability_table(df: pd.DataFrame, supports: dict[str, np.ndarray]) -> pd.DataFrame:
    """Across-fold Jaccard overlap of the selected supports, within each seed."""
    rows = []
    for (ds, sel), g in df.groupby(["dataset", "selector"], sort=False):
        per_seed = []
        for seed, gs in g.groupby("seed"):
            keys = [f"{ds}|{sel}|{seed}|{f}" for f in gs["fold"]]
            sup = [supports[k] for k in keys if k in supports]
            sup = [s for s in sup if s.sum() > 0]
            per_seed += [
                jaccard(sup[i], sup[j])
                for i in range(len(sup))
                for j in range(i + 1, len(sup))
            ]
        rows.append(
            {
                "dataset": ds,
                "selector": sel,
                "jaccard_median": round(float(np.median(per_seed)), 3) if per_seed else float("nan"),
                "jaccard_mean": round(float(np.mean(per_seed)), 3) if per_seed else float("nan"),
                "n_pairs": len(per_seed),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def figure_attainment(fronts: dict, df: pd.DataFrame, out: Path) -> None:
    """Median attainment surface per selector -- a distribution of fronts.

    One front from one seed is one sample; plotting only the best of them is
    the mistake the contract warns about. Single-point methods (the knockoff
    filter, the oracle) are drawn as markers rather than steps, because a
    one-point "curve" is invisible next to a staircase.
    """
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.6 * len(datasets), 5.2), squeeze=False)

    for ax, ds in zip(axes[0], datasets):
        x_max = 0.0
        order = sorted(df[df.dataset == ds]["selector"].unique(),
                       key=lambda s: (s != "Oracle (true support)", s))
        for sel in order:
            keys = [k for k in fronts if k.startswith(f"{ds}|{sel}|")]
            collected = [fronts[k] for k in keys if len(fronts[k])]
            if not collected:
                continue
            # No blanket except here: a broken indicator call must fail loudly
            # rather than quietly degrade into a meaningless curve.
            surface = attainment_surface(collected, percentile=50.0)
            if not len(surface):
                continue
            x_max = max(x_max, float(surface[:, 0].max()))
            colour = PALETTE.get(sel, "#777777")

            if len(surface) == 1 or sel == "Oracle (true support)":
                marker = "*" if sel == "Oracle (true support)" else "D"
                size = 320 if sel == "Oracle (true support)" else 70
                ax.scatter(surface[:, 0], surface[:, 1], label=sel, color=colour,
                           marker=marker, s=size, zorder=6,
                           edgecolor="white", linewidth=1.0)
            else:
                o = np.argsort(surface[:, 0])
                ax.step(surface[o, 0], surface[o, 1], where="post",
                        label=sel, color=colour, lw=2.0, alpha=0.9)

        ax.set_title(f"{ds}\nmedian attainment surface (50th percentile of {10} seeds x 5 folds)",
                     fontsize=10)
        ax.set_xlabel("features selected / cap   (minimise)")
        ax.set_ylabel("inner-CV error   (minimise)")
        # Fit the axis to the data. A fixed 0-1 range leaves most of the panel
        # empty, because no sensible selector approaches the cardinality cap.
        ax.set_xlim(-0.01, min(1.02, max(0.12, x_max * 1.12)))
        ax.set_ylim(-0.01, 0.45)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.95)

    fig.suptitle(
        "These are INNER-CV fronts -- the objective the search optimises, not held-out "
        "performance.\n"
        f"Hypervolume reference point {REFERENCE_POINT}; lower-left is better. "
        "Markers denote single-solution methods. Compare against outer-fold error in fig2.",
        y=0.005, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_distributions(df: pd.DataFrame, out: Path) -> None:
    """Hypervolume and outer-error distributions. Never a single bar."""
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(2, len(datasets), figsize=(7 * len(datasets), 9), squeeze=False)
    for col, ds in enumerate(datasets):
        sub = df[df.dataset == ds]
        order = sub.groupby("selector")["outer_balanced_error"].median().sort_values().index
        for row, (metric, label) in enumerate(
            [("hypervolume", "hypervolume (higher better)"),
             ("outer_balanced_error", "outer-fold balanced error (lower better)")]
        ):
            ax = axes[row][col]
            data = [sub[sub.selector == s][metric].dropna().to_numpy() for s in order]
            bp = ax.boxplot(data, tick_labels=list(order), showfliers=False, patch_artist=True)
            for patch, s in zip(bp["boxes"], order):
                patch.set_facecolor(PALETTE.get(s, "#999999"))
                patch.set_alpha(0.65)
            if metric == "outer_balanced_error":
                ax.axhline(0.5, color="crimson", ls=":", lw=1.5, label="chance (balanced)")
                ax.legend(fontsize=8)
            ax.set_title(f"{ds} - {label}", fontsize=10)
            if metric == "hypervolume":
                ax.set_xlabel("")  # shared with the panel below
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.grid(alpha=0.25, axis="y")
    fig.suptitle(
        "Top row: the inner objective the search optimises.  Bottom row: held-out error.  "
        "Both panels share the same selector ordering, so vertical movement between rows is the "
        "gap between optimising and generalising.",
        y=0.005, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_permutation(main: pd.DataFrame, permuted: pd.DataFrame, out: Path) -> None:
    """The leak detector, as a picture."""
    fig, ax = plt.subplots(figsize=(9, 5))
    order = sorted(set(main["selector"]) & set(permuted["selector"]))
    width = 0.38
    x = np.arange(len(order))
    for offset, (frame, label, colour) in enumerate(
        [(main, "real labels", "#1f77b4"), (permuted, "permuted labels", "#d62728")]
    ):
        med = [frame[frame.selector == s]["outer_balanced_error"].median() for s in order]
        q1 = [frame[frame.selector == s]["outer_balanced_error"].quantile(0.25) for s in order]
        q3 = [frame[frame.selector == s]["outer_balanced_error"].quantile(0.75) for s in order]
        err = np.vstack([np.array(med) - np.array(q1), np.array(q3) - np.array(med)])
        ax.bar(x + offset * width, med, width, yerr=err, capsize=3, label=label, color=colour, alpha=0.85)
    ax.axhline(0.5, color="black", ls=":", lw=1.5, label="chance")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("outer-fold balanced error")
    ax.set_title("Permutation check: with labels shuffled, every selector must sit at chance")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def build(results_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(results_dir / "folds_main.csv")
    supports = dict(np.load(results_dir / "supports_main.npz"))
    fronts = dict(np.load(results_dir / "fronts_main.npz"))
    manifest = json.loads((results_dir / "manifest_main.json").read_text())

    permuted_path = results_dir / "folds_permuted.csv"
    permuted = pd.read_csv(permuted_path) if permuted_path.exists() else None

    summary = summary_table(main)
    stability = stability_table(main, supports)
    budget = budget_table(main)
    summary.to_csv(out_dir / "table1_summary.csv", index=False)
    stability.to_csv(out_dir / "table2_stability.csv", index=False)
    budget.to_csv(out_dir / "table3_budget.csv", index=False)

    stats_out = {}
    for ds, g in main.groupby("dataset"):
        stats_out[ds] = {
            "hypervolume": statistical_comparison(g, "hypervolume", lower_is_better=False),
            "outer_balanced_error": statistical_comparison(
                g, "outer_balanced_error", lower_is_better=True
            ),
        }

    figure_attainment(fronts, main, out_dir / "fig1_attainment.png")
    figure_distributions(main, out_dir / "fig2_distributions.png")
    if permuted is not None:
        figure_permutation(main, permuted, out_dir / "fig3_permutation.png")

    _write_markdown(out_dir, manifest, main, permuted, summary, stability, budget, stats_out)
    print(f"  report written to {out_dir}/RESULTS.md")


def _write_markdown(out_dir, manifest, main, permuted, summary, stability, budget, stats_out) -> None:
    L: list[str] = ["# Results\n"]
    L.append(f"Generated from `{manifest['n_fold_results']}` fold results "
             f"({manifest['elapsed_seconds']}s wall clock).\n")
    L.append(f"- Hypervolume reference point: `{tuple(manifest['reference_point'])}`, "
             f"size axis normalised by `{manifest.get('size_scale')}` (the cardinality cap)")
    L.append(f"- Indicator backend: `{manifest['indicator_backend']}`")
    L.append(f"- Master seed `{manifest['config']['seed']}`, "
             f"{manifest['config']['n_seeds']} seeds x {manifest['config']['n_outer']} outer folds")
    L.append(f"- Evolutionary budget: {manifest['config']['pop_size']} x "
             f"{manifest['config']['n_gen']} = "
             f"{manifest['config']['pop_size'] * manifest['config']['n_gen']} evaluations, "
             "matched exactly by the random-subset ablation\n")

    L.append("## Datasets\n")
    for d in manifest["datasets"]:
        L.append(f"- **{d['name']}** - n={d['n']}, p={d['p']}, `sha256:{d['fingerprint']}`  ")
        L.append(f"  {d['provenance']['source']} ({d['provenance']['identifier']}); "
                 f"preprocessing: {', '.join(d['provenance']['preprocessing'])}")
    L.append("")

    L.append("## Table 1 - headline results, median [IQR]\n")
    L.append(summary.to_markdown(index=False))
    L.append("")

    L.append("## Table 2 - across-fold support stability (Jaccard)\n")
    L.append(stability.to_markdown(index=False))
    L.append("")

    L.append("## Table 3 - budget actually spent (unique objective evaluations per fold)\n")
    L.append("The random-subset ablation is only meaningful if random received the same budget as "
             "the evolutionary search. Configured budget is "
             f"{manifest['config']['pop_size']} x {manifest['config']['n_gen']} = "
             f"{manifest['config']['pop_size'] * manifest['config']['n_gen']}; "
             "NSGA-II's *unique* count falls below that wherever the search re-visited "
             "solutions and the objective cache absorbed the duplicates.\n")
    L.append(budget.to_markdown(index=False))
    L.append("")

    L.append("## Table 4 - statistical comparison\n")
    L.append("Friedman across selectors with (dataset, seed, fold) as blocks; post-hoc "
             "pairwise Wilcoxon signed-rank with Holm correction, run only where Friedman "
             "was significant; Cliff's delta against the top-ranked selector.\n")
    for ds, per_metric in stats_out.items():
        L.append(f"### {ds}\n")
        for metric, res in per_metric.items():
            if "skipped" in res:
                L.append(f"**{metric}** - skipped: {res['skipped']}\n")
                continue
            L.append(f"**{metric}** ({'lower' if res['lower_is_better'] else 'higher'} is better), "
                     f"{res['n_blocks']} blocks  ")
            L.append(f"Friedman chi2 = {res['friedman_stat']:.2f}, p = {res['friedman_p']:.3e}\n")
            ranks = pd.Series(res["mean_ranks"]).sort_values()
            L.append("Mean ranks (1 = best): " +
                     ", ".join(f"`{k}` {v:.2f}" for k, v in ranks.items()) + "\n")
            if res.get("posthoc_note"):
                L.append(f"_{res['posthoc_note']}_\n")
            if res["cliffs"]:
                ref = res["cliffs"]["reference"]
                L.append(f"Cliff's delta vs **{ref}** (raw per-block values):\n")
                rows = [{"selector": k, "delta": v["delta"], "magnitude": v["magnitude"]}
                        for k, v in res["cliffs"]["vs"].items()]
                L.append(pd.DataFrame(rows).to_markdown(index=False))
                L.append("")
            if res["posthoc"] is not None:
                L.append("<details><summary>Holm-corrected pairwise p-values</summary>\n")
                L.append(res["posthoc"].to_markdown())
                L.append("\n</details>\n")

    if permuted is not None:
        L.append("## Table 5 - permutation check\n")
        L.append("Labels shuffled before splitting. Balanced error must sit at chance (0.5). "
                 "Anything meaningfully below it means selection saw the outer test fold.\n")
        comp = (
            permuted.groupby("selector")["outer_balanced_error"].median().rename("permuted").to_frame()
            .join(main.groupby("selector")["outer_balanced_error"].median().rename("real labels"))
            .round(3).reset_index()
        )
        L.append(comp.to_markdown(index=False))
        worst = permuted.groupby("selector")["outer_balanced_error"].median().min()
        verdict = "PASS" if worst > 0.40 else "FAIL - possible leak"
        L.append(f"\n**Verdict: {verdict}** (lowest permuted median = {worst:.3f})\n")

    L.append("## Figures\n")
    L.append("![attainment](fig1_attainment.png)\n")
    L.append("![distributions](fig2_distributions.png)\n")
    if permuted is not None:
        L.append("![permutation](fig3_permutation.png)\n")

    (out_dir / "RESULTS.md").write_text("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build tables and figures from raw results.")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("report"))
    a = ap.parse_args(argv)
    build(a.results, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
