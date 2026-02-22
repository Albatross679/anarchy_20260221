#!/usr/bin/env python3
"""Multi-signal building scoring for energy investment prioritization.

Computes four independent signals per building, then combines them into
a weighted composite score for ranking.

Signals:
    1. Residual magnitude  – |mean residual| z-scored within utility
    2. Weather sensitivity – consumption spike in extreme weather vs peers
    3. Baseline load ratio – off-hours / peak-hours consumption ratio
    4. Variability score   – residual std normalized by mean consumption
    5. Peer comparison     – z-score within same building-size tier

Usage:
    python -m src.scoring                          # all 5 utilities
    python -m src.scoring --utilities ELECTRICITY  # single utility
    python -m src.scoring --weights 0.3 0.2 0.2 0.15 0.15
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Default configuration ────────────────────────────────────────────

UTILITY_DIRS = {
    "ELECTRICITY": "output/electricity_xgboost_20260222_021830",
    "COOLING": "output/cooling_xgboost_20260222_021836",
    "GAS": "output/gas_xgboost_20260222_021833",
    "HEAT": "output/heat_xgboost_20260222_021837",
    "STEAM": "output/steam_xgboost_20260222_021840",
}

BUILDING_METADATA = "data/building_metadata.csv"
OUT_DIR = Path("doc/scoring")

# Size tiers for peer comparison (sqft)
SIZE_TIERS = [
    (0, 10_000, "Small (<10k sqft)"),
    (10_000, 50_000, "Medium (10k-50k sqft)"),
    (50_000, 150_000, "Large (50k-150k sqft)"),
    (150_000, float("inf"), "Very Large (>150k sqft)"),
]

DEFAULT_WEIGHTS = {
    "residual": 0.30,
    "weather_sensitivity": 0.20,
    "baseline_load": 0.20,
    "variability": 0.15,
    "peer_comparison": 0.15,
}


# ── Signal computation ───────────────────────────────────────────────


def compute_residual_signal(preds: pd.DataFrame) -> pd.DataFrame:
    """Signal 1: absolute mean residual per building, z-scored."""
    gb = preds.groupby("buildingnumber")
    df = pd.DataFrame({
        "mean_residual": gb["residual"].mean(),
        "abs_mean_residual": gb["residual"].mean().abs(),
        "std_residual": gb["residual"].std(),
        "n_samples": gb["residual"].count(),
    })
    df["residual_z"] = stats.zscore(df["abs_mean_residual"], nan_policy="omit")
    return df


def compute_weather_sensitivity(preds: pd.DataFrame) -> pd.DataFrame:
    """Signal 2: how much does consumption spike in extreme weather?

    Compare mean energy_per_sqft in top/bottom 10% temperature conditions
    vs the middle 80%.  Higher ratio = more weather-sensitive = worse.
    """
    temp = preds["temperature_2m"]
    q10, q90 = temp.quantile(0.10), temp.quantile(0.90)

    extreme = preds[(temp <= q10) | (temp >= q90)]
    normal = preds[(temp > q10) & (temp < q90)]

    ext_mean = extreme.groupby("buildingnumber")["energy_per_sqft"].mean()
    nor_mean = normal.groupby("buildingnumber")["energy_per_sqft"].mean()

    ratio = (ext_mean / nor_mean.replace(0, np.nan)).rename("weather_ratio")
    df = ratio.to_frame()
    df["weather_sensitivity_z"] = stats.zscore(
        df["weather_ratio"].fillna(df["weather_ratio"].median()),
        nan_policy="omit",
    )
    return df


def compute_baseline_load(preds: pd.DataFrame) -> pd.DataFrame:
    """Signal 3: off-hours consumption / peak-hours consumption.

    Off-hours: weekends + weekday 22:00–06:00
    Peak hours: weekday 08:00–18:00
    Higher ratio → building wastes energy during off-hours.
    """
    weekday_mask = preds["is_weekend"] == 0
    hour = preds["hour_of_day"]

    off_hours = preds[
        (preds["is_weekend"] == 1) | ((weekday_mask) & ((hour < 6) | (hour >= 22)))
    ]
    peak_hours = preds[weekday_mask & (hour >= 8) & (hour < 18)]

    off_mean = off_hours.groupby("buildingnumber")["energy_per_sqft"].mean()
    peak_mean = peak_hours.groupby("buildingnumber")["energy_per_sqft"].mean()

    ratio = (off_mean / peak_mean.replace(0, np.nan)).rename("baseline_load_ratio")
    df = ratio.to_frame()
    df["baseline_load_z"] = stats.zscore(
        df["baseline_load_ratio"].fillna(df["baseline_load_ratio"].median()),
        nan_policy="omit",
    )
    return df


def compute_variability(preds: pd.DataFrame) -> pd.DataFrame:
    """Signal 4: residual std / mean consumption (coefficient of variation of error).

    High variability = unpredictable consumption = potential control issues.
    """
    gb = preds.groupby("buildingnumber")
    std_res = gb["residual"].std()
    mean_energy = gb["energy_per_sqft"].mean()

    cv = (std_res / mean_energy.replace(0, np.nan)).rename("variability_cv")
    df = cv.to_frame()
    df["variability_z"] = stats.zscore(
        df["variability_cv"].fillna(df["variability_cv"].median()),
        nan_policy="omit",
    )
    return df


def compute_peer_comparison(
    preds: pd.DataFrame, building_meta: pd.DataFrame
) -> pd.DataFrame:
    """Signal 5: z-score of |mean residual| within same building-size tier.

    Buildings compared only against peers of similar size.
    """
    gb = preds.groupby("buildingnumber")
    abs_res = gb["residual"].mean().abs().rename("abs_mean_residual_peer")

    # Get grossarea per building
    area = (
        building_meta[["buildingnumber", "grossarea"]]
        .drop_duplicates("buildingnumber")
        .set_index("buildingnumber")["grossarea"]
    )

    df = abs_res.to_frame()
    df["grossarea"] = area
    df["grossarea"] = df["grossarea"].fillna(df["grossarea"].median())

    # Assign size tier
    df["size_tier"] = "Unknown"
    for lo, hi, label in SIZE_TIERS:
        mask = (df["grossarea"] >= lo) & (df["grossarea"] < hi)
        df.loc[mask, "size_tier"] = label

    # Z-score within tier
    df["peer_z"] = df.groupby("size_tier")["abs_mean_residual_peer"].transform(
        lambda x: stats.zscore(x, nan_policy="omit") if len(x) > 1 else 0.0
    )
    return df[["size_tier", "peer_z"]]


# ── Composite scoring ────────────────────────────────────────────────


def score_utility(
    preds: pd.DataFrame,
    building_meta: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Compute all 5 signals and combine into a composite score for one utility."""
    w = weights or DEFAULT_WEIGHTS

    # Ensure buildingnumber is string for consistent joins
    preds = preds.copy()
    preds["buildingnumber"] = preds["buildingnumber"].astype(str)
    building_meta = building_meta.copy()
    building_meta["buildingnumber"] = building_meta["buildingnumber"].astype(str)

    s1 = compute_residual_signal(preds)
    s2 = compute_weather_sensitivity(preds)
    s3 = compute_baseline_load(preds)
    s4 = compute_variability(preds)
    s5 = compute_peer_comparison(preds, building_meta)

    # Join all signals
    df = s1[["mean_residual", "abs_mean_residual", "std_residual", "n_samples", "residual_z"]]
    df = df.join(s2[["weather_ratio", "weather_sensitivity_z"]], how="left")
    df = df.join(s3[["baseline_load_ratio", "baseline_load_z"]], how="left")
    df = df.join(s4[["variability_cv", "variability_z"]], how="left")
    df = df.join(s5[["size_tier", "peer_z"]], how="left")

    # Fill NaN z-scores with 0 (neutral)
    for col in ["residual_z", "weather_sensitivity_z", "baseline_load_z",
                 "variability_z", "peer_z"]:
        df[col] = df[col].fillna(0.0)

    # Composite score (higher = worse performer = higher priority for investment)
    df["composite_score"] = (
        w["residual"] * df["residual_z"]
        + w["weather_sensitivity"] * df["weather_sensitivity_z"]
        + w["baseline_load"] * df["baseline_load_z"]
        + w["variability"] * df["variability_z"]
        + w["peer_comparison"] * df["peer_z"]
    )

    df["rank"] = df["composite_score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("rank")


def score_all_utilities(
    utility_dirs: dict | None = None,
    building_meta_path: str = BUILDING_METADATA,
    weights: dict | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Score every utility and produce a cross-utility composite ranking.

    Returns:
        per_utility: dict mapping utility name → per-building scoring DataFrame
        composite:   cross-utility composite ranking DataFrame
    """
    utility_dirs = utility_dirs or UTILITY_DIRS
    building_meta = pd.read_csv(building_meta_path)

    per_utility = {}
    all_scores = []

    for utility, dir_path in utility_dirs.items():
        pred_path = Path(dir_path) / "predictions.parquet"
        if not pred_path.exists():
            print(f"  [SKIP] {utility}: {pred_path} not found")
            continue

        print(f"  Scoring {utility}...")
        preds = pd.read_parquet(pred_path)
        scored = score_utility(preds, building_meta, weights)
        scored["utility"] = utility
        per_utility[utility] = scored

        # Collect for cross-utility composite
        all_scores.append(
            scored[["composite_score"]].rename(
                columns={"composite_score": f"score_{utility.lower()}"}
            )
        )

    # Cross-utility composite: average composite score across utilities
    if all_scores:
        composite = all_scores[0]
        for s in all_scores[1:]:
            composite = composite.join(s, how="outer")

        score_cols = [c for c in composite.columns if c.startswith("score_")]
        composite["cross_utility_score"] = composite[score_cols].mean(axis=1)
        composite["n_utilities"] = composite[score_cols].notna().sum(axis=1)
        composite["cross_utility_rank"] = (
            composite["cross_utility_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        # Add building metadata
        bldg = building_meta[["buildingnumber", "buildingname", "grossarea"]].copy()
        bldg["buildingnumber"] = bldg["buildingnumber"].astype(str)
        bldg = bldg.drop_duplicates("buildingnumber").set_index("buildingnumber")
        composite = composite.join(bldg, how="left")

        composite = composite.sort_values("cross_utility_rank")
    else:
        composite = pd.DataFrame()

    return per_utility, composite


# ── Visualization ────────────────────────────────────────────────────


def plot_composite_ranking(composite: pd.DataFrame, out_dir: Path, top_n: int = 25):
    """Horizontal bar chart of top buildings by cross-utility composite score."""
    top = composite.head(top_n).copy()
    top["label"] = top.apply(
        lambda r: f"{r.name} – {r.get('buildingname', 'N/A')}" if pd.notna(r.get("buildingname")) else str(r.name),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top)))
    ax.barh(range(len(top)), top["cross_utility_score"].values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Composite Investment Priority Score")
    ax.set_title(f"Top {top_n} Buildings for Energy Efficiency Investment")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_dir / "composite_ranking.png", dpi=150)
    plt.close()
    print(f"  Saved {out_dir / 'composite_ranking.png'}")


def plot_signal_heatmap(per_utility: dict, out_dir: Path, top_n: int = 20):
    """Heatmap of the 5 signal z-scores for top buildings across utilities."""
    signal_cols = [
        "residual_z", "weather_sensitivity_z", "baseline_load_z",
        "variability_z", "peer_z",
    ]
    signal_labels = [
        "Residual", "Weather\nSensitivity", "Baseline\nLoad",
        "Variability", "Peer\nComparison",
    ]

    # Pick top buildings from first utility
    first_utility = list(per_utility.values())[0]
    top_buildings = first_utility.head(top_n).index.tolist()

    fig, axes = plt.subplots(1, len(per_utility), figsize=(4 * len(per_utility), 10),
                             sharey=True)
    if len(per_utility) == 1:
        axes = [axes]

    for ax, (utility, scored) in zip(axes, per_utility.items()):
        sub = scored.loc[scored.index.isin(top_buildings), signal_cols].reindex(top_buildings)
        im = ax.imshow(sub.values, aspect="auto", cmap="RdYlGn_r", vmin=-2, vmax=3)
        ax.set_xticks(range(len(signal_labels)))
        ax.set_xticklabels(signal_labels, fontsize=7, rotation=45, ha="right")
        if ax == axes[0]:
            ax.set_yticks(range(len(top_buildings)))
            ax.set_yticklabels(top_buildings, fontsize=8)
        ax.set_title(utility, fontsize=10)

    fig.colorbar(im, ax=axes, label="Z-score (higher = worse)", shrink=0.6)
    fig.suptitle("Signal Breakdown by Utility (Top Buildings)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "signal_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir / 'signal_heatmap.png'}")


def plot_signal_radar(scored: pd.DataFrame, building_id: str, utility: str, out_dir: Path):
    """Radar/spider chart of 5 signals for a single building."""
    signal_cols = [
        "residual_z", "weather_sensitivity_z", "baseline_load_z",
        "variability_z", "peer_z",
    ]
    labels = ["Residual", "Weather\nSensitivity", "Baseline\nLoad",
              "Variability", "Peer\nComparison"]

    if building_id not in scored.index:
        return

    values = scored.loc[building_id, signal_cols].values.tolist()
    values += values[:1]  # close the polygon

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.25, color="red")
    ax.plot(angles, values, "o-", color="red", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"Building {building_id} — {utility}", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(out_dir / f"radar_{building_id}_{utility.lower()}.png", dpi=150)
    plt.close()


# ── I/O ──────────────────────────────────────────────────────────────


def save_results(
    per_utility: dict[str, pd.DataFrame],
    composite: pd.DataFrame,
    out_dir: Path,
):
    """Save all scoring results to CSVs and generate summary report."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-utility CSVs
    for utility, df in per_utility.items():
        path = out_dir / f"scores_{utility.lower()}.csv"
        df.to_csv(path)
        print(f"  Saved {path} ({len(df)} buildings)")

    # Composite ranking
    if not composite.empty:
        composite.to_csv(out_dir / "composite_ranking.csv")
        print(f"  Saved {out_dir / 'composite_ranking.csv'} ({len(composite)} buildings)")

    # Summary report
    report_lines = ["=" * 70, "MULTI-SIGNAL BUILDING SCORING REPORT", "=" * 70, ""]

    report_lines.append(f"Weights: {json.dumps(DEFAULT_WEIGHTS, indent=2)}")
    report_lines.append(f"Utilities scored: {list(per_utility.keys())}")
    report_lines.append(f"Total buildings ranked: {len(composite)}")
    report_lines.append("")

    if not composite.empty:
        report_lines.append("TOP 15 BUILDINGS FOR INVESTMENT:")
        report_lines.append("-" * 50)
        for i, (bldg_id, row) in enumerate(composite.head(15).iterrows(), 1):
            name = row.get("buildingname", "N/A")
            area = row.get("grossarea", "N/A")
            score = row["cross_utility_score"]
            n_util = int(row["n_utilities"])
            report_lines.append(
                f"  {i:2d}. Building {bldg_id} ({name}) — "
                f"Score: {score:.3f}, Area: {area}, Utilities: {n_util}"
            )
        report_lines.append("")

    # Per-utility top 5
    for utility, df in per_utility.items():
        report_lines.append(f"\n{utility} — Top 5:")
        report_lines.append("-" * 40)
        for i, (bldg_id, row) in enumerate(df.head(5).iterrows(), 1):
            report_lines.append(
                f"  {i}. Bldg {bldg_id}: composite={row['composite_score']:.3f}  "
                f"residual_z={row['residual_z']:.2f}  "
                f"weather_z={row['weather_sensitivity_z']:.2f}  "
                f"baseline_z={row['baseline_load_z']:.2f}  "
                f"var_z={row['variability_z']:.2f}  "
                f"peer_z={row['peer_z']:.2f}"
            )

    report_text = "\n".join(report_lines)
    report_path = out_dir / "scoring_report.txt"
    report_path.write_text(report_text)
    print(f"  Saved {report_path}")

    return report_text


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Multi-signal building scoring")
    parser.add_argument(
        "--utilities", nargs="+", default=list(UTILITY_DIRS.keys()),
        help="Utilities to score (default: all)",
    )
    parser.add_argument(
        "--weights", nargs=5, type=float, default=None,
        help="Weights: residual weather baseline variability peer (must sum ~1.0)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(OUT_DIR),
        help="Output directory",
    )
    args = parser.parse_args()

    # Parse weights
    weights = None
    if args.weights:
        keys = ["residual", "weather_sensitivity", "baseline_load", "variability", "peer_comparison"]
        weights = dict(zip(keys, args.weights))

    # Filter utility dirs
    dirs = {u: UTILITY_DIRS[u] for u in args.utilities if u in UTILITY_DIRS}
    out_dir = Path(args.out_dir)

    print("Multi-Signal Building Scoring")
    print("=" * 40)
    per_utility, composite = score_all_utilities(dirs, BUILDING_METADATA, weights)

    print("\nSaving results...")
    report = save_results(per_utility, composite, out_dir)

    print("\nGenerating plots...")
    if not composite.empty:
        plot_composite_ranking(composite, out_dir)
    if per_utility:
        plot_signal_heatmap(per_utility, out_dir)

    # Radar charts for top 5 buildings per utility
    for utility, scored in per_utility.items():
        for bldg_id in scored.head(5).index:
            plot_signal_radar(scored, bldg_id, utility, out_dir)

    print("\n" + report)
    print("\nDone!")


if __name__ == "__main__":
    main()
