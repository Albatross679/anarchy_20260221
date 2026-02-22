#!/usr/bin/env python3
"""Cross-utility comparison of XGBoost model outputs.

Compares model performance across ELECTRICITY, COOLING, GAS, HEAT, STEAM
and produces tables + plots for investment prioritization.

Usage:
    python compare_utilities.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Configuration ─────────────────────────────────────────────────────

UTILITY_DIRS = {
    "ELECTRICITY": "output/electricity_xgboost_20260222_021830",
    "COOLING": "output/cooling_xgboost_20260222_021836",
    "GAS": "output/gas_xgboost_20260222_021833",
    "HEAT": "output/heat_xgboost_20260222_021837",
    "STEAM": "output/steam_xgboost_20260222_021840",
}

OUT_DIR = Path("doc/cross_utility_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "ELECTRICITY": "#2196F3",
    "COOLING": "#00BCD4",
    "GAS": "#FF9800",
    "HEAT": "#F44336",
    "STEAM": "#9C27B0",
}


# ── Data loading ──────────────────────────────────────────────────────

def load_all():
    """Load metrics, predictions, and per-building metrics for all utilities."""
    metrics = {}
    predictions = {}
    per_building = {}

    for utility, dir_path in UTILITY_DIRS.items():
        d = Path(dir_path)

        # metrics.json
        with open(d / "metrics.json") as f:
            metrics[utility] = json.load(f)

        # predictions.parquet
        predictions[utility] = pd.read_parquet(d / "predictions.parquet")

        # per-building metrics
        per_building[utility] = pd.read_csv(d / "postprocess_per_building_metrics.csv")

    return metrics, predictions, per_building


# ── Table 1: Overall metrics ─────────────────────────────────────────

def table_overall_metrics(metrics: dict, predictions: dict) -> pd.DataFrame:
    """Side-by-side R², RMSE, MAE, n_buildings, n_test per utility."""
    rows = []
    for utility in UTILITY_DIRS:
        m = metrics[utility]
        pred = predictions[utility]
        rows.append({
            "Utility": utility,
            "R²": m["r2"],
            "RMSE": m["rmse"],
            "MAE": m["mae"],
            "Buildings": pred["simscode"].nunique(),
            "Test Samples": m["n_test"],
            "Trees Used": m.get("n_trees_used", "N/A"),
        })
    df = pd.DataFrame(rows)
    return df


# ── Table 2: Building coverage matrix ────────────────────────────────

def table_building_coverage(predictions: dict) -> pd.DataFrame:
    """Binary matrix: which buildings have which utilities."""
    all_buildings = set()
    building_utils = {}
    for utility, pred in predictions.items():
        buildings = set(pred["buildingnumber"].unique())
        all_buildings |= buildings
        for b in buildings:
            building_utils.setdefault(b, set()).add(utility)

    rows = []
    for b in sorted(all_buildings, key=lambda x: str(x)):
        row = {"building": b}
        for utility in UTILITY_DIRS:
            row[utility] = 1 if utility in building_utils.get(b, set()) else 0
        row["n_utilities"] = sum(row[u] for u in UTILITY_DIRS)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("n_utilities", ascending=False)
    return df


# ── Table 3: Top worst buildings per utility ─────────────────────────

def table_worst_buildings(per_building: dict, n: int = 10) -> dict[str, pd.DataFrame]:
    """Top N worst-performing buildings per utility (highest mean |residual|)."""
    result = {}
    for utility in UTILITY_DIRS:
        bld = per_building[utility].copy()
        bld["abs_mean_residual"] = bld["mean_residual"].abs()
        result[utility] = bld.nlargest(n, "abs_mean_residual")[
            ["building", "r2", "rmse", "mae", "mean_residual", "std_residual", "n_samples"]
        ]
    return result


# ── Plot 1: R² bar chart ─────────────────────────────────────────────

def plot_r2_bar(metrics: dict):
    """Bar chart of overall R² per utility."""
    utilities = list(UTILITY_DIRS.keys())
    r2_vals = [metrics[u]["r2"] for u in utilities]
    colors = [COLORS[u] for u in utilities]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(utilities, r2_vals, color=colors, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("XGBoost Model R² by Utility", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0.9, color="green", linestyle="--", alpha=0.5, label="R²=0.9 threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_r2_by_utility.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / '01_r2_by_utility.png'}")


# ── Plot 2: Per-building R² boxplot ──────────────────────────────────

def plot_r2_boxplot(per_building: dict):
    """Boxplot of per-building R² across utilities."""
    utilities = list(UTILITY_DIRS.keys())

    # Clip extreme R² for visualization (some are -1e24)
    data = []
    for u in utilities:
        r2_vals = per_building[u]["r2"].clip(lower=-2.0)
        data.append(r2_vals)

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=utilities, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, u in zip(bp["boxes"], utilities):
        patch.set_facecolor(COLORS[u])
        patch.set_alpha(0.6)
    ax.axhline(0.0, color="red", linestyle="--", alpha=0.5, label="R²=0")
    ax.axhline(0.9, color="green", linestyle="--", alpha=0.5, label="R²=0.9")
    ax.set_ylabel("R² per Building (clipped at -2)", fontsize=11)
    ax.set_title("Per-Building R² Distribution by Utility", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_r2_boxplot.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / '02_r2_boxplot.png'}")


# ── Plot 3: Feature importance heatmap ───────────────────────────────

def plot_feature_importance_heatmap(metrics: dict):
    """Heatmap of top features across all utilities."""
    utilities = list(UTILITY_DIRS.keys())

    # Collect all features
    all_features = set()
    for u in utilities:
        all_features |= set(metrics[u].get("top_features", {}).keys())

    # Build importance matrix, keep only features that appear in >=2 utilities
    feat_counts = {}
    for f in all_features:
        feat_counts[f] = sum(1 for u in utilities if f in metrics[u].get("top_features", {}))
    features = sorted([f for f, c in feat_counts.items() if c >= 1],
                      key=lambda f: -max(metrics[u].get("top_features", {}).get(f, 0) for u in utilities))
    features = features[:15]  # top 15

    matrix = np.zeros((len(features), len(utilities)))
    for j, u in enumerate(utilities):
        top = metrics[u].get("top_features", {})
        for i, f in enumerate(features):
            matrix[i, j] = top.get(f, 0)

    fig, ax = plt.subplots(figsize=(10, max(5, len(features) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(utilities)))
    ax.set_xticklabels(utilities, fontsize=10)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)

    # Annotate cells
    for i in range(len(features)):
        for j in range(len(utilities)):
            val = matrix[i, j]
            if val > 0.001:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if val > 0.2 else "black")

    ax.set_title("Feature Importance Across Utilities", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Importance", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_feature_importance_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / '03_feature_importance_heatmap.png'}")


# ── Plot 4: Building residual heatmap ────────────────────────────────

def plot_building_residual_heatmap(per_building: dict, predictions: dict):
    """Heatmap of mean |residual| for buildings across utilities.

    Only shows buildings that appear in >=2 utilities.
    """
    utilities = list(UTILITY_DIRS.keys())

    # Collect per-building mean_residual for each utility
    bld_data = {}
    for u in utilities:
        bld = per_building[u]
        for _, row in bld.iterrows():
            b = str(row["building"])
            bld_data.setdefault(b, {})[u] = abs(row["mean_residual"])

    # Filter to buildings in >=2 utilities
    multi_util = {b: d for b, d in bld_data.items() if len(d) >= 2}
    if not multi_util:
        print("  No buildings with >=2 utilities — skipping heatmap")
        return

    # Sort by total residual across utilities
    buildings = sorted(multi_util.keys(),
                       key=lambda b: -sum(multi_util[b].values()))
    buildings = buildings[:40]  # top 40

    matrix = np.full((len(buildings), len(utilities)), np.nan)
    for i, b in enumerate(buildings):
        for j, u in enumerate(utilities):
            if u in multi_util[b]:
                matrix[i, j] = multi_util[b][u]

    fig, ax = plt.subplots(figsize=(10, max(6, len(buildings) * 0.35)))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap="Reds", interpolation="nearest")

    ax.set_xticks(range(len(utilities)))
    ax.set_xticklabels(utilities, fontsize=10)
    ax.set_yticks(range(len(buildings)))
    ax.set_yticklabels(buildings, fontsize=7)
    ax.set_title("Building |Mean Residual| Across Utilities (Top 40)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Utility")
    ax.set_ylabel("Building Number")
    fig.colorbar(im, ax=ax, label="|Mean Residual|", shrink=0.8)

    # Mark missing data with X
    for i in range(len(buildings)):
        for j in range(len(utilities)):
            if np.isnan(matrix[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_building_residual_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / '04_building_residual_heatmap.png'}")


# ── Plot 5: Residual vs gross area scatter ───────────────────────────

def plot_residual_vs_area(per_building: dict, predictions: dict):
    """Scatter: per-building mean |residual| vs grossArea, by utility."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for u in UTILITY_DIRS:
        bld = per_building[u].copy()
        pred = predictions[u]

        # Get grossarea per building from predictions
        area_map = pred.groupby("buildingnumber")["grossarea"].first()
        bld["grossarea"] = bld["building"].map(area_map)
        bld = bld.dropna(subset=["grossarea"])
        bld["abs_mean_residual"] = bld["mean_residual"].abs()
        # Filter out zeros for log scale
        bld = bld[bld["abs_mean_residual"] > 0]
        bld = bld[bld["grossarea"] > 0]

        ax.scatter(bld["grossarea"], bld["abs_mean_residual"],
                   alpha=0.5, s=25, label=u, color=COLORS[u], edgecolors="none")

    ax.set_xlabel("Gross Area (sqft)", fontsize=11)
    ax.set_ylabel("|Mean Residual| (energy/sqft)", fontsize=11)
    ax.set_title("Building Size vs Model Error by Utility", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if ax.get_xlim()[0] > 0:
        ax.set_xscale("log")
    if ax.get_ylim()[0] > 0:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_residual_vs_area.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / '05_residual_vs_area.png'}")


# ── Plot 6: Composite ranking ────────────────────────────────────────

def compute_composite_ranking(per_building: dict, predictions: dict) -> pd.DataFrame:
    """Composite investment score across all utilities.

    Score = sum of z-scored |mean_residual| across utilities (higher = worse performer = invest).
    Buildings with more utility types contributing get a higher total signal.
    """
    utilities = list(UTILITY_DIRS.keys())
    all_scores = []

    for u in utilities:
        bld = per_building[u].copy()
        pred = predictions[u]

        # Get building info
        bld_info = pred.groupby("buildingnumber").agg(
            grossarea=("grossarea", "first"),
        ).reset_index()
        bld["building"] = bld["building"].astype(str)
        bld_info["buildingnumber"] = bld_info["buildingnumber"].astype(str)
        bld = bld.merge(bld_info, left_on="building", right_on="buildingnumber", how="left")

        # z-score the |mean_residual| within utility
        bld["abs_residual"] = bld["mean_residual"].abs()
        mu = bld["abs_residual"].mean()
        sigma = bld["abs_residual"].std()
        bld["z_residual"] = (bld["abs_residual"] - mu) / sigma if sigma > 0 else 0

        all_scores.append(bld[["building", "grossarea", "z_residual", "r2"]].assign(utility=u))

    combined = pd.concat(all_scores, ignore_index=True)

    # Aggregate per building
    ranking = combined.groupby("building").agg(
        composite_z=("z_residual", "sum"),
        mean_z=("z_residual", "mean"),
        n_utilities=("utility", "count"),
        min_r2=("r2", "min"),
        mean_r2=("r2", "mean"),
        grossarea=("grossarea", "first"),
        utilities=("utility", lambda x: ", ".join(sorted(x))),
    ).reset_index()

    ranking = ranking.sort_values("composite_z", ascending=False).reset_index(drop=True)
    return ranking


def plot_composite_ranking(ranking: pd.DataFrame, top_n: int = 20):
    """Horizontal bar chart of top N buildings by composite score."""
    top = ranking.head(top_n).iloc[::-1]  # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top)))
    ax.barh(range(len(top)), top["composite_z"], color=colors, edgecolor="black", alpha=0.85)
    labels = [f"Bldg {b} ({n}u)" for b, n in zip(top["building"], top["n_utilities"])]
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Composite Z-Score (higher = more anomalous)", fontsize=11)
    ax.set_title(f"Top {top_n} Buildings for Energy Investment", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_composite_ranking.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / '06_composite_ranking.png'}")


# ── Summary report ───────────────────────────────────────────────────

def write_summary_report(
    overall: pd.DataFrame,
    coverage: pd.DataFrame,
    worst: dict,
    ranking: pd.DataFrame,
):
    """Write combined text + CSV reports."""
    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append("CROSS-UTILITY COMPARISON REPORT — XGBoost Models")
    lines.append(sep)

    # Overall metrics
    lines.append("\n--- Overall Model Metrics ---")
    lines.append(overall.to_string(index=False))

    # Coverage summary
    lines.append(f"\n--- Building Coverage ---")
    lines.append(f"  Total unique buildings: {len(coverage)}")
    for n in range(5, 0, -1):
        count = (coverage["n_utilities"] == n).sum()
        if count > 0:
            lines.append(f"  Buildings with {n} utilities: {count}")

    # Worst buildings per utility
    lines.append(f"\n--- Top 10 Worst Buildings Per Utility ---")
    for u, df in worst.items():
        lines.append(f"\n  {u}:")
        for _, row in df.iterrows():
            lines.append(
                f"    Bldg {str(row['building']):>6s}  R²={row['r2']:.4f}  "
                f"RMSE={row['rmse']:.6e}  mean_resid={row['mean_residual']:.6e}"
            )

    # Composite ranking (top 20)
    lines.append(f"\n--- Composite Investment Ranking (Top 20) ---")
    lines.append(f"  Score = sum of z-scored |mean_residual| across utilities\n")
    for i, (_, row) in enumerate(ranking.head(20).iterrows(), 1):
        lines.append(
            f"  {i:2d}. Bldg {row['building']:>6s}  "
            f"z={row['composite_z']:.3f}  "
            f"utilities={row['n_utilities']}  "
            f"mean_R²={row['mean_r2']:.4f}  "
            f"area={row['grossarea']:.0f} sqft  "
            f"[{row['utilities']}]"
        )

    lines.append(f"\n{sep}")
    report = "\n".join(lines)

    # Write text report
    report_path = OUT_DIR / "comparison_report.txt"
    report_path.write_text(report)
    print(f"  Saved: {report_path}")
    print()
    print(report)

    # Write ranking CSV
    ranking_path = OUT_DIR / "composite_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    print(f"\n  Saved: {ranking_path}")

    # Write overall CSV
    overall_path = OUT_DIR / "overall_metrics.csv"
    overall.to_csv(overall_path, index=False)
    print(f"  Saved: {overall_path}")

    # Write coverage CSV
    coverage_path = OUT_DIR / "building_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    print(f"  Saved: {coverage_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Loading data from 5 utility outputs...")
    metrics, predictions, per_building = load_all()

    print("\n--- Generating tables ---")
    overall = table_overall_metrics(metrics, predictions)
    coverage = table_building_coverage(predictions)
    worst = table_worst_buildings(per_building)

    print("\n--- Generating plots ---")
    plot_r2_bar(metrics)
    plot_r2_boxplot(per_building)
    plot_feature_importance_heatmap(metrics)
    plot_building_residual_heatmap(per_building, predictions)
    plot_residual_vs_area(per_building, predictions)

    print("\n--- Computing composite ranking ---")
    ranking = compute_composite_ranking(per_building, predictions)
    plot_composite_ranking(ranking)

    print("\n--- Writing reports ---")
    write_summary_report(overall, coverage, worst, ranking)

    print("\nDone. All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
