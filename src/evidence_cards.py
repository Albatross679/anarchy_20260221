#!/usr/bin/env python3
"""Per-building evidence cards for energy investment prioritization.

Generates one-page PNG summaries per building containing:
    - Rank, composite score, confidence tier
    - 5-signal breakdown (radar chart)
    - Time series: actual vs predicted energy consumption
    - Residual distribution
    - Plain-language "Why this building?" explanation
    - Key metadata (area, age, utilities)

Usage:
    python -m src.evidence_cards                        # top 10 electricity
    python -m src.evidence_cards --top-n 20             # top 20
    python -m src.evidence_cards --building 42 123      # specific buildings
    python -m src.evidence_cards --utility COOLING
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────

UTILITY_DIRS = {
    "ELECTRICITY": "output/electricity_xgboost_20260222_021830",
    "COOLING": "output/cooling_xgboost_20260222_021836",
    "GAS": "output/gas_xgboost_20260222_021833",
    "HEAT": "output/heat_xgboost_20260222_021837",
    "STEAM": "output/steam_xgboost_20260222_021840",
}

BUILDING_METADATA = "data/building_metadata.csv"
OUT_DIR = Path("doc/evidence_cards")

# Colors
TIER_COLORS = {"High": "#4CAF50", "Medium": "#FF9800", "Low": "#F44336"}
SIGNAL_NAMES = [
    "Residual", "Weather\nSensitivity", "Baseline\nLoad",
    "Variability", "Peer\nComparison",
]
SIGNAL_COLS = [
    "residual_z", "weather_sensitivity_z", "baseline_load_z",
    "variability_z", "peer_z",
]


# ── Data loading ────────────────────────────────────────────────────


def load_building_data(
    building_id: str,
    utility: str,
    utility_dirs: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load predictions and metadata for a specific building."""
    utility_dirs = utility_dirs or UTILITY_DIRS
    dir_path = Path(utility_dirs[utility])

    preds = pd.read_parquet(dir_path / "predictions.parquet")
    preds["buildingnumber"] = preds["buildingnumber"].astype(str)

    bldg_preds = preds[preds["buildingnumber"] == str(building_id)].copy()
    if bldg_preds.empty:
        raise ValueError(f"Building {building_id} not found in {utility} predictions")

    # Sort by time
    bldg_preds = bldg_preds.sort_values("readingtime")

    # Load metrics
    with open(dir_path / "metrics.json") as f:
        metrics = json.load(f)

    return bldg_preds, metrics


def load_scoring_data(scoring_dir: str = "doc/scoring") -> dict[str, pd.DataFrame]:
    """Load per-utility scoring results if available."""
    scoring_path = Path(scoring_dir)
    scores = {}
    for utility in UTILITY_DIRS:
        path = scoring_path / f"scores_{utility.lower()}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            df.index = df.index.astype(str)
            scores[utility] = df
    return scores


def load_uncertainty_data(uncertainty_dir: str = "doc/uncertainty") -> dict[str, pd.DataFrame]:
    """Load confidence tier data if available."""
    unc_path = Path(uncertainty_dir)
    tiers = {}
    for utility in UTILITY_DIRS:
        path = unc_path / f"confidence_tiers_{utility.lower()}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            df.index = df.index.astype(str)
            tiers[utility] = df
    return tiers


# ── Evidence card generation ────────────────────────────────────────


def generate_explanation(
    building_id: str,
    utility: str,
    scores: pd.DataFrame | None,
    bldg_preds: pd.DataFrame,
    bldg_meta: dict,
) -> str:
    """Generate plain-language 'Why this building?' explanation."""
    lines = []

    # Building context
    name = bldg_meta.get("buildingname", "Unknown")
    area = bldg_meta.get("grossarea", "N/A")
    age = bldg_meta.get("building_age", "N/A")
    lines.append(f"Building {building_id} ({name})")
    lines.append(f"Area: {area:,} sqft | Age: {age} yrs" if isinstance(area, (int, float)) else f"Area: {area} | Age: {age}")

    # Residual analysis
    mean_res = bldg_preds["residual"].mean()
    std_res = bldg_preds["residual"].std()
    if mean_res > 0:
        lines.append(f"Consumes {abs(mean_res):.6f} kWh/sqft MORE than predicted on average.")
    else:
        lines.append(f"Consumes {abs(mean_res):.6f} kWh/sqft LESS than predicted on average.")

    # Signal breakdown
    if scores is not None and str(building_id) in scores.index:
        row = scores.loc[str(building_id)]
        drivers = []
        for col, label in zip(SIGNAL_COLS, SIGNAL_NAMES):
            if col in row.index:
                val = row[col]
                if abs(val) > 1.0:
                    direction = "high" if val > 0 else "low"
                    drivers.append(f"{direction} {label.replace(chr(10), ' ')} (z={val:.2f})")

        if drivers:
            lines.append("Key drivers: " + "; ".join(drivers[:3]))
        else:
            lines.append("No extreme signal drivers detected.")

    return "\n".join(lines)


def create_evidence_card(
    building_id: str,
    utility: str,
    out_dir: Path,
    scores: dict[str, pd.DataFrame] | None = None,
    tiers: dict[str, pd.DataFrame] | None = None,
):
    """Create a one-page evidence card PNG for a building."""
    scores = scores or {}
    tiers = tiers or {}

    # Load data
    bldg_preds, model_metrics = load_building_data(building_id, utility)

    # Get building metadata
    bldg_meta_df = pd.read_csv(BUILDING_METADATA)
    bldg_meta_df["buildingnumber"] = bldg_meta_df["buildingnumber"].astype(str)
    bldg_row = bldg_meta_df[bldg_meta_df["buildingnumber"] == str(building_id)]
    bldg_meta = bldg_row.iloc[0].to_dict() if not bldg_row.empty else {}

    # Get scoring data
    util_scores = scores.get(utility)
    # Get tier data
    util_tiers = tiers.get(utility)

    # Extract key values
    rank = "N/A"
    composite_score = "N/A"
    confidence_tier = "Medium"

    if util_scores is not None and str(building_id) in util_scores.index:
        row = util_scores.loc[str(building_id)]
        rank = int(row.get("rank", 0))
        composite_score = f"{row.get('composite_score', 0):.3f}"

    if util_tiers is not None and str(building_id) in util_tiers.index:
        confidence_tier = util_tiers.loc[str(building_id)].get("confidence_tier", "Medium")

    # Building stats
    n_samples = len(bldg_preds)
    mean_residual = bldg_preds["residual"].mean()
    r2 = 1 - (bldg_preds["residual"].var() / bldg_preds["energy_per_sqft"].var()) if bldg_preds["energy_per_sqft"].var() > 0 else 0
    name = bldg_meta.get("buildingname", "Unknown")
    area = bldg_meta.get("grossarea", "N/A")

    # ── Create figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#FAFAFA")

    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.94, top=0.88, bottom=0.05)

    # ── Header ───────────────────────────────────────────────────────
    tier_color = TIER_COLORS.get(confidence_tier, "#999")
    fig.text(0.5, 0.96, f"Building {building_id} — {name}",
             ha="center", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.925,
             f"{utility} | Rank #{rank} | Score: {composite_score} | "
             f"Area: {area:,} sqft | R²: {r2:.3f}" if isinstance(area, (int, float))
             else f"{utility} | Rank #{rank} | Score: {composite_score} | R²: {r2:.3f}",
             ha="center", fontsize=11, color="#555")

    # Confidence badge
    fig.text(0.92, 0.96, f"  {confidence_tier}  ",
             ha="center", fontsize=12, fontweight="bold",
             color="white",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=tier_color, alpha=0.9))

    # ── Panel 1: Time series (actual vs predicted) ───────────────────
    ax1 = fig.add_subplot(gs[0, :])
    times = bldg_preds["readingtime"]
    # Subsample if too many points
    step = max(1, len(times) // 2000)
    ax1.plot(times.iloc[::step], bldg_preds["energy_per_sqft"].iloc[::step],
             alpha=0.6, linewidth=0.5, label="Actual", color="#2196F3")
    ax1.plot(times.iloc[::step], bldg_preds["predicted"].iloc[::step],
             alpha=0.6, linewidth=0.5, label="Predicted", color="#FF5722")
    ax1.set_ylabel("Energy/sqft")
    ax1.set_title("Actual vs Predicted Consumption", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.tick_params(axis="x", labelsize=8)

    # ── Panel 2: Residual distribution ───────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    residuals = bldg_preds["residual"].dropna()
    ax2.hist(residuals, bins=50, alpha=0.7, color="#7E57C2", edgecolor="white")
    ax2.axvline(0, color="red", linewidth=1, linestyle="--")
    ax2.axvline(mean_residual, color="orange", linewidth=2, label=f"Mean: {mean_residual:.6f}")
    ax2.set_xlabel("Residual (actual − predicted)")
    ax2.set_title("Residual Distribution", fontsize=11)
    ax2.legend(fontsize=8)

    # ── Panel 3: Radar chart of 5 signals ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1], polar=True)
    if util_scores is not None and str(building_id) in util_scores.index:
        row = util_scores.loc[str(building_id)]
        values = [row.get(c, 0) for c in SIGNAL_COLS]
        values_plot = values + values[:1]
        angles = np.linspace(0, 2 * np.pi, len(SIGNAL_NAMES), endpoint=False).tolist()
        angles += angles[:1]

        ax3.fill(angles, values_plot, alpha=0.25, color="#E91E63")
        ax3.plot(angles, values_plot, "o-", color="#E91E63", linewidth=2, markersize=5)
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(SIGNAL_NAMES, fontsize=8)
        ax3.set_title("Signal Breakdown", fontsize=11, pad=15)
    else:
        ax3.text(0.5, 0.5, "Scoring data\nnot available",
                 transform=ax3.transAxes, ha="center", fontsize=10, color="#999")
        ax3.set_title("Signal Breakdown", fontsize=11, pad=15)

    # ── Panel 4: Residual by hour of day ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    hourly = bldg_preds.groupby("hour_of_day")["residual"].agg(["mean", "std"])
    ax4.bar(hourly.index, hourly["mean"], alpha=0.7, color="#26A69A", edgecolor="white")
    ax4.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax4.set_xlabel("Hour of Day")
    ax4.set_ylabel("Mean Residual")
    ax4.set_title("Residual by Hour", fontsize=11)
    ax4.set_xticks(range(0, 24, 3))

    # ── Panel 5: Explanation text ────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0:2])
    ax5.axis("off")

    explanation = generate_explanation(building_id, utility, util_scores, bldg_preds, bldg_meta)
    wrapped = "\n".join(textwrap.fill(line, width=80) for line in explanation.split("\n"))

    ax5.text(0.02, 0.95, "WHY THIS BUILDING?", fontsize=12, fontweight="bold",
             transform=ax5.transAxes, va="top", color="#333")
    ax5.text(0.02, 0.78, wrapped, fontsize=10, transform=ax5.transAxes,
             va="top", color="#444", family="monospace",
             linespacing=1.5)

    # ── Panel 6: Key stats table ─────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")

    stats_data = [
        ["Metric", "Value"],
        ["Samples", f"{n_samples:,}"],
        ["R²", f"{r2:.4f}"],
        ["Mean Residual", f"{mean_residual:.6f}"],
        ["Std Residual", f"{bldg_preds['residual'].std():.6f}"],
        ["Confidence", confidence_tier],
        ["Rank", f"#{rank}"],
        ["Composite", composite_score],
    ]

    table = ax6.table(
        cellText=stats_data[1:],
        colLabels=stats_data[0],
        cellLoc="center",
        loc="center",
        colWidths=[0.5, 0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(2):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style confidence cell
    for i in range(1, len(stats_data)):
        if stats_data[i][0] == "Confidence":
            table[i, 1].set_facecolor(tier_color)
            table[i, 1].set_text_props(color="white", fontweight="bold")

    ax6.set_title("Key Statistics", fontsize=11, pad=10)

    # ── Save ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"card_{building_id}_{utility.lower()}.png"
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    return out_path


# ── Batch generation ────────────────────────────────────────────────


def generate_cards(
    utility: str = "ELECTRICITY",
    top_n: int = 10,
    building_ids: list[str] | None = None,
    out_dir: Path = OUT_DIR,
):
    """Generate evidence cards for top buildings or specific building IDs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvidence Card Generation — {utility}")
    print("=" * 50)

    # Load supporting data
    scores = load_scoring_data()
    tiers = load_uncertainty_data()

    # Determine buildings
    if building_ids is None:
        util_scores = scores.get(utility)
        if util_scores is not None:
            building_ids = util_scores.head(top_n).index.tolist()
        else:
            # Fall back to predictions
            dir_path = Path(UTILITY_DIRS[utility])
            preds = pd.read_parquet(dir_path / "predictions.parquet")
            preds["buildingnumber"] = preds["buildingnumber"].astype(str)
            residuals = preds.groupby("buildingnumber")["residual"].mean().abs().sort_values(ascending=False)
            building_ids = residuals.head(top_n).index.tolist()

    building_ids = [str(b) for b in building_ids]
    print(f"  Generating {len(building_ids)} evidence cards...")

    generated = []
    for bldg_id in building_ids:
        try:
            path = create_evidence_card(bldg_id, utility, out_dir, scores, tiers)
            generated.append(path)
        except Exception as e:
            print(f"  [ERROR] Building {bldg_id}: {e}")

    # Summary
    print(f"\n  Generated {len(generated)}/{len(building_ids)} evidence cards")
    print(f"  Output directory: {out_dir}")

    return generated


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate per-building evidence cards")
    parser.add_argument("--utility", default="ELECTRICITY", choices=list(UTILITY_DIRS.keys()))
    parser.add_argument("--top-n", type=int, default=10, help="Number of top buildings")
    parser.add_argument("--building", type=str, nargs="*", help="Specific building IDs")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    generate_cards(
        utility=args.utility,
        top_n=args.top_n,
        building_ids=args.building,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
