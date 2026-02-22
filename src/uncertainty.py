#!/usr/bin/env python3
"""Uncertainty quantification and confidence tiers for building rankings.

Uses multiple sources of uncertainty:
    1. QRF prediction intervals (q10-q90 width)
    2. NGBoost predicted std
    3. Residual-based bootstrap confidence intervals
    4. Cross-model agreement (if multiple model outputs exist)

Assigns each building a confidence tier: High / Medium / Low.

Usage:
    python -m src.uncertainty                          # electricity
    python -m src.uncertainty --utility COOLING
    python -m src.uncertainty --utilities ELECTRICITY COOLING GAS HEAT STEAM
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────

# Best XGBoost runs per utility
XGB_DIRS = {
    "ELECTRICITY": "output/electricity_xgboost_20260222_021830",
    "COOLING": "output/cooling_xgboost_20260222_021836",
    "GAS": "output/gas_xgboost_20260222_021833",
    "HEAT": "output/heat_xgboost_20260222_021837",
    "STEAM": "output/steam_xgboost_20260222_021840",
}

# Probabilistic model runs (electricity only for now)
QRF_DIR = "output/electricity_qrf_20260222_005552"
NGB_DIR = "output/electricity_ngboost_20260222_005552"

BUILDING_METADATA = "data/building_metadata.csv"
OUT_DIR = Path("doc/uncertainty")

# Confidence tier thresholds (percentiles of uncertainty score)
TIER_THRESHOLDS = {"High": 0.33, "Medium": 0.67}  # bottom third = high confidence


# ── Uncertainty from residuals ───────────────────────────────────────


def residual_uncertainty(preds: pd.DataFrame) -> pd.DataFrame:
    """Compute per-building uncertainty from residual distribution.

    Uses bootstrap-like statistics:
    - std of residuals (spread)
    - IQR of residuals (robust spread)
    - coefficient of variation of residuals
    - sample size (more samples = lower uncertainty)
    """
    gb = preds.groupby("buildingnumber")

    df = pd.DataFrame({
        "n_samples": gb["residual"].count(),
        "residual_std": gb["residual"].std(),
        "residual_iqr": gb["residual"].apply(lambda x: x.quantile(0.75) - x.quantile(0.25)),
        "residual_mean": gb["residual"].mean(),
        "energy_mean": gb["energy_per_sqft"].mean(),
    })

    # Coefficient of variation of residuals (normalized uncertainty)
    df["residual_cv"] = df["residual_std"] / df["energy_mean"].replace(0, np.nan).abs()

    # Sample-size penalty: fewer samples = higher uncertainty
    # Log-scaled so it diminishes for large sample sizes
    median_n = df["n_samples"].median()
    df["sample_factor"] = np.clip(median_n / df["n_samples"], 0.5, 3.0)

    # Combined residual uncertainty score
    # Normalize each component to 0-1 then average
    for col in ["residual_std", "residual_iqr", "residual_cv"]:
        col_norm = f"{col}_norm"
        vals = df[col].fillna(df[col].median())
        mn, mx = vals.min(), vals.max()
        df[col_norm] = (vals - mn) / (mx - mn + 1e-12)

    df["residual_uncertainty"] = (
        0.4 * df["residual_std_norm"]
        + 0.3 * df["residual_iqr_norm"]
        + 0.2 * df["residual_cv_norm"]
        + 0.1 * (df["sample_factor"] - 0.5) / 2.5  # normalize to ~0-1
    )

    return df


# ── Uncertainty from QRF prediction intervals ───────────────────────


def qrf_uncertainty(qrf_dir: str = QRF_DIR) -> pd.DataFrame | None:
    """Compute per-building uncertainty from QRF prediction intervals.

    Uses the width of the 80% prediction interval (q90 - q10).
    """
    pred_path = Path(qrf_dir) / "predictions.parquet"
    if not pred_path.exists():
        print(f"  [SKIP] QRF predictions not found: {pred_path}")
        return None

    preds = pd.read_parquet(pred_path)
    preds["buildingnumber"] = preds.get("buildingnumber", preds.get("simscode", "")).astype(str)

    # Check for quantile columns
    if "prediction_interval_width" not in preds.columns:
        print("  [SKIP] QRF predictions missing interval columns")
        return None

    gb = preds.groupby("buildingnumber")
    df = pd.DataFrame({
        "qrf_mean_interval": gb["prediction_interval_width"].mean(),
        "qrf_median_interval": gb["prediction_interval_width"].median(),
        "qrf_max_interval": gb["prediction_interval_width"].max(),
        "qrf_energy_mean": gb["energy_per_sqft"].mean(),
    })

    # Normalized interval width (relative to mean consumption)
    df["qrf_relative_width"] = (
        df["qrf_mean_interval"] / df["qrf_energy_mean"].replace(0, np.nan).abs()
    )

    # Normalize to 0-1
    vals = df["qrf_relative_width"].fillna(df["qrf_relative_width"].median())
    mn, mx = vals.min(), vals.max()
    df["qrf_uncertainty"] = (vals - mn) / (mx - mn + 1e-12)

    return df


# ── Uncertainty from NGBoost predicted std ───────────────────────────


def ngboost_uncertainty(ngb_dir: str = NGB_DIR) -> pd.DataFrame | None:
    """Compute per-building uncertainty from NGBoost predicted std."""
    pred_path = Path(ngb_dir) / "predictions.parquet"
    if not pred_path.exists():
        print(f"  [SKIP] NGBoost predictions not found: {pred_path}")
        return None

    preds = pd.read_parquet(pred_path)
    preds["buildingnumber"] = preds.get("buildingnumber", preds.get("simscode", "")).astype(str)

    if "predicted_std" not in preds.columns:
        print("  [SKIP] NGBoost predictions missing predicted_std")
        return None

    gb = preds.groupby("buildingnumber")
    df = pd.DataFrame({
        "ngb_mean_std": gb["predicted_std"].mean(),
        "ngb_median_std": gb["predicted_std"].median(),
        "ngb_energy_mean": gb["energy_per_sqft"].mean(),
    })

    # Normalized predicted std
    df["ngb_relative_std"] = (
        df["ngb_mean_std"] / df["ngb_energy_mean"].replace(0, np.nan).abs()
    )

    vals = df["ngb_relative_std"].fillna(df["ngb_relative_std"].median())
    mn, mx = vals.min(), vals.max()
    df["ngb_uncertainty"] = (vals - mn) / (mx - mn + 1e-12)

    return df


# ── Cross-model agreement ───────────────────────────────────────────


def cross_model_agreement(utility: str = "ELECTRICITY") -> pd.DataFrame | None:
    """Compare rankings across multiple model types for the same utility.

    High agreement = high confidence. Low agreement = low confidence.
    """
    model_dirs = {}
    output_dir = Path("output")
    utility_lower = utility.lower()

    # Find all model runs for this utility
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and d.name.startswith(utility_lower):
            metrics_path = d / "postprocess_per_building_metrics.csv"
            if metrics_path.exists():
                # Extract model type from dir name
                parts = d.name.replace(f"{utility_lower}_", "").split("_")
                model_type = parts[0]
                if model_type not in model_dirs:  # keep latest
                    model_dirs[model_type] = d

    if len(model_dirs) < 2:
        print(f"  [SKIP] Only {len(model_dirs)} model(s) found for {utility}")
        return None

    print(f"  Cross-model comparison: {list(model_dirs.keys())}")

    # Load per-building metrics from each model
    rankings = {}
    for model_type, d in model_dirs.items():
        metrics = pd.read_csv(d / "postprocess_per_building_metrics.csv")
        metrics["building"] = metrics["building"].astype(str)
        metrics = metrics.set_index("building")
        # Rank by |mean_residual| descending
        metrics["rank"] = metrics["mean_residual"].abs().rank(ascending=False)
        rankings[model_type] = metrics["rank"]

    # Combine ranks
    rank_df = pd.DataFrame(rankings)

    # Rank disagreement = std of ranks across models (higher = less agreement)
    rank_df["rank_std"] = rank_df.std(axis=1)
    rank_df["rank_range"] = rank_df.drop(columns=["rank_std"]).max(axis=1) - rank_df.drop(columns=["rank_std", "rank_range"], errors="ignore").min(axis=1)
    rank_df["n_models"] = rank_df.drop(columns=["rank_std", "rank_range"], errors="ignore").notna().sum(axis=1)

    # Normalize to 0-1 (higher = more disagreement = less confidence)
    vals = rank_df["rank_std"].fillna(rank_df["rank_std"].median())
    mn, mx = vals.min(), vals.max()
    rank_df["model_disagreement"] = (vals - mn) / (mx - mn + 1e-12)

    return rank_df[["rank_std", "rank_range", "n_models", "model_disagreement"]]


# ── Combined uncertainty + tiers ─────────────────────────────────────


def compute_confidence_tiers(
    utility: str = "ELECTRICITY",
    xgb_dir: str | None = None,
    qrf_dir: str = QRF_DIR,
    ngb_dir: str = NGB_DIR,
) -> pd.DataFrame:
    """Combine all uncertainty sources into a single score and assign tiers."""
    xgb_dir = xgb_dir or XGB_DIRS.get(utility)
    if not xgb_dir:
        raise ValueError(f"No XGB directory for utility {utility}")

    print(f"\nUncertainty Analysis — {utility}")
    print("=" * 50)

    # 1. Residual-based uncertainty (always available)
    print("  Computing residual uncertainty...")
    preds = pd.read_parquet(Path(xgb_dir) / "predictions.parquet")
    preds["buildingnumber"] = preds["buildingnumber"].astype(str)
    res_unc = residual_uncertainty(preds)

    # Start with residual uncertainty
    combined = res_unc[["n_samples", "residual_std", "residual_cv", "residual_uncertainty"]].copy()
    weights = {"residual": 0.5}
    available_sources = ["residual"]

    # 2. QRF intervals (if available and same utility)
    if utility == "ELECTRICITY":
        print("  Computing QRF uncertainty...")
        qrf_unc = qrf_uncertainty(qrf_dir)
        if qrf_unc is not None:
            combined = combined.join(qrf_unc[["qrf_mean_interval", "qrf_uncertainty"]], how="left")
            weights["qrf"] = 0.25
            weights["residual"] = 0.35  # rebalance
            available_sources.append("qrf")

        # 3. NGBoost predicted std
        print("  Computing NGBoost uncertainty...")
        ngb_unc = ngboost_uncertainty(ngb_dir)
        if ngb_unc is not None:
            combined = combined.join(ngb_unc[["ngb_mean_std", "ngb_uncertainty"]], how="left")
            weights["ngb"] = 0.15
            weights["residual"] = 0.30
            available_sources.append("ngb")

    # 4. Cross-model agreement
    print("  Computing cross-model agreement...")
    model_agree = cross_model_agreement(utility)
    if model_agree is not None:
        combined = combined.join(model_agree[["rank_std", "n_models", "model_disagreement"]], how="left")
        remaining = 1.0 - sum(weights.values())
        weights["model"] = max(remaining, 0.10)
        # Normalize weights
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}
        available_sources.append("model")

    print(f"  Uncertainty sources: {available_sources}")
    print(f"  Weights: {weights}")

    # Combine into final uncertainty score
    combined["combined_uncertainty"] = 0.0
    if "residual" in weights:
        combined["combined_uncertainty"] += weights["residual"] * combined["residual_uncertainty"].fillna(0.5)
    if "qrf" in weights and "qrf_uncertainty" in combined.columns:
        combined["combined_uncertainty"] += weights["qrf"] * combined["qrf_uncertainty"].fillna(0.5)
    if "ngb" in weights and "ngb_uncertainty" in combined.columns:
        combined["combined_uncertainty"] += weights["ngb"] * combined["ngb_uncertainty"].fillna(0.5)
    if "model" in weights and "model_disagreement" in combined.columns:
        combined["combined_uncertainty"] += weights["model"] * combined["model_disagreement"].fillna(0.5)

    # Assign tiers based on percentiles
    q33 = combined["combined_uncertainty"].quantile(TIER_THRESHOLDS["High"])
    q67 = combined["combined_uncertainty"].quantile(TIER_THRESHOLDS["Medium"])

    combined["confidence_tier"] = "Medium"
    combined.loc[combined["combined_uncertainty"] <= q33, "confidence_tier"] = "High"
    combined.loc[combined["combined_uncertainty"] > q67, "confidence_tier"] = "Low"

    # Add building metadata
    bldg_meta = pd.read_csv(BUILDING_METADATA)
    bldg_meta["buildingnumber"] = bldg_meta["buildingnumber"].astype(str)
    bldg_meta = bldg_meta[["buildingnumber", "buildingname", "grossarea"]].drop_duplicates("buildingnumber").set_index("buildingnumber")
    combined = combined.join(bldg_meta, how="left")

    combined["utility"] = utility
    combined = combined.sort_values("combined_uncertainty")

    return combined


# ── Visualization ────────────────────────────────────────────────────


def plot_confidence_distribution(df: pd.DataFrame, utility: str, out_dir: Path):
    """Histogram of uncertainty scores colored by confidence tier."""
    fig, ax = plt.subplots(figsize=(10, 6))

    tier_colors = {"High": "#4CAF50", "Medium": "#FF9800", "Low": "#F44336"}
    for tier, color in tier_colors.items():
        subset = df[df["confidence_tier"] == tier]
        ax.hist(
            subset["combined_uncertainty"], bins=20, alpha=0.7,
            label=f"{tier} ({len(subset)} buildings)", color=color,
        )

    ax.set_xlabel("Combined Uncertainty Score")
    ax.set_ylabel("Number of Buildings")
    ax.set_title(f"Confidence Tier Distribution — {utility}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"confidence_distribution_{utility.lower()}.png", dpi=150)
    plt.close()
    print(f"  Saved confidence distribution plot")


def plot_uncertainty_sources(df: pd.DataFrame, utility: str, out_dir: Path):
    """Scatter plot comparing uncertainty sources."""
    unc_cols = [c for c in df.columns if c.endswith("_uncertainty") and c != "combined_uncertainty"]
    if len(unc_cols) < 2:
        return

    fig, axes = plt.subplots(1, len(unc_cols) - 1, figsize=(6 * (len(unc_cols) - 1), 5))
    if len(unc_cols) - 1 == 1:
        axes = [axes]

    tier_colors = {"High": "#4CAF50", "Medium": "#FF9800", "Low": "#F44336"}
    base_col = unc_cols[0]

    for ax, compare_col in zip(axes, unc_cols[1:]):
        for tier, color in tier_colors.items():
            mask = df["confidence_tier"] == tier
            ax.scatter(
                df.loc[mask, base_col], df.loc[mask, compare_col],
                c=color, alpha=0.6, label=tier, s=20,
            )
        ax.set_xlabel(base_col.replace("_", " ").title())
        ax.set_ylabel(compare_col.replace("_", " ").title())
        ax.legend(fontsize=8)

    fig.suptitle(f"Uncertainty Source Comparison — {utility}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / f"uncertainty_sources_{utility.lower()}.png", dpi=150)
    plt.close()
    print(f"  Saved uncertainty sources plot")


def plot_tier_bar(df: pd.DataFrame, utility: str, out_dir: Path, top_n: int = 30):
    """Horizontal bar chart of top buildings colored by confidence tier."""
    # Sort by some investment priority (use combined_uncertainty for now)
    top = df.head(top_n).copy()
    top["label"] = top.apply(
        lambda r: f"{r.name} – {r.get('buildingname', '')}"[:40] if pd.notna(r.get("buildingname")) else str(r.name),
        axis=1,
    )

    tier_colors = {"High": "#4CAF50", "Medium": "#FF9800", "Low": "#F44336"}
    colors = [tier_colors[t] for t in top["confidence_tier"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top)), top["combined_uncertainty"].values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Combined Uncertainty Score (lower = more confident)")
    ax.set_title(f"Building Confidence Tiers — {utility}")

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=t) for t, c in tier_colors.items()]
    ax.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_dir / f"tier_bar_{utility.lower()}.png", dpi=150)
    plt.close()
    print(f"  Saved tier bar chart")


# ── I/O ──────────────────────────────────────────────────────────────


def save_results(df: pd.DataFrame, utility: str, out_dir: Path) -> str:
    """Save uncertainty results and generate summary report."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / f"confidence_tiers_{utility.lower()}.csv"
    df.to_csv(csv_path)
    print(f"  Saved {csv_path}")

    # Summary report
    tier_counts = df["confidence_tier"].value_counts()
    lines = [
        "=" * 60,
        f"UNCERTAINTY & CONFIDENCE REPORT — {utility}",
        "=" * 60,
        "",
        f"Total buildings analyzed: {len(df)}",
        f"Confidence tiers:",
        f"  High:   {tier_counts.get('High', 0)} buildings",
        f"  Medium: {tier_counts.get('Medium', 0)} buildings",
        f"  Low:    {tier_counts.get('Low', 0)} buildings",
        "",
        "Uncertainty sources used:",
    ]

    unc_cols = [c for c in df.columns if c.endswith("_uncertainty") or c == "model_disagreement"]
    for col in unc_cols:
        lines.append(f"  - {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    lines.extend(["", "TOP 10 HIGHEST CONFIDENCE BUILDINGS:", "-" * 40])
    for i, (bldg_id, row) in enumerate(df.head(10).iterrows(), 1):
        name = row.get("buildingname", "N/A")
        lines.append(
            f"  {i:2d}. Building {bldg_id} ({name}) — "
            f"Uncertainty: {row['combined_uncertainty']:.4f} [{row['confidence_tier']}]"
        )

    lines.extend(["", "TOP 10 LOWEST CONFIDENCE BUILDINGS:", "-" * 40])
    for i, (bldg_id, row) in enumerate(df.tail(10).iloc[::-1].iterrows(), 1):
        name = row.get("buildingname", "N/A")
        lines.append(
            f"  {i:2d}. Building {bldg_id} ({name}) — "
            f"Uncertainty: {row['combined_uncertainty']:.4f} [{row['confidence_tier']}]"
        )

    report = "\n".join(lines)
    report_path = out_dir / f"uncertainty_report_{utility.lower()}.txt"
    report_path.write_text(report)
    print(f"  Saved {report_path}")

    return report


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Uncertainty and confidence tiers")
    parser.add_argument(
        "--utilities", nargs="+", default=["ELECTRICITY"],
        help="Utilities to analyze (default: ELECTRICITY)",
    )
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    for utility in args.utilities:
        if utility not in XGB_DIRS:
            print(f"[SKIP] Unknown utility: {utility}")
            continue

        df = compute_confidence_tiers(utility)

        report = save_results(df, utility, out_dir)
        plot_confidence_distribution(df, utility, out_dir)
        plot_uncertainty_sources(df, utility, out_dir)
        plot_tier_bar(df, utility, out_dir)

        print(f"\n{report}")

    print("\nDone!")


if __name__ == "__main__":
    main()
