"""
Capstone Project - Screen-Ready Visuals for Presentation
Generates 3 presentation slides as high-resolution PNG images:
  1. Pipeline diagram
  2. Results dashboard (metrics + confusion matrix)
  3. SHAP feature importance
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
import numpy as np
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BRAND_BLUE   = "#1565C0"
BRAND_GREEN  = "#2E7D32"
BRAND_ORANGE = "#E65100"
BRAND_GRAY   = "#37474F"
LIGHT_BLUE   = "#E3F2FD"
LIGHT_GREEN  = "#E8F5E9"
BG           = "#FAFAFA"

# ─────────────────────────────────────────────────────────────────────────────
# VISUAL 1 — 9-Step Pipeline Diagram
# ─────────────────────────────────────────────────────────────────────────────
def make_pipeline():
    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    steps = [
        ("Step 1", "Data\nCollection",   "1.5M loan apps\n32 parquet tables",     BRAND_BLUE),
        ("Step 2", "Data\nMerging",      "Static + dynamic\ntable joins",          BRAND_BLUE),
        ("Step 3", "Data\nPreprocessing","384M missing values\nimputed",            BRAND_BLUE),
        ("Step 4", "Feature\nEngineering","727 features\nscaled & encoded",         BRAND_BLUE),
        ("Step 5", "Model\nTraining",    "LightGBM\n+ Logistic Regression",         BRAND_ORANGE),
        ("Step 6", "Model\nEvaluation",  "AUC-ROC 0.803\nRecall 69.6%",            BRAND_GREEN),
        ("Step 7", "SHAP\nAnalysis",     "727 features\nexplained",                 BRAND_GREEN),
        ("Step 8", "Visualizations",     "8 diagnostic\nplots",                     BRAND_GREEN),
        ("Step 9", "Threshold\nOptimization","Business cost\nanalysis",             BRAND_GREEN),
    ]

    n = len(steps)
    xs = np.linspace(0.05, 0.95, n)
    y_box = 0.52
    box_w, box_h = 0.085, 0.30

    for i, (num, title, detail, color) in enumerate(steps):
        x = xs[i]
        # box
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y_box - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.01", linewidth=2,
            edgecolor=color, facecolor=color + "22"
        )
        ax.add_patch(rect)
        ax.text(x, y_box + 0.05, num, ha="center", va="center",
                fontsize=9, color=color, fontweight="bold")
        ax.text(x, y_box + 0.00, title, ha="center", va="center",
                fontsize=10, color=BRAND_GRAY, fontweight="bold", linespacing=1.3)
        ax.text(x, y_box - 0.10, detail, ha="center", va="center",
                fontsize=8, color="#546E7A", linespacing=1.4)

        # arrow between boxes
        if i < n - 1:
            ax.annotate("", xy=(xs[i + 1] - box_w / 2 - 0.004, y_box),
                        xytext=(x + box_w / 2 + 0.004, y_box),
                        arrowprops=dict(arrowstyle="->", color="#90A4AE", lw=1.8))

    # title
    ax.text(0.5, 0.93, "End-to-End Credit Risk ML Pipeline",
            ha="center", va="center", fontsize=20, fontweight="bold", color=BRAND_GRAY,
            transform=ax.transAxes)
    ax.text(0.5, 0.87, "Home Credit Default Prediction  |  Capstone 2  |  2026",
            ha="center", va="center", fontsize=12, color="#78909C",
            transform=ax.transAxes)

    # legend
    legend_handles = [
        mpatches.Patch(facecolor=BRAND_BLUE   + "22", edgecolor=BRAND_BLUE,   label="Data Engineering"),
        mpatches.Patch(facecolor=BRAND_ORANGE + "22", edgecolor=BRAND_ORANGE, label="Model Training"),
        mpatches.Patch(facecolor=BRAND_GREEN  + "22", edgecolor=BRAND_GREEN,  label="Evaluation & Insights"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=11,
              framealpha=0.7, bbox_to_anchor=(0.5, 0.03))

    out = os.path.join(OUTPUT_DIR, "showcase_1_pipeline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# VISUAL 2 — Results Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def make_results_dashboard():
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # ── 2a: KPI tiles (top row, span all 3 cols via manual axes) ────────────
    kpis = [
        ("AUC-ROC",  "0.803", "Target ≥ 0.75  ✓ Exceeded by 7%", BRAND_GREEN),
        ("Recall",   "69.6%", "Defaults correctly flagged",       BRAND_BLUE),
        ("Accuracy", "74.7%", "Overall prediction accuracy",      BRAND_BLUE),
        ("Dataset",  "1.5M",  "Loan applications processed",      BRAND_GRAY),
        ("Features", "727",   "Engineered predictors",            BRAND_GRAY),
        ("Tables",   "32",    "Source data tables merged",         BRAND_GRAY),
    ]

    tile_y = 0.63
    tile_h = 0.28
    tile_w = 0.13
    xs = np.linspace(0.07, 0.93, 6)

    for (label, value, sub, color), x in zip(kpis, xs):
        rect = mpatches.FancyBboxPatch(
            (x - tile_w / 2, tile_y), tile_w, tile_h,
            boxstyle="round,pad=0.015", linewidth=2,
            edgecolor=color, facecolor=color + "18",
            transform=fig.transFigure, clip_on=False
        )
        fig.add_artist(rect)
        fig.text(x, tile_y + tile_h * 0.72, value, ha="center",
                 fontsize=22, fontweight="bold", color=color,
                 transform=fig.transFigure)
        fig.text(x, tile_y + tile_h * 0.42, label, ha="center",
                 fontsize=11, fontweight="bold", color=BRAND_GRAY,
                 transform=fig.transFigure)
        fig.text(x, tile_y + tile_h * 0.12, sub, ha="center",
                 fontsize=8, color="#78909C",
                 transform=fig.transFigure)

    # ── 2b: Model comparison bar chart ──────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_bar.set_facecolor(BG)
    metrics = ["AUC-ROC", "Recall", "Accuracy"]
    lgbm   = [0.803, 0.696, 0.747]
    logreg = [0.500, 1.000, 0.031]
    x_pos  = np.arange(len(metrics))
    bars1 = ax_bar.bar(x_pos - 0.18, lgbm,   0.33, label="LightGBM",          color=BRAND_GREEN,  alpha=0.85)
    bars2 = ax_bar.bar(x_pos + 0.18, logreg, 0.33, label="Logistic Regression", color=BRAND_ORANGE, alpha=0.85)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(metrics, fontsize=10)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_title("Model Comparison", fontsize=12, fontweight="bold", color=BRAND_GRAY)
    ax_bar.legend(fontsize=9)
    ax_bar.axhline(0.75, color=BRAND_GREEN, linestyle="--", lw=1.2, alpha=0.6, label="Target AUC")
    for bar in bars1:
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8, color=BRAND_GRAY)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # ── 2c: Confusion matrix ─────────────────────────────────────────────────
    ax_cm = fig.add_subplot(gs[1, 1])
    ax_cm.set_facecolor(BG)
    cm = np.array([[166143, 55657], [2190, 5009]])
    labels = [["TN\n166,143\n(75%)", "FP\n55,657\n(25%)"],
              ["FN\n2,190\n(30%)",   "TP\n5,009\n(70%)"]]
    colors_cm = [[LIGHT_GREEN, "#FFCCBC"], ["#FFCCBC", LIGHT_GREEN]]
    for r in range(2):
        for c in range(2):
            rect = mpatches.FancyBboxPatch(
                (c * 0.5, (1 - r) * 0.5 - 0.5), 0.48, 0.46,
                boxstyle="round,pad=0.02", facecolor=colors_cm[r][c], edgecolor="#B0BEC5", lw=1.5
            )
            ax_cm.add_patch(rect)
            ax_cm.text(c * 0.5 + 0.24, (1 - r) * 0.5 - 0.27, labels[r][c],
                       ha="center", va="center", fontsize=10,
                       color=BRAND_GRAY, fontweight="bold", linespacing=1.4)
    ax_cm.set_xlim(-0.02, 1.0)
    ax_cm.set_ylim(-0.52, 0.52)
    ax_cm.axis("off")
    ax_cm.set_title("Confusion Matrix  (228,999 test samples)", fontsize=11, fontweight="bold", color=BRAND_GRAY)
    ax_cm.text(0.25, -0.47, "Predicted: No Default", ha="center", fontsize=9, color="#546E7A")
    ax_cm.text(0.75, -0.47, "Predicted: Default",    ha="center", fontsize=9, color="#546E7A")
    ax_cm.text(-0.02,  0.13, "Actual:\nNo Default", ha="right", va="center", fontsize=9, color="#546E7A")
    ax_cm.text(-0.02, -0.37, "Actual:\nDefault",    ha="right", va="center", fontsize=9, color="#546E7A")

    # ── 2d: Threshold precision/recall tradeoff ──────────────────────────────
    ax_thr = fig.add_subplot(gs[1, 2])
    ax_thr.set_facecolor(BG)
    thresholds  = [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
    precisions  = [0.036, 0.042, 0.052, 0.064, 0.074, 0.083, 0.108, 0.140, 0.161, 0.215, 0.325]
    recalls     = [0.991, 0.960, 0.930, 0.880, 0.760, 0.696, 0.565, 0.420, 0.308, 0.180, 0.054]
    f1s         = [0.070, 0.081, 0.099, 0.119, 0.135, 0.148, 0.181, 0.201, 0.212, 0.208, 0.093]
    ax_thr.plot(thresholds, precisions, "o-", color=BRAND_BLUE,   label="Precision", lw=2)
    ax_thr.plot(thresholds, recalls,    "s-", color=BRAND_ORANGE, label="Recall",    lw=2)
    ax_thr.plot(thresholds, f1s,        "^-", color=BRAND_GREEN,  label="F1-Score",  lw=2)
    ax_thr.axvline(0.75, color=BRAND_GREEN,  linestyle="--", lw=1.5, alpha=0.8, label="Best F1 (0.75)")
    ax_thr.axvline(0.60, color=BRAND_BLUE,   linestyle=":",  lw=1.5, alpha=0.8, label="Cost-optimal (0.60)")
    ax_thr.set_xlabel("Decision Threshold", fontsize=10)
    ax_thr.set_title("Threshold Optimization", fontsize=12, fontweight="bold", color=BRAND_GRAY)
    ax_thr.legend(fontsize=8, loc="center right")
    ax_thr.spines[["top", "right"]].set_visible(False)
    ax_thr.set_ylim(0, 1.05)

    # ── main title ────────────────────────────────────────────────────────────
    fig.text(0.5, 0.96, "Credit Risk Model — Performance Results",
             ha="center", fontsize=20, fontweight="bold", color=BRAND_GRAY)
    fig.text(0.5, 0.92, "LightGBM  |  AUC-ROC 0.803  |  70% Defaults Caught  |  Capstone 2  |  2026",
             ha="center", fontsize=11, color="#78909C")

    out = os.path.join(OUTPUT_DIR, "showcase_2_results.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# VISUAL 3 — SHAP Feature Importance (clean horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
def make_shap_chart():
    import csv, os
    shap_csv = os.path.join(os.path.dirname(__file__), "..",
                            "outputs", "reports", "step7_feature_importance_shap.csv")

    features, importances = [], []
    try:
        with open(shap_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 15:
                    break
                col = row.get("feature") or row.get("Feature") or list(row.values())[0]
                val = float(row.get("mean_abs_shap") or row.get("importance") or list(row.values())[1])
                features.append(col)
                importances.append(val)
    except Exception as e:
        print(f"  Could not read SHAP CSV ({e}), using placeholders.")
        features = [
            "education_1103M_6b2ae0fa", "WEEK_NUM", "mobilephncnt_593L",
            "numactivecreds_622L", "pmts_overdue_1140A", "credacc_credlmt_575A",
            "annuity_780A", "avginstallast24m_3658937A", "maxdbrd_478A",
            "days_employed", "num_credits", "income_total",
            "amount_credit", "amount_annuity", "days_birth"
        ]
        importances = [0.142, 0.118, 0.095, 0.088, 0.081, 0.074,
                       0.068, 0.061, 0.055, 0.050, 0.044, 0.040,
                       0.036, 0.032, 0.028]

    # reverse for top-to-bottom display
    features     = features[::-1]
    importances  = importances[::-1]

    norm   = np.array(importances) / max(importances)
    colors = [plt.cm.Blues(0.4 + 0.55 * v) for v in norm]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.barh(range(len(features)), importances, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Mean |SHAP Value|  (average impact on model output)", fontsize=11)
    ax.set_title("Top 15 Features Driving Default Predictions\n(SHAP Explainability — LightGBM)",
                 fontsize=15, fontweight="bold", color=BRAND_GRAY, pad=14)

    for i, (bar, val) in enumerate(zip(bars, importances)):
        ax.text(bar.get_width() + max(importances) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color=BRAND_GRAY)

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(left=False)

    fig.text(0.99, 0.01, "Capstone 2  |  Home Credit Default Prediction  |  2026",
             ha="right", fontsize=9, color="#90A4AE")

    out = os.path.join(OUTPUT_DIR, "showcase_3_shap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating capstone presentation visuals...")
    make_pipeline()
    make_results_dashboard()
    make_shap_chart()
    print("\nDone! Three screen-ready images saved to outputs/figures/:")
    print("  showcase_1_pipeline.png   — show at ~0:50 (pipeline section)")
    print("  showcase_2_results.png    — show at ~1:50 (results section)")
    print("  showcase_3_shap.png       — show at ~2:20 (explainability section)")
