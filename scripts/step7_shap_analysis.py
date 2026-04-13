"""
STEP 7: SHAP ANALYSIS - MODEL EXPLAINABILITY
Generate SHAP values and explanations for model predictions
Implements User Story 2: Understanding Predictions
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

import joblib
import shap
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Set plotting style
plt.style.use('default')
shap.initjs()

print("=" * 80)
print("STEP 7: SHAP ANALYSIS - MODEL EXPLAINABILITY")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load test data
print("\n### Loading Test Data ###")
X_test = pd.read_parquet(DATA_PROCESSED_DIR / 'step4_X_test.parquet')
y_test = pd.read_parquet(DATA_PROCESSED_DIR / 'step4_y_test.parquet')[TARGET_COL]
print(f"Test samples: {len(X_test):,}")
print(f"Features: {X_test.shape[1]}")
print(f"Default rate: {y_test.mean():.2%}")

# Load best model (LightGBM)
print("\n### Loading Best Model ###")
model_path = MODELS_DIR / 'lightgbm.pkl'
if not model_path.exists():
    print(f"X  Model not found: {model_path}")
    print("   Run step5_model_training.py first")
    sys.exit(1)

model = joblib.load(model_path)
print(f"OK Loaded: LightGBM model")

# Sample data for SHAP (to manage computation time)
print(f"\n### Sampling Data for SHAP ###")
print(f"Using {SHAP_SAMPLE_SIZE:,} samples for SHAP calculation")

if len(X_test) > SHAP_SAMPLE_SIZE:
    # Stratified sampling to maintain class distribution
    sample_indices = y_test.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE).index
    X_shap = X_test.loc[sample_indices]
    y_shap = y_test.loc[sample_indices]
else:
    X_shap = X_test
    y_shap = y_test

print(f"SHAP sample size: {len(X_shap):,}")
print(f"SHAP sample default rate: {y_shap.mean():.2%}")

# Create SHAP explainer
print("\n### Creating SHAP Explainer ###")
print("This may take several minutes...")

# For LightGBM, use TreeExplainer (much faster than KernelExplainer)
explainer = shap.TreeExplainer(model)
print("OK TreeExplainer created")

# Calculate SHAP values
print("\n### Calculating SHAP Values ###")
print("Computing SHAP values for all samples...")
shap_values = explainer.shap_values(X_shap)

# LightGBM binary classification returns array directly (for positive class)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Use positive class (default) SHAP values

print(f"OK SHAP values computed: {shap_values.shape}")

# ============================================================================
# VISUALIZATION 1: SHAP SUMMARY PLOT (Feature Importance)
# ============================================================================
print("\n### Creating SHAP Summary Plot ###")
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(
    shap_values, 
    X_shap, 
    max_display=SHAP_MAX_DISPLAY,
    show=False
)
plt.title("SHAP Feature Importance Summary\nTop 20 Most Impactful Features", 
          fontsize=14, pad=20)
plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
plt.tight_layout()
summary_path = FIGURES_DIR / 'step7_shap_summary_plot.png'
plt.savefig(summary_path, dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print(f"OK Saved: {summary_path.name}")

# ============================================================================
# VISUALIZATION 2: SHAP BAR PLOT (Mean Absolute SHAP Values)
# ============================================================================
print("\n### Creating SHAP Bar Plot ###")
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(
    shap_values, 
    X_shap, 
    plot_type="bar",
    max_display=SHAP_MAX_DISPLAY,
    show=False
)
plt.title("Mean Absolute SHAP Values\nOverall Feature Importance", 
          fontsize=14, pad=20)
plt.xlabel("Mean |SHAP Value|", fontsize=12)
plt.tight_layout()
bar_path = FIGURES_DIR / 'step7_shap_bar_plot.png'
plt.savefig(bar_path, dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print(f"OK Saved: {bar_path.name}")

# ============================================================================
# VISUALIZATION 3: SHAP WATERFALL PLOT (Example Predictions)
# ============================================================================
print("\n### Creating SHAP Waterfall Plots ###")

# Find examples: High risk default (predicted correctly), Low risk non-default
y_pred_proba = model.predict(X_shap)

# Example 1: High-risk default (True Positive)
high_risk_defaults = np.where((y_shap == 1) & (y_pred_proba > 0.7))[0]
if len(high_risk_defaults) > 0:
    idx_high = high_risk_defaults[0]
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx_high],
            base_values=explainer.expected_value,
            data=X_shap.iloc[idx_high],
            feature_names=X_shap.columns.tolist()
        ),
        max_display=15,
        show=False
    )
    plt.title(f"SHAP Waterfall: High-Risk Default Example\n"
              f"Predicted Probability: {y_pred_proba[idx_high]:.2%} | Actual: Default",
              fontsize=12, pad=20)
    plt.tight_layout()
    waterfall_high_path = FIGURES_DIR / 'step7_shap_waterfall_high_risk.png'
    plt.savefig(waterfall_high_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"OK Saved: {waterfall_high_path.name}")

# Example 2: Low-risk non-default (True Negative)
low_risk_non_defaults = np.where((y_shap == 0) & (y_pred_proba < 0.3))[0]
if len(low_risk_non_defaults) > 0:
    idx_low = low_risk_non_defaults[0]
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx_low],
            base_values=explainer.expected_value,
            data=X_shap.iloc[idx_low],
            feature_names=X_shap.columns.tolist()
        ),
        max_display=15,
        show=False
    )
    plt.title(f"SHAP Waterfall: Low-Risk Non-Default Example\n"
              f"Predicted Probability: {y_pred_proba[idx_low]:.2%} | Actual: No Default",
              fontsize=12, pad=20)
    plt.tight_layout()
    waterfall_low_path = FIGURES_DIR / 'step7_shap_waterfall_low_risk.png'
    plt.savefig(waterfall_low_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"OK Saved: {waterfall_low_path.name}")

# ============================================================================
# VISUALIZATION 4: SHAP DEPENDENCE PLOTS (Top 3 Features)
# ============================================================================
print("\n### Creating SHAP Dependence Plots ###")

# Get top 3 most important features
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_feature_indices = np.argsort(mean_abs_shap)[-3:][::-1]
top_features = X_shap.columns[top_feature_indices].tolist()

print(f"Top 3 features: {top_features}")

for i, feature in enumerate(top_features, 1):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        feature,
        shap_values,
        X_shap,
        show=False,
        ax=ax
    )
    plt.title(f"SHAP Dependence Plot: {feature}\n"
              f"How Feature Value Affects Model Output",
              fontsize=12, pad=15)
    plt.tight_layout()
    dep_path = FIGURES_DIR / f'step7_shap_dependence_{i}_{feature[:30]}.png'
    plt.savefig(dep_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"OK Saved: {dep_path.name}")

# ============================================================================
# FEATURE IMPORTANCE RANKING
# ============================================================================
print("\n### Computing Feature Importance Rankings ###")

# Calculate mean absolute SHAP values for each feature
feature_importance = pd.DataFrame({
    'feature': X_shap.columns,
    'importance': np.abs(shap_values).mean(axis=0)
})

# Sort by importance
feature_importance = feature_importance.sort_values('importance', ascending=False)
feature_importance['rank'] = range(1, len(feature_importance) + 1)

# Add percentage contribution
total_importance = feature_importance['importance'].sum()
feature_importance['importance_pct'] = (
    feature_importance['importance'] / total_importance * 100
)

# Cumulative importance
feature_importance['cumulative_pct'] = feature_importance['importance_pct'].cumsum()

print(f"\nTop 10 Most Important Features:")
print("-" * 80)
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['rank']:>2}. {row['feature']:<50} "
          f"SHAP: {row['importance']:.4f} ({row['importance_pct']:.2f}%)")

# ============================================================================
# SHAP VALUE STATISTICS
# ============================================================================
print("\n### SHAP Value Statistics ###")

shap_stats = {
    'total_samples_analyzed': int(len(X_shap)),
    'total_features': int(X_shap.shape[1]),
    'base_value': float(explainer.expected_value),
    'mean_prediction': float(y_pred_proba.mean()),
    'top_20_features': feature_importance.head(20)[['feature', 'importance', 'importance_pct']].to_dict('records'),
    'top_20_cumulative_importance_pct': float(feature_importance.head(20)['cumulative_pct'].iloc[-1]),
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

print(f"Base value (expected model output): {shap_stats['base_value']:.4f}")
print(f"Mean prediction: {shap_stats['mean_prediction']:.2%}")
print(f"Top 20 features explain {shap_stats['top_20_cumulative_importance_pct']:.1f}% of importance")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n### Saving Results ###")

# Save feature importance CSV
importance_path = REPORTS_DIR / 'step7_feature_importance_shap.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"OK Feature importance CSV: {importance_path.name}")

# Save SHAP statistics JSON
stats_path = REPORTS_DIR / 'step7_shap_statistics.json'
with open(stats_path, 'w') as f:
    json.dump(shap_stats, f, indent=2)
print(f"OK SHAP statistics JSON: {stats_path.name}")

# Save SHAP values (for potential future use)
shap_values_path = DATA_PROCESSED_DIR / 'step7_shap_values.npy'
np.save(shap_values_path, shap_values)
print(f"OK SHAP values array: {shap_values_path.name}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SHAP ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nAnalyzed: {len(X_shap):,} samples")
print(f"Generated: 6+ visualizations")
print(f"Top feature: {feature_importance.iloc[0]['feature']}")
print(f"   Importance: {feature_importance.iloc[0]['importance']:.4f}")
print(f"   Contribution: {feature_importance.iloc[0]['importance_pct']:.2f}%")

print(f"\nFiles saved:")
print(f"  - {FIGURES_DIR}/step7_shap_*.png (6 visualizations)")
print(f"  - {REPORTS_DIR}/step7_feature_importance_shap.csv")
print(f"  - {REPORTS_DIR}/step7_shap_statistics.json")
print(f"  - {DATA_PROCESSED_DIR}/step7_shap_values.npy")

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
