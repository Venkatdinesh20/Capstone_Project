"""
STEP 9: THRESHOLD OPTIMIZATION
Find optimal classification threshold for business objectives
Implements User Story 5: Model Evaluation - Best Threshold Identification
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score, recall_score,
    accuracy_score, roc_curve, confusion_matrix
)
from datetime import datetime
import json

# Set plotting style
plt.style.use('default')
sns.set_palette("Set2")

print("=" * 80)
print("STEP 9: THRESHOLD OPTIMIZATION")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load test data
print("\n### Loading Test Data ###")
X_test = pd.read_parquet(DATA_PROCESSED_DIR / 'step4_X_test.parquet')
y_test = pd.read_parquet(DATA_PROCESSED_DIR / 'step4_y_test.parquet')[TARGET_COL]
print(f"Test samples: {len(X_test):,}")
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

# Get predictions
print("\n### Generating Predictions ###")
y_pred_proba = model.predict(X_test)
print(f"Predicted probabilities: min={y_pred_proba.min():.4f}, "
      f"max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")

# ============================================================================
# THRESHOLD ANALYSIS
# ============================================================================
print("\n### Analyzing Thresholds ###")
print("Testing thresholds from 0.1 to 0.9...")

# Test multiple thresholds
thresholds = np.arange(0.1, 0.91, 0.05)
threshold_results = []

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Confusion matrix elements
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # Calculate Youden's J statistic (sensitivity + specificity - 1)
    youden_j = recall + specificity - 1
    
    threshold_results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'npv': npv,
        'youden_j': youden_j,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'predicted_positives': tp + fp,
        'predicted_positive_rate': (tp + fp) / len(y_test)
    })

df_thresholds = pd.DataFrame(threshold_results)

# ============================================================================
# BUSINESS COST ANALYSIS
# ============================================================================
print("\n### Business Cost Analysis ###")
print("Estimating financial impact of different thresholds...")

# Define business costs (example values - adjust based on real business case)
COST_FALSE_NEGATIVE = 10000  # Cost of missing a default (loan defaults completely)
COST_FALSE_POSITIVE = 500    # Cost of rejecting good customer (lost interest revenue)
COST_TRUE_POSITIVE = 1000    # Cost of correctly rejecting bad loan (manual review)
BENEFIT_TRUE_NEGATIVE = 100  # Benefit of correctly approving good loan (interest revenue)

print(f"\nAssumed Business Costs:")
print(f"  False Negative (missed default): ${COST_FALSE_NEGATIVE:,}")
print(f"  False Positive (rejected good customer): ${COST_FALSE_POSITIVE:,}")
print(f"  True Positive (caught default): ${COST_TRUE_POSITIVE:,}")
print(f"  True Negative (approved good loan): ${BENEFIT_TRUE_NEGATIVE:,}")

# Calculate financial impact for each threshold
df_thresholds['total_cost'] = (
    df_thresholds['fn'] * COST_FALSE_NEGATIVE +
    df_thresholds['fp'] * COST_FALSE_POSITIVE +
    df_thresholds['tp'] * COST_TRUE_POSITIVE -
    df_thresholds['tn'] * BENEFIT_TRUE_NEGATIVE
)

df_thresholds['net_benefit'] = -df_thresholds['total_cost']

# ============================================================================
# OPTIMAL THRESHOLDS FOR DIFFERENT OBJECTIVES
# ============================================================================
print("\n### Optimal Thresholds by Objective ###")
print("=" * 80)

objectives = {
    'Max F1-Score': ('f1_score', 'max'),
    'Max Recall (Catch Defaults)': ('recall', 'max'),
    'Max Precision (Minimize False Alarms)': ('precision', 'max'),
    'Max Accuracy': ('accuracy', 'max'),
    'Max Youden\'s J': ('youden_j', 'max'),
    'Min Total Cost': ('total_cost', 'min'),
    'Max Net Benefit': ('net_benefit', 'max')
}

optimal_thresholds = {}

for objective_name, (metric, direction) in objectives.items():
    if direction == 'max':
        best_idx = df_thresholds[metric].idxmax()
    else:
        best_idx = df_thresholds[metric].idxmin()
    
    best_row = df_thresholds.loc[best_idx]
    optimal_thresholds[objective_name] = {
        'threshold': float(best_row['threshold']),
        'precision': float(best_row['precision']),
        'recall': float(best_row['recall']),
        'f1_score': float(best_row['f1_score']),
        'accuracy': float(best_row['accuracy']),
        'net_benefit': float(best_row['net_benefit'])
    }
    
    print(f"\n{objective_name}:")
    print(f"  Threshold: {best_row['threshold']:.2f}")
    print(f"  Precision: {best_row['precision']:.4f}")
    print(f"  Recall:    {best_row['recall']:.4f}")
    print(f"  F1-Score:  {best_row['f1_score']:.4f}")
    print(f"  Accuracy:  {best_row['accuracy']:.4f}")
    print(f"  Net Benefit: ${best_row['net_benefit']:,.0f}")
    print(f"  Predicted Positive Rate: {best_row['predicted_positive_rate']:.2%}")

# ============================================================================
# VISUALIZATION 1: METRICS vs THRESHOLD
# ============================================================================
print("\n### Creating Threshold Analysis Plots ###")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Precision, Recall, F1-Score
ax = axes[0, 0]
ax.plot(df_thresholds['threshold'], df_thresholds['precision'], 
        'b-', linewidth=2, label='Precision', marker='o', markersize=4)
ax.plot(df_thresholds['threshold'], df_thresholds['recall'], 
        'r-', linewidth=2, label='Recall', marker='s', markersize=4)
ax.plot(df_thresholds['threshold'], df_thresholds['f1_score'], 
        'g-', linewidth=2, label='F1-Score', marker='^', markersize=4)
ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Default (0.5)')
ax.set_xlabel('Classification Threshold', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Precision, Recall, F1-Score vs Threshold', fontsize=12, pad=10)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Plot 2: Predicted Positive Rate
ax = axes[0, 1]
ax.plot(df_thresholds['threshold'], df_thresholds['predicted_positive_rate'] * 100, 
        'purple', linewidth=2, marker='o', markersize=4)
ax.axhline(y_test.mean() * 100, color='red', linestyle='--', 
          label=f'Actual Default Rate ({y_test.mean()*100:.2f}%)')
ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Default (0.5)')
ax.set_xlabel('Classification Threshold', fontsize=11)
ax.set_ylabel('Predicted Positive Rate (%)', fontsize=11)
ax.set_title('Predicted Default Rate vs Threshold', fontsize=12, pad=10)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Net Benefit
ax = axes[1, 0]
ax.plot(df_thresholds['threshold'], df_thresholds['net_benefit'] / 1000000, 
        'darkgreen', linewidth=2, marker='o', markersize=4)
best_benefit_idx = df_thresholds['net_benefit'].idxmax()
best_benefit_thresh = df_thresholds.loc[best_benefit_idx, 'threshold']
best_benefit_value = df_thresholds.loc[best_benefit_idx, 'net_benefit']
ax.axvline(best_benefit_thresh, color='red', linestyle='--', 
          label=f'Optimal ({best_benefit_thresh:.2f})')
ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Default (0.5)')
ax.set_xlabel('Classification Threshold', fontsize=11)
ax.set_ylabel('Net Benefit (Million $)', fontsize=11)
ax.set_title('Business Net Benefit vs Threshold', fontsize=12, pad=10)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Confusion Matrix Components
ax = axes[1, 1]
ax.plot(df_thresholds['threshold'], df_thresholds['tp'], 
        'g-', linewidth=2, label='True Positives', marker='o', markersize=4)
ax.plot(df_thresholds['threshold'], df_thresholds['fp'] / 10, 
        'r-', linewidth=2, label='False Positives / 10', marker='s', markersize=4)
ax.plot(df_thresholds['threshold'], df_thresholds['fn'], 
        'orange', linewidth=2, label='False Negatives', marker='^', markersize=4)
ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Default (0.5)')
ax.set_xlabel('Classification Threshold', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Confusion Matrix Components vs Threshold', fontsize=12, pad=10)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
threshold_analysis_path = FIGURES_DIR / 'step9_threshold_analysis.png'
plt.savefig(threshold_analysis_path, dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print(f"OK Saved: {threshold_analysis_path.name}")

# ============================================================================
# VISUALIZATION 2: PRECISION-RECALL TRADEOFF
# ============================================================================
print("\n### Creating Precision-Recall Tradeoff Plot ###")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot precision vs recall curve
ax.plot(df_thresholds['recall'], df_thresholds['precision'], 
        'b-', linewidth=3, label='Precision-Recall Curve')

# Mark key thresholds
interesting_thresholds = [0.3, 0.5, 0.7]
for thresh in interesting_thresholds:
    # Find nearest threshold (to handle floating point precision)
    nearest_idx = (df_thresholds['threshold'] - thresh).abs().idxmin()
    row = df_thresholds.loc[nearest_idx]
    actual_thresh = row['threshold']
    ax.plot(row['recall'], row['precision'], 'ro', markersize=10)
    ax.annotate(f'Threshold = {actual_thresh:.2f}', 
               xy=(row['recall'], row['precision']),
               xytext=(10, -10), textcoords='offset points',
               fontsize=10, ha='left',
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Tradeoff\nChoosing the Right Threshold', 
             fontsize=14, pad=15)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])

plt.tight_layout()
pr_tradeoff_path = FIGURES_DIR / 'step9_precision_recall_tradeoff.png'
plt.savefig(pr_tradeoff_path, dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print(f"OK Saved: {pr_tradeoff_path.name}")

# ============================================================================
# VISUALIZATION 3: COST ANALYSIS
# ============================================================================
print("\n### Creating Cost Analysis Plot ###")

fig, ax = plt.subplots(figsize=(12, 8))

# Plot costs
ax.plot(df_thresholds['threshold'], df_thresholds['fn'] * COST_FALSE_NEGATIVE / 1000000,
        'r-', linewidth=2, label='Cost: Missed Defaults (FN)', marker='o', markersize=4)
ax.plot(df_thresholds['threshold'], df_thresholds['fp'] * COST_FALSE_POSITIVE / 1000000,
        'orange', linewidth=2, label='Cost: False Alarms (FP)', marker='s', markersize=4)
ax.plot(df_thresholds['threshold'], df_thresholds['total_cost'] / 1000000,
        'darkred', linewidth=3, label='Total Cost', marker='^', markersize=5)

# Mark minimum cost
min_cost_idx = df_thresholds['total_cost'].idxmin()
min_cost_thresh = df_thresholds.loc[min_cost_idx, 'threshold']
min_cost_value = df_thresholds.loc[min_cost_idx, 'total_cost']
ax.axvline(min_cost_thresh, color='green', linestyle='--', linewidth=2,
          label=f'Minimum Cost Threshold ({min_cost_thresh:.2f})')

ax.set_xlabel('Classification Threshold', fontsize=12)
ax.set_ylabel('Cost (Million $)', fontsize=12)
ax.set_title('Business Cost Analysis by Threshold\nMinimizing Total Cost', 
             fontsize=14, pad=15)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
cost_analysis_path = FIGURES_DIR / 'step9_cost_analysis.png'
plt.savefig(cost_analysis_path, dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print(f"OK Saved: {cost_analysis_path.name}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n### Saving Optimization Results ###")

# Save threshold analysis CSV
threshold_csv_path = REPORTS_DIR / 'step9_threshold_analysis.csv'
df_thresholds.to_csv(threshold_csv_path, index=False)
print(f"OK Threshold analysis CSV: {threshold_csv_path.name}")

# Save optimal thresholds JSON
optimal_path = REPORTS_DIR / 'step9_optimal_thresholds.json'
optimization_results = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'test_set_size': int(len(X_test)),
    'actual_default_rate': float(y_test.mean()),
    'business_assumptions': {
        'cost_false_negative': COST_FALSE_NEGATIVE,
        'cost_false_positive': COST_FALSE_POSITIVE,
        'cost_true_positive': COST_TRUE_POSITIVE,
        'benefit_true_negative': BENEFIT_TRUE_NEGATIVE
    },
    'optimal_thresholds': optimal_thresholds,
    'recommendation': {
        'threshold': float(df_thresholds.loc[min_cost_idx, 'threshold']),
        'objective': 'Minimize Total Business Cost',
        'expected_net_benefit': float(df_thresholds.loc[min_cost_idx, 'net_benefit']),
        'precision': float(df_thresholds.loc[min_cost_idx, 'precision']),
        'recall': float(df_thresholds.loc[min_cost_idx, 'recall']),
        'f1_score': float(df_thresholds.loc[min_cost_idx, 'f1_score'])
    }
}

with open(optimal_path, 'w') as f:
    json.dump(optimization_results, f, indent=2)
print(f"OK Optimal thresholds JSON: {optimal_path.name}")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("THRESHOLD OPTIMIZATION COMPLETE")
print("=" * 80)

print("\n### RECOMMENDATIONS ###")
print("\nDepending on business objectives, consider these thresholds:\n")

print(f"1. BALANCED APPROACH (Max F1-Score):")
print(f"   Threshold: {optimal_thresholds['Max F1-Score']['threshold']:.2f}")
print(f"   → Best for balanced precision-recall\n")

print(f"2. RISK-AVERSE (Max Recall):")
print(f"   Threshold: {optimal_thresholds['Max Recall (Catch Defaults)']['threshold']:.2f}")
print(f"   → Catch as many defaults as possible\n")

print(f"3. COST-OPTIMIZED (Min Total Cost):")
print(f"   Threshold: {optimal_thresholds['Min Total Cost']['threshold']:.2f}")
print(f"   → Minimize financial losses (RECOMMENDED)\n")

print(f"4. CONSERVATIVE (Max Precision):")
print(f"   Threshold: {optimal_thresholds['Max Precision (Minimize False Alarms)']['threshold']:.2f}")
print(f"   → Minimize false alarms\n")

print(f"DEFAULT THRESHOLD (0.5):")
# Find nearest to 0.5
nearest_50_idx = (df_thresholds['threshold'] - 0.5).abs().idxmin()
row_50 = df_thresholds.loc[nearest_50_idx]
print(f"   Threshold: {row_50['threshold']:.2f}")
print(f"   Precision: {row_50['precision']:.4f}")
print(f"   Recall:    {row_50['recall']:.4f}")
print(f"   F1-Score:  {row_50['f1_score']:.4f}")
print(f"   Net Benefit: ${row_50['net_benefit']:,.0f}\n")

print(f"Files saved:")
print(f"  - {FIGURES_DIR}/step9_threshold_analysis.png")
print(f"  - {FIGURES_DIR}/step9_precision_recall_tradeoff.png")
print(f"  - {FIGURES_DIR}/step9_cost_analysis.png")
print(f"  - {REPORTS_DIR}/step9_threshold_analysis.csv")
print(f"  - {REPORTS_DIR}/step9_optimal_thresholds.json")

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
