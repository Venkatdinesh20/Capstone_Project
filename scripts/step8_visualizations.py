"""
STEP 8: COMPREHENSIVE VISUALIZATIONS
Create visualization dashboard for model performance and insights
Implements User Story 6: Feature Importance & Analysis
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
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.calibration import calibration_curve
from datetime import datetime
import json

# Set plotting style
plt.style.use('default')
sns.set_palette("Set2")

print("=" * 80)
print("STEP 8: COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load test data
print("\n### Loading Test Data ###")
X_test = pd.read_parquet(DATA_PROCESSED_DIR / 'step4_X_test.parquet')
y_test = pd.read_parquet(DATA_PROCESSED_DIR / 'step4_y_test.parquet')[TARGET_COL]
print(f"Test samples: {len(X_test):,}")
print(f"Default rate: {y_test.mean():.2%}")

# Load trained models
print("\n### Loading Trained Models ###")
models = {}
model_files = {
    'LightGBM': 'lightgbm.pkl',
    'Logistic Regression': 'logistic_regression.pkl'
}

for model_name, model_file in model_files.items():
    model_path = MODELS_DIR / model_file
    if model_path.exists():
        models[model_name] = joblib.load(model_path)
        print(f"OK Loaded: {model_name}")
    else:
        print(f"X  Not found: {model_name}")

if not models:
    print("\nX  No models found! Run step5_model_training.py first")
    sys.exit(1)

# ============================================================================
# VISUALIZATION 1: ROC CURVES (All Models)
# ============================================================================
print("\n### Creating ROC Curves ###")

fig, ax = plt.subplots(figsize=(10, 8))

for model_name, model in models.items():
    # Get predictions
    if model_name == 'LightGBM':
        y_pred_proba = model.predict(X_test)
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    ax.plot(fpr, tpr, linewidth=2, 
            label=f'{model_name} (AUC = {roc_auc:.4f})')

# Plot random classifier
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.50)')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison\nHigher AUC = Better Performance', 
             fontsize=14, pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = FIGURES_DIR / 'step8_roc_curves.png'
plt.savefig(roc_path, dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print(f"OK Saved: {roc_path.name}")

# ============================================================================
# VISUALIZATION 2: PRECISION-RECALL CURVES
# ============================================================================
print("\n### Creating Precision-Recall Curves ###")

fig, ax = plt.subplots(figsize=(10, 8))

for model_name, model in models.items():
    # Get predictions
    if model_name == 'LightGBM':
        y_pred_proba = model.predict(X_test)
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Plot
    ax.plot(recall, precision, linewidth=2,
            label=f'{model_name} (AP = {avg_precision:.4f})')

# Baseline (proportion of positives)
baseline = y_test.mean()
ax.plot([0, 1], [baseline, baseline], 'k--', linewidth=2,
        label=f'Baseline (No Skill = {baseline:.4f})')

ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - Model Comparison\n'
             'Important for Imbalanced Classification', 
             fontsize=14, pad=15)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
pr_path = FIGURES_DIR / 'step8_precision_recall_curves.png'
plt.savefig(pr_path, dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print(f"OK Saved: {pr_path.name}")

# ============================================================================
# VISUALIZATION 3: CONFUSION MATRIX HEATMAP (Best Model)
# ============================================================================
print("\n### Creating Confusion Matrix Heatmap ###")

# Use LightGBM (best model)
best_model = models['LightGBM']
y_pred_proba = best_model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted Label', fontsize=12)
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_title('Confusion Matrix (Raw Counts)\nLightGBM Model', 
                  fontsize=13, pad=15)
axes[0].set_xticklabels(['No Default', 'Default'])
axes[0].set_yticklabels(['No Default', 'Default'])

# Normalized
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
            cbar_kws={'label': 'Proportion'})
axes[1].set_xlabel('Predicted Label', fontsize=12)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_title('Confusion Matrix (Normalized)\nLightGBM Model', 
                  fontsize=13, pad=15)
axes[1].set_xticklabels(['No Default', 'Default'])
axes[1].set_yticklabels(['No Default', 'Default'])

plt.tight_layout()
cm_path = FIGURES_DIR / 'step8_confusion_matrix.png'
plt.savefig(cm_path, dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print(f"OK Saved: {cm_path.name}")

# ============================================================================
# VISUALIZATION 4: PREDICTION DISTRIBUTION
# ============================================================================
print("\n### Creating Prediction Distribution Plot ###")

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Get predictions for LightGBM
y_pred_proba = best_model.predict(X_test)

# Separate by actual class
defaults = y_pred_proba[y_test == 1]
non_defaults = y_pred_proba[y_test == 0]

# Plot 1: Overlapping histograms
axes[0].hist(non_defaults, bins=50, alpha=0.6, label='Actual: No Default', 
             color='green', edgecolor='black')
axes[0].hist(defaults, bins=50, alpha=0.6, label='Actual: Default', 
             color='red', edgecolor='black')
axes[0].axvline(0.5, color='black', linestyle='--', linewidth=2, 
                label='Decision Threshold (0.5)')
axes[0].set_xlabel('Predicted Default Probability', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Prediction Distribution by Actual Class\nLightGBM Model', 
                  fontsize=13, pad=15)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2: Box plots
data_for_box = pd.DataFrame({
    'Predicted Probability': y_pred_proba,
    'Actual Class': ['Default' if y == 1 else 'No Default' for y in y_test]
})
sns.boxplot(data=data_for_box, x='Actual Class', y='Predicted Probability', 
            ax=axes[1], palette=['green', 'red'])
axes[1].axhline(0.5, color='black', linestyle='--', linewidth=2, 
                label='Threshold (0.5)')
axes[1].set_ylabel('Predicted Default Probability', fontsize=12)
axes[1].set_xlabel('Actual Class', fontsize=12)
axes[1].set_title('Prediction Distribution - Box Plot Comparison', 
                  fontsize=13, pad=15)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
dist_path = FIGURES_DIR / 'step8_prediction_distribution.png'
plt.savefig(dist_path, dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print(f"OK Saved: {dist_path.name}")

# ============================================================================
# VISUALIZATION 5: CALIBRATION CURVE
# ============================================================================
print("\n### Creating Calibration Curve ###")

fig, ax = plt.subplots(figsize=(10, 8))

# Calculate calibration curve for LightGBM
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_pred_proba, n_bins=10, strategy='uniform'
)

# Plot calibration curve
ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
        linewidth=2, label='LightGBM', markersize=8)

# Perfect calibration line
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Fraction of Positives (Actual)', fontsize=12)
ax.set_title('Calibration Curve\nHow Well Does Predicted Probability Match Reality?', 
             fontsize=14, pad=15)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
calib_path = FIGURES_DIR / 'step8_calibration_curve.png'
plt.savefig(calib_path, dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print(f"OK Saved: {calib_path.name}")

# ============================================================================
# VISUALIZATION 6: MODEL COMPARISON BAR CHART
# ============================================================================
print("\n### Creating Model Comparison Chart ###")

# Load evaluation results
results_path = REPORTS_DIR / 'step6_evaluation_results.json'
if results_path.exists():
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract metrics
    metrics = ['auc_roc', 'recall', 'precision', 'f1_score', 'accuracy']
    metric_labels = ['AUC-ROC', 'Recall', 'Precision', 'F1-Score', 'Accuracy']
    
    # Create data for plotting
    comparison_data = []
    for model_name in results.keys():
        model_readable = 'LightGBM' if model_name == 'lightgbm' else 'Logistic Regression'
        for metric, label in zip(metrics, metric_labels):
            comparison_data.append({
                'Model': model_readable,
                'Metric': label,
                'Value': results[model_name][metric]
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(metric_labels))
    width = 0.35
    
    models_list = df_comparison['Model'].unique()
    for i, model in enumerate(models_list):
        model_data = df_comparison[df_comparison['Model'] == model]
        offset = width * (i - len(models_list)/2 + 0.5)
        bars = ax.bar(x + offset, model_data['Value'], width, 
                     label=model, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Evaluation Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison\nAll Metrics on Test Set', 
                 fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    comparison_path = FIGURES_DIR / 'step8_model_comparison.png'
    plt.savefig(comparison_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"OK Saved: {comparison_path.name}")
else:
    print(f"X  Evaluation results not found: {results_path}")

# ============================================================================
# VISUALIZATION 7: FEATURE IMPORTANCE (from LightGBM)
# ============================================================================
print("\n### Creating Feature Importance Plot ###")

if hasattr(best_model, 'feature_importance'):
    # Get feature importance
    importance = best_model.feature_importance()
    feature_names = X_test.columns
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(20)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                  color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance (LightGBM Gain)', fontsize=12)
    ax.set_title('Top 20 Most Important Features\nLightGBM Model', 
                 fontsize=14, pad=15)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{width:.0f}',
               ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    importance_path = FIGURES_DIR / 'step8_feature_importance_lightgbm.png'
    plt.savefig(importance_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"OK Saved: {importance_path.name}")
    
    # Save to CSV
    importance_csv_path = REPORTS_DIR / 'step8_feature_importance_lightgbm.csv'
    importance_df_full = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    importance_df_full.to_csv(importance_csv_path, index=False)
    print(f"OK Saved: {importance_csv_path.name}")

# ============================================================================
# VISUALIZATION 8: CLASS DISTRIBUTION
# ============================================================================
print("\n### Creating Class Distribution Plot ###")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual distribution
class_counts = y_test.value_counts()
colors = ['green', 'red']
axes[0].bar(['No Default', 'Default'], class_counts.values, 
           color=colors, alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Actual Class Distribution\nTest Set', 
                 fontsize=13, pad=15)
axes[0].grid(True, alpha=0.3, axis='y')

# Add counts on bars
for i, (label, count) in enumerate(zip(['No Default', 'Default'], class_counts.values)):
    pct = count / len(y_test) * 100
    axes[0].text(i, count, f'{count:,}\n({pct:.2f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Predicted distribution (using threshold 0.5)
y_pred = (y_pred_proba > 0.5).astype(int)
pred_counts = pd.Series(y_pred).value_counts().sort_index()
axes[1].bar(['No Default', 'Default'], pred_counts.values, 
           color=colors, alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Predicted Class Distribution\nLightGBM (Threshold=0.5)', 
                 fontsize=13, pad=15)
axes[1].grid(True, alpha=0.3, axis='y')

# Add counts on bars
for i, count in enumerate(pred_counts.values):
    pct = count / len(y_pred) * 100
    axes[1].text(i, count, f'{count:,}\n({pct:.2f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
class_dist_path = FIGURES_DIR / 'step8_class_distribution.png'
plt.savefig(class_dist_path, dpi=FIGURE_DPI, bbox_inches='tight')
plt.close()
print(f"OK Saved: {class_dist_path.name}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n### Creating Summary Statistics ###")

summary_stats = {
    'visualization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'test_set_size': int(len(X_test)),
    'default_rate': float(y_test.mean()),
    'models_evaluated': list(models.keys()),
    'visualizations_created': [
        'ROC Curves',
        'Precision-Recall Curves',
        'Confusion Matrix',
        'Prediction Distribution',
        'Calibration Curve',
        'Model Comparison',
        'Feature Importance',
        'Class Distribution'
    ],
    'best_model': 'LightGBM',
    'best_model_auc': float(roc_auc_score(y_test, best_model.predict(X_test)))
}

# Save summary
summary_path = REPORTS_DIR / 'step8_visualization_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary_stats, f, indent=2)
print(f"\nOK Summary statistics: {summary_path.name}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION DASHBOARD COMPLETE")
print("=" * 80)
print(f"\nCreated 8 comprehensive visualizations:")
print(f"  1. ROC Curves (model comparison)")
print(f"  2. Precision-Recall Curves (imbalanced data focus)")
print(f"  3. Confusion Matrix (raw + normalized)")
print(f"  4. Prediction Distribution (by actual class)")
print(f"  5. Calibration Curve (probability accuracy)")
print(f"  6. Model Comparison Bar Chart (all metrics)")
print(f"  7. Feature Importance (LightGBM)")
print(f"  8. Class Distribution (actual vs predicted)")

print(f"\nAll visualizations saved to: {FIGURES_DIR}/")
print(f"Reports saved to: {REPORTS_DIR}/")

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
