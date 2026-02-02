"""
Visualization utilities for Credit Risk Prediction System
Common plotting functions for data exploration and model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def plot_target_distribution(df: pd.DataFrame, target_col: str = TARGET_COL,
                             figsize: tuple = (10, 6), save: bool = True):
    """
    Plot target variable distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
    target_col : str
    figsize : tuple
    save : bool
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Count plot
    target_counts = df[target_col].value_counts()
    axes[0].bar(target_counts.index, target_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0].set_xlabel(target_col.capitalize(), fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Target Distribution (Counts)', fontsize=14)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['No Default (0)', 'Default (1)'])
    
    # Add value labels
    for i, v in enumerate(target_counts.values):
        axes[0].text(i, v, f'{v:,}', ha='center', va='bottom')
    
    # Percentage plot
    target_pct = df[target_col].value_counts(normalize=True) * 100
    axes[1].pie(target_pct.values, labels=['No Default', 'Default'], 
               autopct='%1.2f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[1].set_title('Target Distribution (Percentage)', fontsize=14)
    
    plt.tight_layout()
    
    if save:
        output_path = FIGURES_DIR / 'target_distribution.png'
        plt.savefig(output_path, dpi=FIGURE_DPI)
        logger.info(f"Plot saved to {output_path}")
    
    plt.show()


def plot_missing_values(df: pd.DataFrame, top_n: int = 20, 
                        figsize: tuple = (12, 8), save: bool = True):
    """
    Plot missing values summary.
    
    Parameters:
    -----------
    df : pd.DataFrame
    top_n : int
    figsize : tuple
    save : bool
    """
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0].head(top_n)
    
    if len(missing_pct) == 0:
        print("No missing values to plot!")
        return
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(missing_pct)), missing_pct.values)
    plt.yticks(range(len(missing_pct)), missing_pct.index, fontsize=10)
    plt.xlabel('Missing Percentage (%)', fontsize=12)
    plt.title(f'Top {top_n} Columns by Missing Values', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Add percentage labels
    for i, v in enumerate(missing_pct.values):
        plt.text(v, i, f' {v:.1f}%', va='center')
    
    plt.tight_layout()
    
    if save:
        output_path = FIGURES_DIR / 'missing_values_summary.png'
        plt.savefig(output_path, dpi=FIGURE_DPI)
        logger.info(f"Plot saved to {output_path}")
    
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, target_col: str = TARGET_COL,
                             top_n: int = 20, figsize: tuple = (12, 10), 
                             save: bool = True):
    """
    Plot correlation heatmap with target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
    target_col : str
    top_n : int
    figsize : tuple
    save : bool
    """
    # Get numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col not in num_cols:
        print(f"Target column {target_col} not found!")
        return
    
    # Calculate correlations with target
    correlations = df[num_cols].corr()[target_col].abs().sort_values(ascending=False)
    top_features = correlations.head(top_n + 1).index.tolist()  # +1 for target itself
    
    # Create correlation matrix for top features
    corr_matrix = df[top_features].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title(f'Top {top_n} Features - Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    
    if save:
        output_path = FIGURES_DIR / 'correlation_heatmap.png'
        plt.savefig(output_path, dpi=FIGURE_DPI)
        logger.info(f"Plot saved to {output_path}")
    
    plt.show()


def plot_feature_distributions(df: pd.DataFrame, features: List[str], 
                               target_col: str = TARGET_COL, 
                               ncols: int = 3, figsize_per_plot: tuple = (5, 4),
                               save: bool = True):
    """
    Plot distributions of features by target class.
    
    Parameters:
    -----------
    df : pd.DataFrame
    features : list
    target_col : str
    ncols : int
    figsize_per_plot : tuple
    save : bool
    """
    nrows = int(np.ceil(len(features) / ncols))
    figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 else [axes]
    
    for idx, feature in enumerate(features):
        if idx >= len(axes):
            break
        
        # Plot distributions for each target class
        for target_val in df[target_col].unique():
            data = df[df[target_col] == target_val][feature].dropna()
            axes[idx].hist(data, alpha=0.6, label=f'Target={target_val}', bins=30)
        
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].legend()
        axes[idx].set_title(f'{feature} Distribution', fontsize=11)
    
    # Hide extra subplots
    for idx in range(len(features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save:
        output_path = FIGURES_DIR / 'feature_distributions.png'
        plt.savefig(output_path, dpi=FIGURE_DPI)
        logger.info(f"Plot saved to {output_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=['No Default', 'Default'],
                          figsize: tuple = (8, 6), save: bool = True):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
    y_pred : array-like
    labels : list
    figsize : tuple
    save : bool
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    
    if save:
        output_path = FIGURES_DIR / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=FIGURE_DPI)
        logger.info(f"Plot saved to {output_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, figsize: tuple = (8, 6), 
                   save: bool = True):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
    y_pred_proba : array-like
    figsize : tuple
    save : bool
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save:
        output_path = FIGURES_DIR / 'roc_curve.png'
        plt.savefig(output_path, dpi=FIGURE_DPI)
        logger.info(f"Plot saved to {output_path}")
    
    plt.show()


def plot_feature_importance(importances, feature_names, top_n: int = 20,
                            figsize: tuple = (10, 8), save: bool = True):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importances : array-like
    feature_names : list
    top_n : int
    figsize : tuple
    save : bool
    """
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'], fontsize=10)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(importance_df['importance']):
        plt.text(v, i, f' {v:.4f}', va='center')
    
    plt.tight_layout()
    
    if save:
        output_path = FIGURES_DIR / 'feature_importance.png'
        plt.savefig(output_path, dpi=FIGURE_DPI)
        logger.info(f"Plot saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities loaded successfully!")
    print("Available functions:")
    print("  - plot_target_distribution()")
    print("  - plot_missing_values()")
    print("  - plot_correlation_heatmap()")
    print("  - plot_feature_distributions()")
    print("  - plot_confusion_matrix()")
    print("  - plot_roc_curve()")
    print("  - plot_feature_importance()")
