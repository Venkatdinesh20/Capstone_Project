"""
Missing data analysis script for Credit Risk Prediction System
Detailed analysis of missing patterns across all tables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.data import DataLoader

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def analyze_missing_patterns(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Analyze missing data patterns for a specific table.
    
    Parameters:
    -----------
    df : pd.DataFrame
    table_name : str
    
    Returns:
    --------
    pd.DataFrame : Missing data summary
    """
    missing_summary = pd.DataFrame({
        'table': table_name,
        'column': df.columns,
        'total_rows': len(df),
        'missing_count': df.isnull().sum().values,
        'missing_pct': (df.isnull().sum() / len(df) * 100).values,
        'dtype': df.dtypes.values,
        'unique_values': [df[col].nunique() for col in df.columns]
    })
    
    # Add severity classification
    missing_summary['severity'] = missing_summary['missing_pct'].apply(
        lambda x: 'Critical' if x > 80 else ('High' if x > 50 else ('Moderate' if x > 20 else 'Low'))
    )
    
    return missing_summary[missing_summary['missing_count'] > 0]


def analyze_all_tables(data_type: str = 'train', file_format: str = 'parquet') -> pd.DataFrame:
    """
    Analyze missing data across all tables.
    
    Parameters:
    -----------
    data_type : str
        'train' or 'test'
    file_format : str
        'parquet' or 'csv'
    
    Returns:
    --------
    pd.DataFrame : Comprehensive missing data summary
    """
    logger.info(f"Analyzing missing data for {data_type} dataset...")
    
    loader = DataLoader(data_type=data_type, file_format=file_format)
    table_names = loader.list_available_tables()
    
    all_missing = []
    
    for table_name in table_names:
        try:
            logger.info(f"Processing {table_name}...")
            df = loader.load_table(table_name)
            missing_df = analyze_missing_patterns(df, table_name)
            
            if len(missing_df) > 0:
                all_missing.append(missing_df)
        
        except Exception as e:
            logger.error(f"Error processing {table_name}: {str(e)}")
    
    # Combine all results
    if all_missing:
        full_summary = pd.concat(all_missing, ignore_index=True)
        full_summary = full_summary.sort_values('missing_pct', ascending=False)
        return full_summary
    else:
        return pd.DataFrame()


def plot_missing_heatmap(df: pd.DataFrame, max_cols: int = 50, 
                         figsize: tuple = (15, 10)):
    """
    Create heatmap visualization of missing values.
    
    Parameters:
    -----------
    df : pd.DataFrame
    max_cols : int
        Maximum number of columns to display
    figsize : tuple
        Figure size
    """
    # Select columns with most missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    top_cols = missing_pct[missing_pct > 0].head(max_cols).index.tolist()
    
    if not top_cols:
        print("No missing values to visualize!")
        return
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df[top_cols].isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title(f'Missing Value Patterns (Top {len(top_cols)} columns)', fontsize=14)
    plt.xlabel('Columns', fontsize=12)
    plt.ylabel('Rows', fontsize=12)
    plt.tight_layout()
    
    # Save
    output_path = FIGURES_DIR / 'missing_values_heatmap.png'
    plt.savefig(output_path, dpi=FIGURE_DPI)
    logger.info(f"Heatmap saved to {output_path}")
    plt.show()


def plot_missing_distribution(missing_summary: pd.DataFrame, 
                              top_n: int = 20):
    """
    Plot distribution of missing values.
    
    Parameters:
    -----------
    missing_summary : pd.DataFrame
        Missing data summary
    top_n : int
        Number of top columns to show
    """
    top_missing = missing_summary.nlargest(top_n, 'missing_pct')
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    axes[0].barh(range(len(top_missing)), top_missing['missing_pct'])
    axes[0].set_yticks(range(len(top_missing)))
    axes[0].set_yticklabels([f"{row['table']}.{row['column']}" 
                             for _, row in top_missing.iterrows()], fontsize=8)
    axes[0].set_xlabel('Missing Percentage (%)', fontsize=12)
    axes[0].set_title(f'Top {top_n} Columns by Missing Percentage', fontsize=14)
    axes[0].invert_yaxis()
    
    # Histogram
    axes[1].hist(missing_summary['missing_pct'], bins=20, edgecolor='black')
    axes[1].set_xlabel('Missing Percentage (%)', fontsize=12)
    axes[1].set_ylabel('Number of Columns', fontsize=12)
    axes[1].set_title('Distribution of Missing Percentages', fontsize=14)
    
    plt.tight_layout()
    
    # Save
    output_path = FIGURES_DIR / 'missing_values_distribution.png'
    plt.savefig(output_path, dpi=FIGURE_DPI)
    logger.info(f"Distribution plot saved to {output_path}")
    plt.show()


def generate_missing_report(data_type: str = 'train'):
    """
    Generate comprehensive missing data report.
    
    Parameters:
    -----------
    data_type : str
        'train' or 'test'
    """
    logger.info("=" * 80)
    logger.info(f"MISSING DATA ANALYSIS: {data_type.upper()} DATASET")
    logger.info("=" * 80)
    
    # Analyze
    missing_summary = analyze_all_tables(data_type=data_type)
    
    if len(missing_summary) == 0:
        print("No missing values found in any table!")
        return
    
    # Overall statistics
    print(f"\n### OVERALL STATISTICS ###")
    print(f"Total columns with missing: {len(missing_summary):,}")
    print(f"Tables affected: {missing_summary['table'].nunique()}")
    
    # By severity
    print(f"\n### BY SEVERITY ###")
    severity_counts = missing_summary['severity'].value_counts()
    for severity, count in severity_counts.items():
        print(f"{severity}: {count} columns")
    
    # Critical columns
    critical = missing_summary[missing_summary['severity'] == 'Critical']
    if len(critical) > 0:
        print(f"\n### CRITICAL (>80% Missing) ###")
        print(f"Found {len(critical)} critical columns")
        print(critical[['table', 'column', 'missing_pct']].head(10))
    
    # High columns
    high = missing_summary[missing_summary['severity'] == 'High']
    if len(high) > 0:
        print(f"\n### HIGH (50-80% Missing) ###")
        print(f"Found {len(high)} high-severity columns")
        print(high[['table', 'column', 'missing_pct']].head(10))
    
    # Recommendations
    print(f"\n### RECOMMENDATIONS ###")
    print(f"1. Drop {len(critical)} columns with >80% missing")
    print(f"2. Create missing indicators for {len(high)} high-severity columns")
    print(f"3. Impute remaining columns with appropriate strategies")
    
    # Save report
    output_path = REPORTS_DIR / f'missing_data_report_{data_type}.csv'
    missing_summary.to_csv(output_path, index=False)
    logger.info(f"Report saved to {output_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_missing_distribution(missing_summary)
    
    return missing_summary


if __name__ == "__main__":
    # Generate report for training data
    missing_summary = generate_missing_report(data_type='train')
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
