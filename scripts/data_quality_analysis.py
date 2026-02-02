"""
Data quality analysis script for Credit Risk Prediction System
Generates comprehensive data quality report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.data import DataLoader, DataMerger

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """Analyze and report on data quality."""
    
    def __init__(self, df: pd.DataFrame, name: str = 'dataset'):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        df : pd.DataFrame
        name : str
            Name of the dataset
        """
        self.df = df
        self.name = name
        self.report = {}
        logger.info(f"DataQualityAnalyzer initialized for {name}")
    
    
    def basic_info(self) -> dict:
        """Get basic dataset information."""
        info = {
            'name': self.name,
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'duplicates': self.df.duplicated().sum()
        }
        
        self.report['basic_info'] = info
        return info
    
    
    def missing_value_analysis(self) -> pd.DataFrame:
        """Analyze missing values."""
        missing_df = pd.DataFrame({
            'column': self.df.columns,
            'missing_count': self.df.isnull().sum().values,
            'missing_pct': (self.df.isnull().sum() / len(self.df) * 100).values,
            'dtype': self.df.dtypes.values
        })
        
        missing_df = missing_df[missing_df['missing_count'] > 0]
        missing_df = missing_df.sort_values('missing_pct', ascending=False)
        
        self.report['missing_values'] = missing_df
        return missing_df
    
    
    def data_types_analysis(self) -> dict:
        """Analyze data types distribution."""
        dtype_counts = self.df.dtypes.value_counts().to_dict()
        
        type_info = {
            'numerical': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical': len(self.df.select_dtypes(include=['object', 'category']).columns),
            'datetime': len(self.df.select_dtypes(include=['datetime']).columns),
            'dtype_breakdown': dtype_counts
        }
        
        self.report['data_types'] = type_info
        return type_info
    
    
    def numerical_summary(self) -> pd.DataFrame:
        """Get summary statistics for numerical columns."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) == 0:
            return pd.DataFrame()
        
        summary = self.df[num_cols].describe().T
        summary['missing_pct'] = (self.df[num_cols].isnull().sum() / len(self.df) * 100).values
        summary['zeros_pct'] = ((self.df[num_cols] == 0).sum() / len(self.df) * 100).values
        summary['unique_count'] = self.df[num_cols].nunique().values
        
        self.report['numerical_summary'] = summary
        return summary
    
    
    def categorical_summary(self) -> pd.DataFrame:
        """Get summary for categorical columns."""
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols) == 0:
            return pd.DataFrame()
        
        cat_info = []
        for col in cat_cols:
            cat_info.append({
                'column': col,
                'unique_count': self.df[col].nunique(),
                'missing_count': self.df[col].isnull().sum(),
                'missing_pct': self.df[col].isnull().sum() / len(self.df) * 100,
                'mode': self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else None,
                'mode_frequency': self.df[col].value_counts().iloc[0] if len(self.df[col]) > 0 else 0
            })
        
        cat_df = pd.DataFrame(cat_info)
        self.report['categorical_summary'] = cat_df
        return cat_df
    
    
    def target_analysis(self, target_col: str = TARGET_COL) -> dict:
        """Analyze target variable."""
        if target_col not in self.df.columns:
            logger.warning(f"Target column {target_col} not found")
            return {}
        
        target_dist = self.df[target_col].value_counts(normalize=True).to_dict()
        target_counts = self.df[target_col].value_counts().to_dict()
        
        target_info = {
            'distribution_pct': target_dist,
            'counts': target_counts,
            'missing': self.df[target_col].isnull().sum(),
            'imbalance_ratio': max(target_counts.values()) / min(target_counts.values()) if len(target_counts) > 1 else None
        }
        
        self.report['target_analysis'] = target_info
        return target_info
    
    
    def high_cardinality_features(self, threshold: int = HIGH_CARDINALITY_THRESHOLD) -> list:
        """Identify high cardinality categorical features."""
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        high_card = []
        for col in cat_cols:
            unique_count = self.df[col].nunique()
            if unique_count > threshold:
                high_card.append({
                    'column': col,
                    'unique_count': unique_count,
                    'unique_pct': unique_count / len(self.df) * 100
                })
        
        self.report['high_cardinality'] = high_card
        return high_card
    
    
    def correlation_analysis(self, top_n: int = 20) -> pd.DataFrame:
        """Calculate correlations with target."""
        if TARGET_COL not in self.df.columns:
            logger.warning(f"Target column {TARGET_COL} not found")
            return pd.DataFrame()
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        num_cols = [col for col in num_cols if col != TARGET_COL]
        
        correlations = []
        for col in num_cols:
            corr = self.df[col].corr(self.df[TARGET_COL])
            if not np.isnan(corr):
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('abs_correlation', ascending=False).head(top_n)
        
        self.report['correlations'] = corr_df
        return corr_df
    
    
    def generate_full_report(self) -> dict:
        """Generate complete data quality report."""
        logger.info("=" * 80)
        logger.info(f"GENERATING DATA QUALITY REPORT: {self.name}")
        logger.info("=" * 80)
        
        # Run all analyses
        self.basic_info()
        self.missing_value_analysis()
        self.data_types_analysis()
        self.numerical_summary()
        self.categorical_summary()
        self.target_analysis()
        self.high_cardinality_features()
        self.correlation_analysis()
        
        logger.info("✓ Data quality report generated")
        return self.report
    
    
    def print_report(self):
        """Print formatted report."""
        if not self.report:
            self.generate_full_report()
        
        print("=" * 80)
        print(f"DATA QUALITY REPORT: {self.name}")
        print("=" * 80)
        
        # Basic info
        if 'basic_info' in self.report:
            info = self.report['basic_info']
            print("\n### BASIC INFORMATION ###")
            print(f"Rows: {info['rows']:,}")
            print(f"Columns: {info['columns']:,}")
            print(f"Memory: {info['memory_mb']:.2f} MB")
            print(f"Duplicates: {info['duplicates']:,}")
        
        # Data types
        if 'data_types' in self.report:
            types = self.report['data_types']
            print("\n### DATA TYPES ###")
            print(f"Numerical: {types['numerical']}")
            print(f"Categorical: {types['categorical']}")
            print(f"Datetime: {types['datetime']}")
        
        # Missing values
        if 'missing_values' in self.report:
            missing = self.report['missing_values']
            print(f"\n### MISSING VALUES ###")
            if len(missing) > 0:
                print(f"Columns with missing values: {len(missing)}")
                print(f"\nTop 10 columns with most missing:")
                print(missing.head(10).to_string())
            else:
                print("No missing values found!")
        
        # Target analysis
        if 'target_analysis' in self.report:
            target = self.report['target_analysis']
            print(f"\n### TARGET VARIABLE ANALYSIS ###")
            print(f"Distribution: {target['distribution_pct']}")
            print(f"Counts: {target['counts']}")
            if target['imbalance_ratio']:
                print(f"Imbalance Ratio: {target['imbalance_ratio']:.2f}:1")
        
        # Correlations
        if 'correlations' in self.report and len(self.report['correlations']) > 0:
            print(f"\n### TOP 10 CORRELATIONS WITH TARGET ###")
            print(self.report['correlations'].head(10).to_string())
        
        print("\n" + "=" * 80)
    
    
    def save_report(self, output_path: Path = None):
        """Save report to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = REPORTS_DIR / f"data_quality_report_{self.name}_{timestamp}.txt"
        
        with open(output_path, 'w') as f:
            f.write(f"DATA QUALITY REPORT: {self.name}\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
            
            for key, value in self.report.items():
                f.write(f"\n### {key.upper().replace('_', ' ')} ###\n")
                if isinstance(value, pd.DataFrame):
                    f.write(value.to_string())
                else:
                    f.write(str(value))
                f.write("\n\n")
        
        logger.info(f"Report saved to: {output_path}")


def analyze_dataset(data_type: str = 'train', file_format: str = 'parquet'):
    """
    Analyze a complete dataset and generate report.
    
    Parameters:
    -----------
    data_type : str
        'train' or 'test'
    file_format : str
        'parquet' or 'csv'
    """
    # Load data
    loader = DataLoader(data_type=data_type, file_format=file_format)
    base_df = loader.load_base_table()
    
    # Analyze
    analyzer = DataQualityAnalyzer(base_df, name=f'{data_type}_base')
    analyzer.generate_full_report()
    analyzer.print_report()
    analyzer.save_report()
    
    return analyzer


if __name__ == "__main__":
    # Analyze training data
    analyzer = analyze_dataset(data_type='train', file_format='parquet')
