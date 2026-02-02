"""
Data merging utilities for Credit Risk Prediction System
Handles merging of multiple tables into a single dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *
from src.data.loader import DataLoader

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataMerger:
    """Merge multiple tables into a single dataset."""
    
    def __init__(self, data_type='train', file_format='parquet'):
        """
        Initialize DataMerger.
        
        Parameters:
        -----------
        data_type : str
            Either 'train' or 'test'
        file_format : str
            Either 'parquet' or 'csv'
        """
        self.data_type = data_type
        self.loader = DataLoader(data_type=data_type, file_format=file_format)
        self.base_df = None
        logger.info(f"DataMerger initialized for {data_type} data")
    
    
    def load_base(self) -> pd.DataFrame:
        """Load the base table."""
        self.base_df = self.loader.load_base_table()
        logger.info(f"Base table loaded: {self.base_df.shape}")
        return self.base_df
    
    
    def merge_static_table(self, table_name: str, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge a static table (1:1 relationship) with base.
        
        Parameters:
        -----------
        table_name : str
            Name of the static table
        base_df : pd.DataFrame
            Base DataFrame to merge with
        
        Returns:
        --------
        pd.DataFrame : Merged DataFrame
        """
        try:
            static_df = self.loader.load_table(table_name)
            
            # Check for duplicates in join key
            if static_df[ID_COL].duplicated().any():
                logger.warning(f"{table_name} has duplicate {ID_COL}, keeping first occurrence")
                static_df = static_df.drop_duplicates(subset=[ID_COL], keep='first')
            
            # Merge
            merged_df = base_df.merge(
                static_df,
                on=ID_COL,
                how='left',
                suffixes=('', f'_{table_name}')
            )
            
            logger.info(f"Merged {table_name}: {base_df.shape} → {merged_df.shape}")
            return merged_df
        
        except FileNotFoundError:
            logger.warning(f"Table {table_name} not found, skipping...")
            return base_df
        except Exception as e:
            logger.error(f"Error merging {table_name}: {str(e)}")
            return base_df
    
    
    def aggregate_dynamic_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Aggregate a dynamic table (1:N relationship) by case_id.
        
        Parameters:
        -----------
        table_name : str
            Name of the dynamic table
        
        Returns:
        --------
        pd.DataFrame : Aggregated DataFrame
        """
        try:
            dynamic_df = self.loader.load_table(table_name)
            
            # Separate numerical and categorical columns
            num_cols = dynamic_df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = dynamic_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Remove ID column from aggregation
            if ID_COL in num_cols:
                num_cols.remove(ID_COL)
            if ID_COL in cat_cols:
                cat_cols.remove(ID_COL)
            
            # Build aggregation dictionary
            agg_dict = {}
            
            # Numerical aggregations
            for col in num_cols:
                agg_dict[col] = ['mean', 'median', 'std', 'min', 'max', 'sum', 'count']
            
            # Categorical aggregations
            for col in cat_cols:
                agg_dict[col] = ['count', 'nunique']
            
            # Perform aggregation
            agg_df = dynamic_df.groupby(ID_COL).agg(agg_dict)
            
            # Flatten multi-level column names
            agg_df.columns = [f"{table_name}_{col}_{stat}" for col, stat in agg_df.columns]
            agg_df = agg_df.reset_index()
            
            logger.info(f"Aggregated {table_name}: {len(dynamic_df)} → {len(agg_df)} rows, "
                       f"{len(agg_df.columns)-1} features")
            
            return agg_df
        
        except FileNotFoundError:
            logger.warning(f"Table {table_name} not found, skipping...")
            return None
        except Exception as e:
            logger.error(f"Error aggregating {table_name}: {str(e)}")
            return None
    
    
    def merge_all_static_tables(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge all static tables with base.
        
        Parameters:
        -----------
        base_df : pd.DataFrame
            Base DataFrame
        
        Returns:
        --------
        pd.DataFrame : DataFrame with all static tables merged
        """
        logger.info("Merging all static tables...")
        merged_df = base_df.copy()
        
        for table_name in tqdm(STATIC_TABLES, desc="Merging static tables"):
            merged_df = self.merge_static_table(table_name, merged_df)
        
        logger.info(f"All static tables merged: {base_df.shape} → {merged_df.shape}")
        return merged_df
    
    
    def merge_all_dynamic_tables(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate and merge all dynamic tables with base.
        
        Parameters:
        -----------
        base_df : pd.DataFrame
            Base DataFrame
        
        Returns:
        --------
        pd.DataFrame : DataFrame with all dynamic tables merged
        """
        logger.info("Aggregating and merging all dynamic tables...")
        merged_df = base_df.copy()
        
        for table_name in tqdm(DYNAMIC_TABLES, desc="Processing dynamic tables"):
            # Aggregate
            agg_df = self.aggregate_dynamic_table(table_name)
            
            if agg_df is not None:
                # Merge
                merged_df = merged_df.merge(
                    agg_df,
                    on=ID_COL,
                    how='left'
                )
        
        logger.info(f"All dynamic tables merged: {base_df.shape} → {merged_df.shape}")
        return merged_df
    
    
    def merge_all_tables(self, include_static: bool = True, 
                        include_dynamic: bool = True) -> pd.DataFrame:
        """
        Merge all tables into a single dataset.
        
        Parameters:
        -----------
        include_static : bool
            Whether to include static tables
        include_dynamic : bool
            Whether to include dynamic tables
        
        Returns:
        --------
        pd.DataFrame : Complete merged dataset
        """
        logger.info("=" * 80)
        logger.info("STARTING FULL DATA MERGE")
        logger.info("=" * 80)
        
        # Load base
        if self.base_df is None:
            self.load_base()
        
        merged_df = self.base_df.copy()
        initial_shape = merged_df.shape
        
        # Merge static tables
        if include_static:
            merged_df = self.merge_all_static_tables(merged_df)
        
        # Merge dynamic tables
        if include_dynamic:
            merged_df = self.merge_all_dynamic_tables(merged_df)
        
        # Validation
        self._validate_merge(self.base_df, merged_df)
        
        logger.info("=" * 80)
        logger.info("MERGE COMPLETE")
        logger.info(f"Initial shape: {initial_shape}")
        logger.info(f"Final shape: {merged_df.shape}")
        logger.info(f"Features added: {merged_df.shape[1] - initial_shape[1]}")
        logger.info("=" * 80)
        
        return merged_df
    
    
    @staticmethod
    def _validate_merge(base_df: pd.DataFrame, merged_df: pd.DataFrame):
        """
        Validate the merged dataset.
        
        Parameters:
        -----------
        base_df : pd.DataFrame
            Original base DataFrame
        merged_df : pd.DataFrame
            Merged DataFrame
        """
        # Check no rows lost
        assert len(base_df) == len(merged_df), \
            f"Row count mismatch: {len(base_df)} → {len(merged_df)}"
        
        # Check no duplicate case_ids
        assert not merged_df[ID_COL].duplicated().any(), \
            "Duplicate case_ids found in merged data"
        
        # Check case_ids match
        assert set(base_df[ID_COL]) == set(merged_df[ID_COL]), \
            "case_id mismatch between base and merged data"
        
        logger.info("✓ Merge validation passed")


def quick_merge_test(n_rows: int = 1000):
    """
    Quick test of merge functionality on a sample.
    
    Parameters:
    -----------
    n_rows : int
        Number of rows to test with
    """
    print("=" * 80)
    print("QUICK MERGE TEST")
    print("=" * 80)
    
    merger = DataMerger(data_type='train')
    base_df = merger.load_base().head(n_rows)
    
    print(f"\nBase shape: {base_df.shape}")
    
    # Test static merge
    print("\nTesting static table merge...")
    merged_static = merger.merge_static_table('static_0_0', base_df)
    print(f"After static merge: {merged_static.shape}")
    
    # Test dynamic aggregation
    print("\nTesting dynamic table aggregation...")
    agg_df = merger.aggregate_dynamic_table('applprev_1_0')
    if agg_df is not None:
        print(f"Aggregated shape: {agg_df.shape}")
        merged_full = merged_static.merge(agg_df, on=ID_COL, how='left')
        print(f"After dynamic merge: {merged_full.shape}")


if __name__ == "__main__":
    # Quick test
    quick_merge_test(n_rows=1000)
    
    # Full merge example (commented out to avoid long runtime)
    # merger = DataMerger(data_type='train')
    # full_df = merger.merge_all_tables()
    # print(full_df.head())
