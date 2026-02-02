"""
Data loading utilities for Credit Risk Prediction System
Handles loading of CSV and Parquet files, memory optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
import logging
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage credit risk dataset files."""
    
    def __init__(self, data_type='train', file_format='parquet'):
        """
        Initialize DataLoader.
        
        Parameters:
        -----------
        data_type : str
            Either 'train' or 'test'
        file_format : str
            Either 'parquet' or 'csv'
        """
        self.data_type = data_type
        self.file_format = file_format
        
        if file_format == 'parquet':
            self.data_dir = PARQUET_TRAIN_DIR if data_type == 'train' else PARQUET_TEST_DIR
        else:
            self.data_dir = TRAIN_DATA_DIR if data_type == 'train' else TEST_DATA_DIR
        
        logger.info(f"DataLoader initialized: {data_type} data, {file_format} format")
    
    
    def load_table(self, table_name: str, optimize_memory: bool = True) -> pd.DataFrame:
        """
        Load a single table.
        
        Parameters:
        -----------
        table_name : str
            Name of the table without prefix (e.g., 'base', 'static_0_0')
        optimize_memory : bool
            Whether to optimize memory usage by downcasting dtypes
        
        Returns:
        --------
        pd.DataFrame
        """
        filename = f"{self.data_type}_{table_name}.{self.file_format}"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading {filename}...")
        
        try:
            if self.file_format == 'parquet':
                df = pd.read_parquet(filepath)
            else:
                df = pd.read_csv(filepath)
            
            if optimize_memory:
                df = self._optimize_dtypes(df)
            
            logger.info(f"Loaded {filename}: {df.shape[0]:,} rows, {df.shape[1]} columns")
            return df
        
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    
    def load_base_table(self) -> pd.DataFrame:
        """
        Load the base table (train_base or test_base).
        
        Returns:
        --------
        pd.DataFrame
        """
        return self.load_table('base')
    
    
    def load_multiple_tables(self, table_names: List[str]) -> dict:
        """
        Load multiple tables and return as dictionary.
        
        Parameters:
        -----------
        table_names : list of str
            Names of tables to load
        
        Returns:
        --------
        dict : {table_name: DataFrame}
        """
        tables = {}
        
        for table_name in tqdm(table_names, desc="Loading tables"):
            try:
                tables[table_name] = self.load_table(table_name)
            except FileNotFoundError:
                logger.warning(f"Table {table_name} not found, skipping...")
                continue
        
        return tables
    
    
    def load_all_static_tables(self) -> dict:
        """
        Load all static tables (1:1 relationship with base).
        
        Returns:
        --------
        dict : {table_name: DataFrame}
        """
        logger.info("Loading all static tables...")
        return self.load_multiple_tables(STATIC_TABLES)
    
    
    def load_all_dynamic_tables(self) -> dict:
        """
        Load all dynamic tables (1:N relationship with base).
        
        Returns:
        --------
        dict : {table_name: DataFrame}
        """
        logger.info("Loading all dynamic tables...")
        return self.load_multiple_tables(DYNAMIC_TABLES)
    
    
    def list_available_tables(self) -> List[str]:
        """
        List all available table files in the data directory.
        
        Returns:
        --------
        list of str : Table names (without prefix and extension)
        """
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return []
        
        pattern = f"{self.data_type}_*.{self.file_format}"
        files = list(self.data_dir.glob(pattern))
        
        # Extract table names
        table_names = []
        for file in files:
            # Remove prefix and extension
            name = file.stem.replace(f"{self.data_type}_", "")
            table_names.append(name)
        
        logger.info(f"Found {len(table_names)} tables")
        return sorted(table_names)
    
    
    def get_table_info(self, table_name: str) -> dict:
        """
        Get metadata about a table without fully loading it.
        
        Parameters:
        -----------
        table_name : str
            Name of the table
        
        Returns:
        --------
        dict : Table metadata
        """
        df = self.load_table(table_name)
        
        info = {
            'name': table_name,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        
        return info
    
    
    @staticmethod
    def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting dtypes.
        
        Parameters:
        -----------
        df : pd.DataFrame
        
        Returns:
        --------
        pd.DataFrame : Optimized DataFrame
        """
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize integers
        int_cols = df.select_dtypes(include=['int']).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Optimize floats
        float_cols = df.select_dtypes(include=['float']).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns with few unique values to category
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (initial_memory - final_memory) / initial_memory * 100
        
        logger.info(f"Memory optimized: {initial_memory:.2f}MB → {final_memory:.2f}MB "
                   f"({reduction:.1f}% reduction)")
        
        return df


def load_sample_data(n_rows: int = 1000, data_type: str = 'train') -> pd.DataFrame:
    """
    Load a sample of the base table for quick testing.
    
    Parameters:
    -----------
    n_rows : int
        Number of rows to load
    data_type : str
        Either 'train' or 'test'
    
    Returns:
    --------
    pd.DataFrame
    """
    loader = DataLoader(data_type=data_type)
    df = loader.load_base_table()
    return df.head(n_rows)


def get_dataset_summary(data_type: str = 'train') -> pd.DataFrame:
    """
    Get a summary of all tables in the dataset.
    
    Parameters:
    -----------
    data_type : str
        Either 'train' or 'test'
    
    Returns:
    --------
    pd.DataFrame : Summary with table names, rows, columns, memory usage
    """
    loader = DataLoader(data_type=data_type)
    table_names = loader.list_available_tables()
    
    summaries = []
    for table_name in tqdm(table_names, desc="Analyzing tables"):
        try:
            info = loader.get_table_info(table_name)
            summaries.append({
                'table': table_name,
                'rows': info['rows'],
                'columns': info['columns'],
                'memory_mb': round(info['memory_mb'], 2)
            })
        except Exception as e:
            logger.error(f"Error analyzing {table_name}: {str(e)}")
    
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values('memory_mb', ascending=False)
    
    logger.info(f"\nDataset Summary:")
    logger.info(f"Total tables: {len(summary_df)}")
    logger.info(f"Total memory: {summary_df['memory_mb'].sum():.2f} MB")
    
    return summary_df


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("DATA LOADER - EXAMPLE USAGE")
    print("=" * 80)
    
    # Initialize loader
    loader = DataLoader(data_type='train', file_format='parquet')
    
    # List available tables
    print("\nAvailable tables:")
    tables = loader.list_available_tables()
    for i, table in enumerate(tables[:10], 1):
        print(f"  {i}. {table}")
    print(f"  ... and {len(tables) - 10} more")
    
    # Load base table
    print("\nLoading base table...")
    base_df = loader.load_base_table()
    print(f"Shape: {base_df.shape}")
    print(f"Columns: {list(base_df.columns[:5])}...")
    
    # Get dataset summary
    print("\nGenerating dataset summary...")
    summary = get_dataset_summary(data_type='train')
    print(summary.head(10))
