"""
Setup Verification Script
Run this to verify your environment is correctly configured
"""

import sys
from pathlib import Path
import importlib

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    """Print section header."""
    print(f"\n{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BLUE}{text.center(80)}{Colors.END}")
    print(f"{Colors.BLUE}{'='*80}{Colors.END}\n")

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"{Colors.GREEN}✓ Python {version.major}.{version.minor}.{version.micro} (OK){Colors.END}")
        return True
    else:
        print(f"{Colors.RED}✗ Python {version.major}.{version.minor} (Need 3.8+){Colors.END}")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking required packages...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'lightgbm': 'lightgbm',
        'xgboost': 'xgboost',
        'catboost': 'catboost',
        'shap': 'shap',
        'imblearn': 'imbalanced-learn',
        'pyarrow': 'pyarrow'
    }
    
    results = []
    for module_name, package_name in required_packages.items():
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"{Colors.GREEN}✓ {package_name:20s} {version}{Colors.END}")
            results.append(True)
        except ImportError:
            print(f"{Colors.RED}✗ {package_name:20s} NOT INSTALLED{Colors.END}")
            results.append(False)
    
    return all(results)

def check_project_structure():
    """Check if project directories exist."""
    print("\nChecking project structure...")
    
    root_dir = Path(__file__).parent
    
    required_dirs = [
        'src',
        'src/data',
        'src/features',
        'src/models',
        'src/visualization',
        'notebooks',
        'scripts',
        'models',
        'outputs',
        'outputs/figures',
        'outputs/reports',
        'outputs/predictions',
        'docs'
    ]
    
    results = []
    for dir_name in required_dirs:
        dir_path = root_dir / dir_name
        if dir_path.exists():
            print(f"{Colors.GREEN}✓ {dir_name}{Colors.END}")
            results.append(True)
        else:
            print(f"{Colors.RED}✗ {dir_name} (missing){Colors.END}")
            results.append(False)
    
    return all(results)

def check_config():
    """Check if config file is accessible."""
    print("\nChecking configuration...")
    
    try:
        import config
        print(f"{Colors.GREEN}✓ config.py loaded{Colors.END}")
        print(f"  Root Directory: {config.ROOT_DIR}")
        print(f"  Data Directory: {config.DATA_DIR}")
        print(f"  Random State: {config.RANDOM_STATE}")
        return True
    except Exception as e:
        print(f"{Colors.RED}✗ Error loading config: {str(e)}{Colors.END}")
        return False

def check_data_directories():
    """Check if data directories exist."""
    print("\nChecking data directories...")
    
    import config
    
    data_dirs = [
        ('CSV Train', config.TRAIN_DATA_DIR),
        ('CSV Test', config.TEST_DATA_DIR),
        ('Parquet Train', config.PARQUET_TRAIN_DIR),
        ('Parquet Test', config.PARQUET_TEST_DIR)
    ]
    
    found_data = False
    for name, path in data_dirs:
        if path.exists():
            file_count = len(list(path.glob('*.*')))
            if file_count > 0:
                print(f"{Colors.GREEN}✓ {name:20s} ({file_count} files){Colors.END}")
                found_data = True
            else:
                print(f"{Colors.YELLOW}! {name:20s} (empty){Colors.END}")
        else:
            print(f"{Colors.YELLOW}! {name:20s} (not found){Colors.END}")
    
    if not found_data:
        print(f"\n{Colors.YELLOW}NOTE: No data files found. Download from Kaggle.{Colors.END}")
    
    return found_data

def check_source_modules():
    """Check if source modules can be imported."""
    print("\nChecking source modules...")
    
    modules = [
        'src.data.loader',
        'src.data.merger',
        'src.data.preprocessor',
        'src.visualization.plots'
    ]
    
    results = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"{Colors.GREEN}✓ {module_name}{Colors.END}")
            results.append(True)
        except Exception as e:
            print(f"{Colors.RED}✗ {module_name}: {str(e)}{Colors.END}")
            results.append(False)
    
    return all(results)

def run_quick_test():
    """Run a quick functionality test."""
    print("\nRunning quick functionality test...")
    
    try:
        # Test config
        from config import ROOT_DIR, TARGET_COL, ID_COL
        print(f"{Colors.GREEN}✓ Config import{Colors.END}")
        
        # Test data loader
        from src.data import DataLoader
        print(f"{Colors.GREEN}✓ DataLoader import{Colors.END}")
        
        # Test preprocessor
        from src.data import DataPreprocessor
        print(f"{Colors.GREEN}✓ DataPreprocessor import{Colors.END}")
        
        # Test visualization
        from src.visualization import plot_target_distribution
        print(f"{Colors.GREEN}✓ Visualization import{Colors.END}")
        
        return True
    except Exception as e:
        print(f"{Colors.RED}✗ Test failed: {str(e)}{Colors.END}")
        return False

def main():
    """Run all verification checks."""
    print_header("CREDIT RISK PREDICTION SYSTEM - SETUP VERIFICATION")
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Project Structure': check_project_structure(),
        'Configuration': check_config(),
        'Data Files': check_data_directories(),
        'Source Modules': check_source_modules(),
        'Functionality Test': run_quick_test()
    }
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    all_passed = True
    for check, passed in results.items():
        if passed:
            print(f"{Colors.GREEN}✓ {check:25s} PASSED{Colors.END}")
        else:
            print(f"{Colors.RED}✗ {check:25s} FAILED{Colors.END}")
            all_passed = False
    
    print()
    
    if all_passed:
        print(f"{Colors.GREEN}{'='*80}{Colors.END}")
        print(f"{Colors.GREEN}🎉 ALL CHECKS PASSED! Environment is ready! 🎉{Colors.END}")
        print(f"{Colors.GREEN}{'='*80}{Colors.END}")
        print("\nNext steps:")
        print("1. Download data from Kaggle (if not done)")
        print("2. Run: jupyter notebook")
        print("3. Open: notebooks/01_data_exploration.ipynb")
        print("4. Or run: python scripts/data_quality_analysis.py")
    else:
        print(f"{Colors.RED}{'='*80}{Colors.END}")
        print(f"{Colors.RED}⚠️  SOME CHECKS FAILED - See above for details{Colors.END}")
        print(f"{Colors.RED}{'='*80}{Colors.END}")
        print("\nTo fix:")
        print("1. pip install -r requirements.txt")
        print("2. Download data from Kaggle")
        print("3. Check file paths in config.py")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
