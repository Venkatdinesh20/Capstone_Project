"""
Monitor the data merging progress in real-time
"""
import time
import psutil
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *

def get_folder_size(folder):
    """Calculate folder size in MB"""
    total = 0
    try:
        for entry in os.scandir(folder):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_folder_size(entry.path)
    except:
        pass
    return total / (1024 * 1024)

def monitor_progress():
    """Monitor merging progress"""
    print("=" * 80)
    print("DATA MERGING MONITOR")
    print("=" * 80)
    print("\nPress Ctrl+C to stop monitoring\n")
    
    iteration = 0
    while True:
        try:
            iteration += 1
            print(f"\n[Update #{iteration}] {time.strftime('%H:%M:%S')}")
            print("-" * 80)
            
            # Check if output file exists
            output_file = DATA_PROCESSED_DIR / 'step2_data_merged.parquet'
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"✓ Output file exists: {size_mb:.2f} MB")
                print("✓ MERGING COMPLETE!")
                break
            else:
                print(f"⏳ Output file not yet created...")
            
            # Check data_processed folder
            if DATA_PROCESSED_DIR.exists():
                folder_size = get_folder_size(DATA_PROCESSED_DIR)
                print(f"📁 data_processed/ size: {folder_size:.2f} MB")
            
            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            print(f"💻 CPU Usage: {cpu_percent:.1f}%")
            print(f"💾 Memory: {memory.percent:.1f}% used ({memory.used/(1024**3):.1f}GB / {memory.total/(1024**3):.1f}GB)")
            
            # Check Python processes
            python_procs = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    if 'python' in proc.info['name'].lower():
                        mem_mb = proc.info['memory_info'].rss / (1024 * 1024)
                        if mem_mb > 50:  # Only show processes using >50MB
                            python_procs.append((proc.info['pid'], mem_mb))
                except:
                    pass
            
            if python_procs:
                print(f"\n🐍 Python processes:")
                for pid, mem in python_procs:
                    print(f"   PID {pid}: {mem:.0f} MB")
            
            # Estimated progress (rough guess based on typical merging time)
            elapsed = iteration * 10  # 10 seconds per iteration
            if elapsed < 300:  # First 5 minutes
                progress = elapsed / 300 * 30  # 30% for static tables
            elif elapsed < 600:  # 5-10 minutes
                progress = 30 + (elapsed - 300) / 300 * 70  # Remaining 70% for dynamic
            else:
                progress = min(95, 30 + (elapsed - 300) / 600 * 70)
            
            print(f"\n📊 Estimated progress: {progress:.0f}%")
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"   [{bar}]")
            
            print("\n⏳ Waiting 10 seconds...")
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n\n✋ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            time.sleep(10)
    
    print("\n" + "=" * 80)
    print("MONITORING COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    monitor_progress()
