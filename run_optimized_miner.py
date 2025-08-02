#!/usr/bin/env python3
"""
Launcher script for OptimizedMiner with configuration options
Usage: python run_optimized_miner.py [--config hardware_tier] [--model_preference speed|balanced|quality]
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from optimized_miner_config import OptimizedMinerConfig, get_hardware_config
from neurons.optimized_miner import OptimizedMiner
import bittensor as bt
from BetterTherapy.utils.config import add_miner_args


def setup_environment():
    """Setup environment variables and paths"""
    # Ensure CUDA is available if requested
    import torch
    if OptimizedMinerConfig.USE_CUDA and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        OptimizedMinerConfig.USE_CUDA = False
    
    # Set environment variables for optimization
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Cleaner output


def print_system_info():
    """Print system information for debugging"""
    import torch
    import os
    import platform
    
    print("🖥️  System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {platform.python_version()}")
    
    try:
        # Cố gắng lấy số CPU cores
        cpu_count = os.cpu_count() or "Unknown"
        print(f"   CPU: {cpu_count} cores")
    except:
        print("   CPU: Unknown")
    
    if torch.cuda.is_available():
        try:
            print(f"   GPU: {torch.cuda.get_device_name()}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(f"   VRAM: {vram_gb} GB")
        except:
            print("   GPU: CUDA available but info unavailable")
    else:
        print("   GPU: Not available")
    print()


def apply_hardware_config(tier):
    """Apply hardware-specific configuration"""
    config = get_hardware_config(tier)
    
    OptimizedMinerConfig.MODEL_PREFERENCE = config['model_preference']
    OptimizedMinerConfig.USE_QUANTIZATION = config['quantization']
    OptimizedMinerConfig.CACHE_MAX_SIZE = config['cache_size']
    OptimizedMinerConfig.MAX_WORKERS = config['max_workers']
    
    print(f"✅ Applied {tier} hardware configuration")
    print(f"   Model: {config['model_preference']}")
    print(f"   Quantization: {config['quantization']}")
    print(f"   Cache Size: {config['cache_size']}")
    print(f"   Workers: {config['max_workers']}")
    print()


def create_bt_config():
    """Create Bittensor config with optimized miner args"""
    parser = argparse.ArgumentParser(description='Run OptimizedMiner')
    
    # Add standard Bittensor arguments
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    
    # Add miner specific arguments
    add_miner_args(None, parser)
    
    # Add optimized miner specific arguments  
    parser.add_argument('--hardware_tier', choices=['low_end', 'mid_range', 'high_end'], 
                       default='mid_range', help='Hardware configuration tier')
    parser.add_argument('--model_mode', choices=['speed', 'balanced', 'quality'],
                       help='Override model preference')
    parser.add_argument('--no_cache', action='store_true', help='Disable response caching')
    parser.add_argument('--cache_size', type=int, help='Override cache size')
    parser.add_argument('--max_workers', type=int, help='Override number of workers')
    parser.add_argument('--show_info', action='store_true', help='Show configuration and exit')
    
    return bt.config(parser)


def main():
    # Create Bittensor config
    config = create_bt_config()
    
    print("🚀 OptimizedMiner Launcher")
    print("=" * 50)
    
    # Show info and exit if requested
    if hasattr(config, 'show_info') and config.show_info:
        OptimizedMinerConfig.print_config()
        print_system_info()
        return 0
    
    # Apply hardware configuration
    hardware_tier = getattr(config, 'hardware_tier', 'mid_range')
    apply_hardware_config(hardware_tier)
    
    # Apply command line overrides
    if hasattr(config, 'model_mode') and config.model_mode:
        OptimizedMinerConfig.MODEL_PREFERENCE = config.model_mode
        print(f"✅ Model preference overridden to: {config.model_mode}")
    
    if hasattr(config, 'no_cache') and config.no_cache:
        OptimizedMinerConfig.ENABLE_CACHE = False
        print("✅ Caching disabled")
    
    if hasattr(config, 'cache_size') and config.cache_size:
        OptimizedMinerConfig.CACHE_MAX_SIZE = config.cache_size
        print(f"✅ Cache size set to: {config.cache_size}")
    
    if hasattr(config, 'max_workers') and config.max_workers:
        OptimizedMinerConfig.MAX_WORKERS = config.max_workers
        print(f"✅ Workers set to: {config.max_workers}")
    
    print()
    
    # Setup environment
    setup_environment()
    print_system_info()
    
    # Show final configuration
    OptimizedMinerConfig.print_config()
    
    print("🏃 Starting OptimizedMiner...")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Start the miner with Bittensor config
        with OptimizedMiner(config=config) as miner:
            print(f"✅ Miner started successfully (UID: {miner.uid})")
            print(f"🎯 Target: <10s response time for 100 time points")
            print(f"🎯 Target: >0.7 quality score for 70 quality points") 
            print()
            
            while True:
                bt.logging.info(f"OptimizedMiner running... Cache: {len(miner.response_cache)} entries")
                time.sleep(30)  # Log every 30 seconds instead of 5
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down OptimizedMiner...")
    except Exception as e:
        print(f"❌ Error running miner: {e}")
        bt.logging.error(f"Miner error: {e}")
        return 1
    
    print("✅ OptimizedMiner stopped successfully")
    return 0


if __name__ == "__main__":
    exit(main())
