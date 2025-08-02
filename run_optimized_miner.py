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
    import psutil
    
    print("🖥️  System Information:")
    print(f"   CPU: {psutil.cpu_count()} cores")
    print(f"   RAM: {psutil.virtual_memory().total // (1024**3)} GB")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
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


def create_optimized_config():
    """Create a config object with optimized settings"""
    class Config:
        def __init__(self):
            self.model_preference = OptimizedMinerConfig.MODEL_PREFERENCE
            self.blacklist = type('obj', (object,), {
                'allow_non_registered': False,
                'force_validator_permit': True
            })()
    
    return Config()


def main():
    parser = argparse.ArgumentParser(description='Run OptimizedMiner with custom configuration')
    parser.add_argument('--config', choices=['low_end', 'mid_range', 'high_end'], 
                       default='mid_range', help='Hardware configuration tier')
    parser.add_argument('--model', choices=['speed', 'balanced', 'quality'],
                       help='Override model preference')
    parser.add_argument('--no-cache', action='store_true', help='Disable response caching')
    parser.add_argument('--cache-size', type=int, help='Override cache size')
    parser.add_argument('--workers', type=int, help='Override number of workers')
    parser.add_argument('--info', action='store_true', help='Show configuration and exit')
    
    args = parser.parse_args()
    
    print("🚀 OptimizedMiner Launcher")
    print("=" * 50)
    
    # Show info and exit if requested
    if args.info:
        OptimizedMinerConfig.print_config()
        print_system_info()
        return
    
    # Apply hardware configuration
    apply_hardware_config(args.config)
    
    # Apply command line overrides
    if args.model:
        OptimizedMinerConfig.MODEL_PREFERENCE = args.model
        print(f"✅ Model preference overridden to: {args.model}")
    
    if args.no_cache:
        OptimizedMinerConfig.ENABLE_CACHE = False
        print("✅ Caching disabled")
    
    if args.cache_size:
        OptimizedMinerConfig.CACHE_MAX_SIZE = args.cache_size
        print(f"✅ Cache size set to: {args.cache_size}")
    
    if args.workers:
        OptimizedMinerConfig.MAX_WORKERS = args.workers
        print(f"✅ Workers set to: {args.workers}")
    
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
        # Create optimized config
        config = create_optimized_config()
        
        # Start the miner
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
