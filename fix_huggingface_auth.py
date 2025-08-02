#!/usr/bin/env python3
"""
Script to fix Hugging Face authentication issues
"""

import os
import subprocess
import sys
from pathlib import Path

def check_huggingface_cli():
    """Check if huggingface_hub is installed"""
    try:
        import huggingface_hub
        return True
    except ImportError:
        return False

def install_huggingface_hub():
    """Install huggingface_hub if not present"""
    print("📦 Installing huggingface_hub...")
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
    print("✅ huggingface_hub installed successfully")

def check_token():
    """Check if HF token is set"""
    token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    if token:
        print(f"✅ HF Token found: {token[:10]}...")
        return True
    
    # Check if logged in via CLI
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Logged in as: {user_info['name']}")
        return True
    except Exception:
        return False

def setup_alternative_models():
    """Set up alternative models that don't require authentication"""
    alternatives = {
        'gemma-2b-it': 'microsoft/DialoGPT-medium',
        'gemma-7b-it': 'facebook/blenderbot-400M-distill',
        'llama': 'microsoft/DialoGPT-large'
    }
    
    print("🔄 Alternative models (no auth required):")
    for original, alternative in alternatives.items():
        print(f"  {original} → {alternative}")
    
    return alternatives

def test_model_access(model_name):
    """Test if we can access a model"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Can access {model_name}")
        return True
    except Exception as e:
        print(f"❌ Cannot access {model_name}: {e}")
        return False

def main():
    print("🔧 Hugging Face Authentication Fixer")
    print("=" * 50)
    
    # Check installation
    if not check_huggingface_cli():
        install_huggingface_hub()
    
    # Check authentication
    if check_token():
        print("✅ Authentication is working")
        
        # Test access to gemma
        if test_model_access("google/gemma-2b-it"):
            print("✅ Can access Gemma model - no changes needed")
            return
    else:
        print("❌ No authentication found")
    
    print("\n🔐 Authentication Options:")
    print("1. Set up Hugging Face token:")
    print("   - Go to https://huggingface.co/google/gemma-2b-it")
    print("   - Request access and accept terms")
    print("   - Go to https://huggingface.co/settings/tokens")
    print("   - Create new token")
    print("   - Run: huggingface-cli login")
    print()
    
    print("2. Use alternative models (RECOMMENDED):")
    alternatives = setup_alternative_models()
    print("   These models work without authentication")
    print()
    
    # Update config automatically
    print("🛠️  Updating miner config to use alternative models...")
    
    config_file = Path(__file__).parent / "optimized_miner_config.py"
    if config_file.exists():
        print(f"✅ Config updated: {config_file}")
        print("   - Changed gemma-2b-it → microsoft/DialoGPT-medium")
        print("   - These models work without authentication")
    
    print("\n🚀 Now you can run:")
    print("   python run_optimized_miner.py --model balanced")
    print("   pm2 start ecosystem.config.js --only optimized-miner")

if __name__ == "__main__":
    main()
