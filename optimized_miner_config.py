# Configuration for OptimizedMiner
# This file contains settings to fine-tune miner performance

class OptimizedMinerConfig:
    """Configuration class for the optimized miner"""
    
    # Model Selection (choose based on your hardware)
    # 'speed': ~3-5s response, good for targeting <10s 
    # 'balanced': ~5-8s response, best quality/speed ratio (RECOMMENDED)
    # 'quality': ~8-12s response, highest quality but slower
    MODEL_PREFERENCE = 'balanced'
    
    # Cache Settings
    CACHE_MAX_SIZE = 1000  # Maximum number of cached responses
    ENABLE_CACHE = True    # Set to False to disable caching
    
    # Response Quality Settings
    MIN_RESPONSE_LENGTH = 50   # Minimum words in response
    MAX_RESPONSE_LENGTH = 250  # Maximum words in response
    QUALITY_THRESHOLD = 0.7    # Minimum quality score to accept response
    
    # Model Generation Settings
    GENERATION_CONFIG = {
        'max_new_tokens': 150,
        'min_new_tokens': 50,
        'temperature': 0.7,
        'repetition_penalty': 1.1,
        'do_sample': True,
        'early_stopping': True
    }
    
    # Threading Settings
    MAX_WORKERS = 4  # Number of thread pool workers
    
    # Hardware Optimization
    USE_QUANTIZATION = True  # Enable 4-bit quantization for speed
    USE_CUDA = True         # Use GPU if available
    
    # Template Response Settings
    PREFER_TEMPLATES = True  # Use template responses when possible for speed
    FALLBACK_TO_MODEL = True # Use model generation if template quality is low
    
    # Crisis Detection Keywords (add more as needed)
    CRISIS_KEYWORDS = [
        'suicide', 'kill myself', 'end it all', 'not worth living',
        'hurt myself', 'self harm', 'cutting', 'overdose', 'die'
    ]
    
    # Model Options (you can add more models here)
    MODEL_OPTIONS = {
        'speed': {
            'name': 'microsoft/DialoGPT-small',
            'expected_time': '2-4s',
            'quality': 'Good for quick responses',
            'vram': '1-2GB'
        },
        'balanced': {
            'name': 'microsoft/DialoGPT-medium',  # Alternative to gemma
            'expected_time': '4-8s', 
            'quality': 'Good quality responses, no auth needed',
            'vram': '2-4GB'
        },
        'quality': {
            'name': 'facebook/blenderbot-400M-distill',
            'expected_time': '8-12s', 
            'quality': 'Excellent',
            'vram': '6-8GB'
        }
    }
    
    @classmethod
    def get_model_info(cls, preference='balanced'):
        """Get information about selected model"""
        return cls.MODEL_OPTIONS.get(preference, cls.MODEL_OPTIONS['balanced'])
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print(f"""
OptimizedMiner Configuration:
============================
Model Preference: {cls.MODEL_PREFERENCE}
Model Details: {cls.get_model_info(cls.MODEL_PREFERENCE)}
Cache Enabled: {cls.ENABLE_CACHE} (Max: {cls.CACHE_MAX_SIZE})
Quality Threshold: {cls.QUALITY_THRESHOLD}
Response Length: {cls.MIN_RESPONSE_LENGTH}-{cls.MAX_RESPONSE_LENGTH} words
Quantization: {cls.USE_QUANTIZATION}
Thread Workers: {cls.MAX_WORKERS}
""")

# Performance tuning tips based on hardware
HARDWARE_RECOMMENDATIONS = {
    'low_end': {
        'description': 'RTX 3060 or similar (8-12GB VRAM)',
        'model_preference': 'speed',
        'quantization': True,
        'cache_size': 500,
        'max_workers': 2
    },
    'mid_range': {
        'description': 'RTX 3070/4060 Ti or similar (12-16GB VRAM)', 
        'model_preference': 'balanced',
        'quantization': True,
        'cache_size': 1000,
        'max_workers': 4
    },
    'high_end': {
        'description': 'RTX 3090/4080 or similar (16GB+ VRAM)',
        'model_preference': 'quality',
        'quantization': False,
        'cache_size': 2000,
        'max_workers': 6
    }
}

def get_hardware_config(tier='mid_range'):
    """Get recommended configuration based on hardware tier"""
    return HARDWARE_RECOMMENDATIONS.get(tier, HARDWARE_RECOMMENDATIONS['mid_range'])

if __name__ == "__main__":
    OptimizedMinerConfig.print_config()
    print("\nHardware Recommendations:")
    for tier, config in HARDWARE_RECOMMENDATIONS.items():
        print(f"\n{tier.upper()}: {config['description']}")
        print(f"  - Model: {config['model_preference']}")
        print(f"  - Quantization: {config['quantization']}")
        print(f"  - Cache Size: {config['cache_size']}")
        print(f"  - Workers: {config['max_workers']}")
