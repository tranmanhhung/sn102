#!/usr/bin/env python3
"""
Test script for OptimizedMiner performance
Usage: python test_optimized_miner.py
"""

import asyncio
import time
import statistics
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neurons.optimized_miner import OptimizedMiner, TherapyResponseGenerator
from BetterTherapy.protocol import InferenceSynapse
from optimized_miner_config import OptimizedMinerConfig
import bittensor as bt


class MinerTester:
    """Test class for evaluating miner performance"""
    
    def __init__(self):
        self.test_prompts = [
            "How can I manage my anxiety?",
            "What should I do if I feel overwhelmed at work?", 
            "How do I improve my sleep quality?",
            "I'm feeling sad lately, what can help?",
            "How can I build better relationships?",
            "What are some tips for handling stress?",
            "How do I set healthy boundaries?",
            "What can I do to boost my self-esteem?",
            "How do I cope with loneliness?",
            "What are effective ways to relax?",
            "I feel like nobody understands me",
            "My partner and I keep fighting about money",
            "I can't sleep because I'm worried about everything",
            "I feel worthless and like a failure",
            "How do I stop procrastinating on important tasks?"
        ]
        
        self.crisis_prompts = [
            "I don't want to live anymore",
            "Sometimes I think about hurting myself",
            "Life feels meaningless and I want it to end"
        ]
    
    async def test_response_times(self, miner, num_tests=10):
        """Test response times"""
        print(f"🕐 Testing response times ({num_tests} requests)...")
        
        times = []
        cache_hits = 0
        
        for i in range(num_tests):
            prompt = self.test_prompts[i % len(self.test_prompts)]
            
            synapse = InferenceSynapse(
                prompt=prompt,
                request_id=f"test_{i}"
            )
            
            start_time = time.time()
            result = await miner.forward(synapse)
            end_time = time.time()
            
            response_time = end_time - start_time
            times.append(response_time)
            
            # Check if it was a cache hit (very fast response)
            if response_time < 0.1:
                cache_hits += 1
            
            print(f"  Request {i+1}: {response_time:.2f}s {'(cached)' if response_time < 0.1 else ''}")
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        
        print(f"\n📊 Response Time Results:")
        print(f"  Average: {avg_time:.2f}s")
        print(f"  Median: {median_time:.2f}s") 
        print(f"  Min: {min_time:.2f}s")
        print(f"  Max: {max_time:.2f}s")
        print(f"  Cache Hits: {cache_hits}/{num_tests} ({cache_hits/num_tests*100:.1f}%)")
        
        # Score based on 30% weight for response time
        under_10s = sum(1 for t in times if t < 10)
        under_20s = sum(1 for t in times if 10 <= t < 20) 
        under_30s = sum(1 for t in times if 20 <= t < 30)
        over_30s = sum(1 for t in times if t >= 30)
        
        time_score = (under_10s * 100 + under_20s * 50 + under_30s * 20 + over_30s * 0) / num_tests
        weighted_time_score = time_score * 0.3
        
        print(f"  <10s: {under_10s} (100 pts each)")
        print(f"  10-20s: {under_20s} (50 pts each)")
        print(f"  20-30s: {under_30s} (20 pts each)")
        print(f"  >30s: {over_30s} (0 pts each)")
        print(f"  Time Score: {time_score:.1f}/100")
        print(f"  Weighted Time Score: {weighted_time_score:.1f}/30")
        
        return times, weighted_time_score
    
    def test_response_quality(self, miner, num_tests=5):
        """Test response quality"""
        print(f"\n💎 Testing response quality ({num_tests} requests)...")
        
        generator = TherapyResponseGenerator()
        quality_scores = []
        
        for i in range(num_tests):
            prompt = self.test_prompts[i]
            
            # Generate response
            try:
                response = miner.generate_optimized_response(prompt)
                quality = miner.validate_response_quality(prompt, response)
                
                print(f"  Request {i+1}:")
                print(f"    Prompt: {prompt}")
                print(f"    Response Length: {len(response.split())} words")
                print(f"    Quality Pass: {'✅' if quality else '❌'}")
                print(f"    Preview: {response[:100]}...")
                print()
                
                # Simple quality scoring (in real scenario, this would be GPT-4)
                quality_score = 0.8 if quality else 0.5  # Simulate good template responses
                quality_scores.append(quality_score)
                
            except Exception as e:
                print(f"    Error: {e}")
                quality_scores.append(0.0)
        
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        weighted_quality_score = avg_quality * 100 * 0.7  # 70% weight
        
        print(f"📊 Quality Results:")
        print(f"  Average Quality: {avg_quality:.2f}/1.0")
        print(f"  Weighted Quality Score: {weighted_quality_score:.1f}/70")
        
        return quality_scores, weighted_quality_score
    
    def test_crisis_detection(self, miner):
        """Test crisis detection"""
        print(f"\n🚨 Testing crisis detection...")
        
        generator = TherapyResponseGenerator()
        
        for i, prompt in enumerate(self.crisis_prompts):
            urgency = generator.assess_urgency(prompt)
            response = generator.get_crisis_response() if urgency == 'crisis' else "Normal response"
            
            print(f"  Crisis Test {i+1}:")
            print(f"    Prompt: {prompt}")
            print(f"    Detected: {'🚨 CRISIS' if urgency == 'crisis' else '✅ Normal'}")
            print(f"    Response Type: {'Crisis hotline info' if urgency == 'crisis' else 'Regular therapy'}")
            print()
    
    def test_cache_performance(self, miner):
        """Test caching effectiveness"""
        print(f"\n💾 Testing cache performance...")
        
        # Test same prompt multiple times
        test_prompt = "How can I manage my anxiety?"
        
        times = []
        for i in range(3):
            synapse = InferenceSynapse(
                prompt=test_prompt,
                request_id=f"cache_test_{i}"
            )
            
            start_time = time.time()
            asyncio.run(miner.forward(synapse))
            end_time = time.time()
            
            times.append(end_time - start_time)
            print(f"  Attempt {i+1}: {times[i]:.3f}s")
        
        print(f"  Cache Effectiveness: {times[0]:.3f}s → {times[1]:.3f}s → {times[2]:.3f}s")
        print(f"  Speedup: {times[0]/times[1]:.1f}x after caching")


async def main():
    """Main test function"""
    print("🧪 OptimizedMiner Performance Test")
    print("=" * 50)
    
    # Create test configuration
    class TestConfig:
        def __init__(self):
            self.model_preference = 'balanced'
            self.blacklist = type('obj', (object,), {
                'allow_non_registered': True,
                'force_validator_permit': False
            })()
    
    config = TestConfig()
    
    # Initialize miner
    print("🔧 Initializing OptimizedMiner...")
    try:
        miner = OptimizedMiner(config=config)
        print("✅ Miner initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize miner: {e}")
        return
    
    # Create tester
    tester = MinerTester()
    
    # Run tests
    try:
        # Test response times
        times, time_score = await tester.test_response_times(miner, num_tests=8)
        
        # Test response quality  
        qualities, quality_score = tester.test_response_quality(miner, num_tests=5)
        
        # Test crisis detection
        tester.test_crisis_detection(miner)
        
        # Test cache performance
        tester.test_cache_performance(miner)
        
        # Calculate total score
        total_score = time_score + quality_score
        
        print("\n" + "=" * 50)
        print("🏆 FINAL PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Response Time Score: {time_score:.1f}/30 ({time_score/30*100:.1f}%)")
        print(f"Quality Score: {quality_score:.1f}/70 ({quality_score/70*100:.1f}%)")
        print(f"TOTAL SCORE: {total_score:.1f}/100 ({total_score:.1f}%)")
        print()
        
        if total_score >= 80:
            print("🥇 EXCELLENT - Miner is highly optimized!")
        elif total_score >= 60:
            print("🥈 GOOD - Miner is well optimized")
        elif total_score >= 40:
            print("🥉 FAIR - Some optimizations needed")
        else:
            print("❌ POOR - Significant optimizations required")
        
        print(f"\nExpected improvement over standard miner: +{total_score-40:.1f} points")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
