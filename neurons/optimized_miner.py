# The MIT License (MIT)
# Copyright © 2025 BetterTherapy - Optimized Miner

import asyncio
import hashlib
import json
import time
import typing
from concurrent.futures import ThreadPoolExecutor

import bittensor as bt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import BitsAndBytesConfig, fallback if not available
try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    bt.logging.warning("BitsAndBytesConfig not available. 4-bit quantization will be disabled.")
    QUANTIZATION_AVAILABLE = False
    BitsAndBytesConfig = None

# Bittensor Miner Template:
import BetterTherapy

# import base miner class which takes care of most of the boilerplate
from BetterTherapy.base.miner import BaseMinerNeuron


class TherapyResponseGenerator:
    """Advanced therapy response generator with templates and optimization"""
    
    def __init__(self):
        self.response_templates = {
            'anxiety': {
                'validation': "I understand how overwhelming anxiety can feel, and you're not alone in experiencing this.",
                'techniques': [
                    "Practice the 4-7-8 breathing technique: breathe in for 4, hold for 7, out for 8",
                    "Use the 5-4-3-2-1 grounding method: 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste",
                    "Challenge anxious thoughts by asking: 'Is this thought helpful? What evidence supports/contradicts it?'"
                ],
                'encouragement': "Remember, anxiety is treatable and you have the strength to work through this step by step."
            },
            'depression': {
                'validation': "I hear the heaviness you're carrying, and it takes real courage to reach out for support.",
                'techniques': [
                    "Start with small, achievable daily goals - even getting dressed or making your bed counts",
                    "Practice behavioral activation: schedule one small pleasant activity each day",
                    "Connect with one supportive person, even if it's just a brief text or call"
                ],
                'encouragement': "Depression tells lies about your worth. You matter, and this feeling won't last forever."
            },
            'stress': {
                'validation': "Feeling overwhelmed by stress is completely understandable given what you're facing.",
                'techniques': [
                    "Practice progressive muscle relaxation: tense and release each muscle group for 5 seconds",
                    "Break overwhelming tasks into smaller, manageable steps",
                    "Set boundaries: it's okay to say no to additional responsibilities right now"
                ],
                'encouragement': "You're stronger than you realize, and learning to manage stress is a skill that improves with practice."
            },
            'relationships': {
                'validation': "Relationship challenges can feel deeply personal and confusing. Your feelings are valid.",
                'techniques': [
                    "Practice 'I' statements: 'I feel...' instead of 'You always...'",
                    "Listen actively: repeat back what you heard before responding",
                    "Take breaks during heated discussions to cool down and reflect"
                ],
                'encouragement': "Healthy relationships take work from both people, and you're showing wisdom by seeking guidance."
            },
            'sleep': {
                'validation': "Sleep difficulties can be incredibly frustrating and impact every aspect of your well-being.",
                'techniques': [
                    "Create a consistent bedtime routine: same time, same calming activities each night",
                    "Keep your bedroom cool, dark, and quiet - consider blackout curtains or white noise",
                    "Avoid screens 1 hour before bed; try reading or gentle stretching instead"
                ],
                'encouragement': "Good sleep hygiene takes time to establish, but your body will thank you for the consistency."
            },
            'general': {
                'validation': "Thank you for sharing what's on your mind. Your feelings and experiences are important.",
                'techniques': [
                    "Practice mindfulness: spend 5 minutes focusing on your breath each day",
                    "Keep a gratitude journal: write down 3 things you're grateful for daily",
                    "Engage in regular physical activity, even a 10-minute walk can boost mood"
                ],
                'encouragement': "Taking care of your mental health is an ongoing journey, and every small step matters."
            }
        }
        
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'not worth living', 
            'hurt myself', 'self harm', 'cutting', 'overdose'
        ]
    
    def classify_prompt_type(self, prompt: str) -> str:
        """Classify the type of therapy prompt"""
        keywords = {
            'anxiety': ['anxiety', 'anxious', 'worry', 'nervous', 'panic', 'fear', 'worried'],
            'depression': ['sad', 'depressed', 'hopeless', 'empty', 'worthless', 'down', 'low'],
            'stress': ['stress', 'overwhelmed', 'pressure', 'burned out', 'exhausted'],
            'relationships': ['relationship', 'partner', 'friends', 'family', 'conflict', 'argument'],
            'sleep': ['sleep', 'insomnia', 'can\'t sleep', 'tired', 'rest', 'sleeping']
        }
        
        prompt_lower = prompt.lower()
        for category, words in keywords.items():
            if any(word in prompt_lower for word in words):
                return category
        return 'general'
    
    def assess_urgency(self, prompt: str) -> str:
        """Assess if this is a crisis situation"""
        prompt_lower = prompt.lower()
        if any(keyword in prompt_lower for keyword in self.crisis_keywords):
            return 'crisis'
        return 'normal'
    
    def get_crisis_response(self) -> str:
        """Return appropriate crisis response"""
        return """I'm deeply concerned about what you've shared. Your life has value and meaning, even when it doesn't feel that way.

Please reach out for immediate support:
• National Suicide Prevention Lifeline: 988 or 1-800-273-8255
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911

You don't have to face this alone. Professional counselors are available 24/7 and want to help. Please consider reaching out to a mental health professional or trusted person in your life right now."""

    def generate_structured_response(self, prompt: str, prompt_type: str) -> str:
        """Generate a structured therapy response"""
        template = self.response_templates[prompt_type]
        
        # Select 2-3 most relevant techniques
        techniques = template['techniques'][:2]  # Take first 2 for conciseness
        
        response_parts = [
            template['validation'],
            "",  # Empty line for spacing
            "Here are some evidence-based strategies that can help:",
        ]
        
        # Add techniques
        for i, technique in enumerate(techniques, 1):
            response_parts.append(f"{i}. {technique}")
        
        response_parts.extend([
            "",  # Empty line
            template['encouragement']
        ])
        
        return "\n".join(response_parts)


class OptimizedMiner(BaseMinerNeuron):
    """
    Optimized miner for maximum performance and quality scores.
    Features:
    - Response caching for speed
    - Advanced therapy response templates  
    - Quantized model for faster inference
    - Async processing
    - Quality validation
    """

    def __init__(self, config=None):
        super(OptimizedMiner, self).__init__(config=config)
        
        # Initialize components
        self.response_cache = {}
        self.cache_max_size = 1000
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.therapy_generator = TherapyResponseGenerator()
        
        # Setup optimized model
        self.setup_optimized_model(config)
        
        bt.logging.info(f"OptimizedMiner initialized with uid: {self.uid}")
        bt.logging.info(f"Model: {self.model_name}")
        bt.logging.info(f"Cache enabled: {len(self.response_cache)} entries")

    def setup_optimized_model(self, config):
        """Setup quantized model for optimal speed/quality balance"""
        # Choose model based on config or use balanced default
        model_options = {
            'speed': 'microsoft/DialoGPT-small',      # ~3-5s, good for <10s target
            'balanced': 'microsoft/DialoGPT-medium',  # ~5-8s, best quality/speed
            'quality': 'facebook/blenderbot-400M-distill'  # ~8-12s, highest quality
        }
        
        model_preference = getattr(config, 'model_preference', 'balanced') if config else 'balanced'
        self.model_name = model_options.get(model_preference, model_options['balanced'])
        
        bt.logging.info(f"Loading model: {self.model_name} (preference: {model_preference})")
        
        # Quantization config for speed (if available)
        quantization_config = None
        if QUANTIZATION_AVAILABLE and OptimizedMinerConfig.USE_QUANTIZATION:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            bt.logging.info("4-bit quantization enabled")
        else:
            bt.logging.info("4-bit quantization disabled (bitsandbytes not available or disabled in config)")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True
            }
            
            # Only add quantization_config if it's available
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            self.model.eval()
            if quantization_config is not None:
                bt.logging.info("Model loaded successfully with 4-bit quantization")
            else:
                bt.logging.info("Model loaded successfully without quantization")
            
        except Exception as e:
            bt.logging.error(f"Error loading quantized model: {e}")
            bt.logging.info("Falling back to standard model loading...")
            
            # Fallback to standard loading
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()
            
            if torch.cuda.is_available():
                self.model.to("cuda")

    def get_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()

    def manage_cache(self):
        """Keep cache size under limit"""
        if len(self.response_cache) > self.cache_max_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.response_cache.keys())[:-self.cache_max_size//2]
            for key in keys_to_remove:
                del self.response_cache[key]

    async def forward(
        self, synapse: BetterTherapy.protocol.InferenceSynapse
    ) -> BetterTherapy.protocol.InferenceSynapse:
        """
        Optimized forward pass with caching and async processing
        """
        start_time = time.time()
        bt.logging.info(f"Processing request: {synapse.request_id}")
        
        try:
            # Check cache first
            cache_key = self.get_cache_key(synapse.prompt)
            if cache_key in self.response_cache:
                bt.logging.info(f"Cache hit for request: {synapse.request_id}")
                synapse.output = self.response_cache[cache_key]
                processing_time = time.time() - start_time
                bt.logging.info(f"Response generated in {processing_time:.2f}s (cached)")
                return synapse

            # Generate new response asynchronously
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                self.executor, 
                self.generate_optimized_response, 
                synapse.prompt
            )
            
            # Cache the response
            self.response_cache[cache_key] = output
            self.manage_cache()
            
            synapse.output = output
            processing_time = time.time() - start_time
            bt.logging.info(f"Response generated in {processing_time:.2f}s (new)")
            
            return synapse
            
        except Exception as e:
            bt.logging.error(f"Error in forward pass: {e}")
            synapse.output = self.get_fallback_response()
            return synapse

    def generate_optimized_response(self, prompt: str) -> str:
        """Generate optimized therapy response"""
        try:
            # Quick safety check
            urgency = self.therapy_generator.assess_urgency(prompt)
            if urgency == 'crisis':
                return self.therapy_generator.get_crisis_response()
            
            # Classify prompt type
            prompt_type = self.therapy_generator.classify_prompt_type(prompt)
            
            # Try template-based response first (fastest)
            if prompt_type != 'general':
                template_response = self.therapy_generator.generate_structured_response(prompt, prompt_type)
                if self.validate_response_quality(prompt, template_response):
                    return template_response
            
            # Fall back to model generation if template doesn't meet quality threshold
            return self.generate_model_response(prompt)
            
        except Exception as e:
            bt.logging.error(f"Error generating response: {e}")
            return self.get_fallback_response()

    def generate_model_response(self, prompt: str) -> str:
        """Generate response using the language model"""
        system_prompt = self.get_optimized_system_prompt()
        
        # Format input
        input_text = f"{system_prompt}\n\nUser: {prompt}\n\nTherapist:"
        
        try:
            # Tokenize
            inputs = self.tokenizer.encode(
                input_text, 
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # Generate with optimized parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=150,
                    min_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    early_stopping=True,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Post-process
            response = self.post_process_response(response)
            return response
            
        except Exception as e:
            bt.logging.error(f"Model generation error: {e}")
            return self.get_fallback_response()

    def get_optimized_system_prompt(self) -> str:
        """Get optimized system prompt for therapy responses"""
        return """You are Dr. Sarah Chen, a licensed clinical psychologist with 15+ years of experience in cognitive behavioral therapy (CBT) and mindfulness-based interventions.

Your response guidelines:
1. EMPATHY FIRST: Always validate the person's feelings
2. EVIDENCE-BASED: Use proven therapeutic techniques (CBT, DBT, mindfulness)
3. ACTIONABLE: Provide 2-3 specific, practical strategies
4. PROFESSIONAL: Maintain appropriate boundaries
5. CONCISE: Keep responses 100-150 words for optimal engagement

Structure your response:
- Acknowledge and validate their experience
- Provide 2-3 evidence-based coping strategies
- End with encouragement and hope

Remember: You're providing supportive guidance, not diagnosing or replacing professional treatment."""

    def post_process_response(self, response: str) -> str:
        """Post-process the generated response"""
        # Remove any unwanted prefixes/suffixes
        response = response.replace("Therapist:", "").strip()
        
        # Ensure appropriate length
        words = response.split()
        if len(words) > 200:
            response = " ".join(words[:200]) + "..."
        elif len(words) < 30:
            response += " I encourage you to keep exploring these feelings and consider reaching out to a mental health professional for personalized support."
        
        # Ensure empathetic tone
        if not any(word in response.lower() for word in ['understand', 'hear', 'feel', 'sense']):
            response = f"I understand this is challenging for you. {response}"
        
        return response.strip()

    def validate_response_quality(self, prompt: str, response: str) -> bool:
        """Quick quality validation for responses"""
        if not response or len(response.strip()) < 50:
            return False
        
        quality_checks = {
            'length': 50 <= len(response.split()) <= 250,
            'empathy': any(word in response.lower() for word in ['understand', 'feel', 'hear', 'sense']),
            'actionable': any(word in response.lower() for word in ['try', 'practice', 'consider', 'can', 'help']),
            'professional': not any(word in response.lower() for word in ['stupid', 'crazy', 'weird', 'dumb']),
            'structure': '.' in response  # Has at least one complete sentence
        }
        
        score = sum(quality_checks.values()) / len(quality_checks)
        return score >= 0.7

    def get_fallback_response(self) -> str:
        """Fallback response when generation fails"""
        return """I understand you're going through a difficult time, and I want you to know that your feelings are valid. While I'm having trouble processing your specific situation right now, I encourage you to:

1. Take some deep breaths and ground yourself in the present moment
2. Reach out to a trusted friend, family member, or mental health professional
3. Practice self-compassion - treat yourself with the same kindness you'd show a good friend

Remember, seeking help is a sign of strength, and you don't have to face this alone. If you're in crisis, please contact a mental health hotline or emergency services immediately."""

    async def blacklist(
        self, synapse: BetterTherapy.protocol.InferenceSynapse
    ) -> typing.Tuple[bool, str]:
        """
        Enhanced blacklisting with performance considerations
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        try:
            # Quick validation check
            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                if not self.config.blacklist.allow_non_registered:
                    bt.logging.trace(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
                    return True, "Unrecognized hotkey"
            
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            
            # Validator permit check
            if self.config.blacklist.force_validator_permit:
                if not self.metagraph.validator_permit[uid]:
                    bt.logging.warning(f"Blacklisting non-validator hotkey {synapse.dendrite.hotkey}")
                    return True, "Non-validator hotkey"

            bt.logging.trace(f"Allowing request from {synapse.dendrite.hotkey}")
            return False, "Hotkey recognized!"
            
        except Exception as e:
            bt.logging.error(f"Error in blacklist check: {e}")
            return True, "Blacklist error"

    async def priority(self, synapse: BetterTherapy.protocol.InferenceSynapse) -> float:
        """
        Enhanced priority calculation
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0

        try:
            caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            base_priority = float(self.metagraph.S[caller_uid])
            
            # Boost priority for urgent requests
            if self.therapy_generator.assess_urgency(synapse.prompt) == 'crisis':
                base_priority *= 2.0
            
            bt.logging.trace(f"Priority for {synapse.dendrite.hotkey}: {base_priority}")
            return base_priority
            
        except Exception as e:
            bt.logging.error(f"Error calculating priority: {e}")
            return 0.0


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with OptimizedMiner() as miner:
        while True:
            bt.logging.info(f"OptimizedMiner running... {time.time()}")
            time.sleep(5)
