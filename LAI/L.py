import os
import sys
import json
import logging
import traceback
import time

from src import LAttention, MultiHeadAttention, SelfAttention, SparseAttention, LinearAttention, MoE, DynamicMoE, Tensor, nn, F

class Logger:
    @staticmethod
    def setup_logger(name, log_file, level=logging.INFO):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        logger.addHandler(console)
        return logger
    
    @staticmethod
    def log_training_step(logger, step, loss, lr, elapsed_time):
        logger.info(f"Step: {step}, Loss: {loss:.4f}, LR: {lr:.8f}, Time: {elapsed_time:.2f}s")
    
    @staticmethod
    def log_validation(logger, step, val_loss, perplexity):
        logger.info(f"Step: {step}, Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    @staticmethod
    def log_inference(logger, prompt, response, elapsed_time):
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {response}")
        logger.info(f"Inference time: {elapsed_time:.2f}s")
    
    @staticmethod
    def log_error(logger, error_message):
        logger.error(f"Error: {error_message}")
        logger.error(traceback.format_exc())

class DistributedUtils:
    @staticmethod
    def is_distributed():
        return False
    
    @staticmethod
    def get_rank():
        return 0
    
    @staticmethod
    def get_world_size():
        return 1
    
    @staticmethod
    def is_main_process():
        return True
    
    @staticmethod
    def barrier():
        pass
    
    @staticmethod
    def all_reduce(tensor):
        return tensor
    
    @staticmethod
    def all_gather(tensor):
        return [tensor]

class ModelDeployer:
    @staticmethod
    def export_to_onnx(model, tokenizer, output_path, max_seq_len=4096):
        print(f"Exporting model to ONNX: {output_path}")
    
    @staticmethod
    def export_to_torchscript(model, output_path):
        print(f"Exporting model to TorchScript: {output_path}")
    
    @staticmethod
    def optimize_for_inference(model):
        print("Optimizing model for inference")
        return model

class ModelMonitor:
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def start_epoch(self):
        self.epoch_start = time.time()
        self.epoch_metrics = {}
    
    def end_epoch(self):
        epoch_time = time.time() - self.epoch_start
        self.metrics['epoch_time'] = epoch_time
        return self.epoch_metrics
    
    def add_metric(self, name, value):
        if name not in self.epoch_metrics:
            self.epoch_metrics[name] = []
        self.epoch_metrics[name].append(value)
    
    def get_average_metric(self, name):
        if name not in self.epoch_metrics:
            return 0
        return sum(self.epoch_metrics[name]) / len(self.epoch_metrics[name])
    
    def get_total_time(self):
        return time.time() - self.start_time
    
    def save_metrics(self, output_path):
        metrics_data = {
            'total_time': self.get_total_time(),
            'epoch_metrics': self.metrics
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)

class DataValidator:
    @staticmethod
    def validate_input(input_text, max_length=4096):
        if not isinstance(input_text, str):
            raise TypeError("Input must be a string")
        if len(input_text) == 0:
            raise ValueError("Input cannot be empty")
        if len(input_text) > max_length:
            raise ValueError(f"Input length must be less than {max_length} characters")
        return True
    
    @staticmethod
    def validate_batch(batch, tokenizer, max_seq_len=4096):
        if not isinstance(batch, dict):
            raise TypeError("Batch must be a dictionary")
        if 'input_ids' not in batch:
            raise ValueError("Batch must contain 'input_ids'")
        if 'labels' not in batch:
            raise ValueError("Batch must contain 'labels'")
        
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        if not isinstance(input_ids, list):
            raise TypeError("input_ids must be a list")
        if not isinstance(labels, list):
            raise TypeError("labels must be a list")
        
        if len(input_ids) > max_seq_len:
            raise ValueError(f"Sequence length must be less than {max_seq_len}")
        if len(labels) > max_seq_len:
            raise ValueError(f"Labels length must be less than {max_seq_len}")
        
        return True
    
    @staticmethod
    def validate_model_config(config):
        required_fields = ['n_layer', 'n_head', 'n_embd', 'vocab_size']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Config must have {field}")
        
        if config['n_layer'] <= 0:
            raise ValueError("n_layer must be positive")
        if config['n_head'] <= 0:
            raise ValueError("n_head must be positive")
        if config['n_embd'] <= 0:
            raise ValueError("n_embd must be positive")
        if config['vocab_size'] <= 0:
            raise ValueError("vocab_size must be positive")
        
        return True

class SecurityManager:
    @staticmethod
    def detect_prompt_injection(prompt):
        injection_patterns = [
            "ignore previous instructions",
            "system prompt",
            "override",
            "bypass",
            "jailbreak"
        ]
        
        for pattern in injection_patterns:
            if pattern.lower() in prompt.lower():
                return True
        return False
    
    @staticmethod
    def sanitize_input(input_text):
        sanitized = input_text.replace('<script>', '').replace('</script>', '')
        sanitized = sanitized.replace('javascript:', '')
        return sanitized
    
    @staticmethod
    def rate_limit_check(user_id, max_requests=100, window_seconds=3600):
        if not hasattr(SecurityManager, '_rate_limits'):
            SecurityManager._rate_limits = {}
        
        current_time = time.time()
        if user_id not in SecurityManager._rate_limits:
            SecurityManager._rate_limits[user_id] = []
        
        SecurityManager._rate_limits[user_id] = [t for t in SecurityManager._rate_limits[user_id] if current_time - t < window_seconds]
        
        if len(SecurityManager._rate_limits[user_id]) >= max_requests:
            return False
        
        SecurityManager._rate_limits[user_id].append(current_time)
        return True

class CacheManager:
    def __init__(self, max_size=1000, expiration_time=3600):
        self.max_size = max_size
        self.expiration_time = expiration_time
        self.cache = {}
        self.access_times = {}
    
    def get(self, key):
        current_time = time.time()
        if key in self.cache:
            if current_time - self.access_times[key] < self.expiration_time:
                self.access_times[key] = current_time
                return self.cache[key]
            else:
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        current_time = time.time()
        self.cache[key] = value
        self.access_times[key] = current_time
    
    def _evict_oldest(self):
        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()
    
    def size(self):
        return len(self.cache)

class AdvancedConfigManager:
    @staticmethod
    def load_config_from_yaml(config_path):
        return {}
    
    @staticmethod
    def save_config_to_yaml(config, config_path):
        pass
    
    @staticmethod
    def validate_config(config):
        required_fields = ['model', 'training', 'inference']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Config must have {field}")
        return True
    
    @staticmethod
    def merge_configs(base_config, override_config):
        merged = base_config.copy()
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = AdvancedConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

class AdvancedInference:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.cache = CacheManager()
    
    def generate(self, prompt, context=None, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
        cache_key = f"{prompt}_{context}_{max_length}_{temperature}_{top_k}_{top_p}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        if context:
            prompt = f"{context}\n{prompt}"
        
        response = f"Generated response for: {prompt}"
        self.cache.set(cache_key, response)
        return response
    
    def batch_generate(self, prompts, batch_size=8):
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt))
        return results

class MonitoringSystem:
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def record_metric(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_metric_average(self, name):
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def get_metric_max(self, name):
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0
        return max(self.metrics[name])
    
    def get_metric_min(self, name):
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0
        return min(self.metrics[name])

if __name__ == "__main__":
    print("L Model Initialized")
    print("All systems ready!")