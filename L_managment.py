import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

class AIConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._get_default_config()
        self._load_config()

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "model": {
                "name": "default_model",
                "type": "transformer",
                "version": "1.0.0",
                "vocab_size": 50257,
                "hidden_size": 768,
                "num_layers": 12,
                "num_heads": 12,
                "max_position_embeddings": 1024
            },
            "generation": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "num_return_sequences": 1,
                "max_new_tokens": 100,
                "min_new_tokens": 0,
                "repetition_penalty": 1.0,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "early_stopping": False,
                "do_sample": True
            },
            "context": {
                "max_length": 1024,
                "window_size": 512,
                "overlap_size": 128,
                "memory_tokens": 0,
                "enable_caching": True,
                "cache_size": 1000
            },
            "sampling": {
                "strategy": "top_p",
                "temperature_range": [0.1, 2.0],
                "top_p_range": [0.1, 1.0],
                "top_k_range": [1, 100],
                "typical_p": 1.0,
                "epsilon_cutoff": 0.0,
                "eta_cutoff": 0.0
            },
            "resources": {
                "device": "auto",
                "num_gpus": 1,
                "batch_size": 1,
                "max_memory": "auto",
                "use_mixed_precision": True,
                "use_fused_attention": True
            },
            "evaluation": {
                "metrics": ["perplexity", "bleu", "rouge"],
                "batch_size": 32,
                "max_eval_samples": 1000,
                "eval_interval": 1000
            },
            "logging": {
                "level": "info",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "log_file": "ai_model.log",
                "enable_tensorboard": False,
                "tensorboard_dir": "runs"
            },
            "safety": {
                "enable_safety_checks": True,
                "block_threshold": 0.7,
                "allowed_categories": ["general", "creative", "technical"],
                "blocked_categories": ["harmful", "illegal", "offensive"]
            },
            "api": {
                "enable_api": False,
                "api_key": "",
                "api_url": "http://localhost:8000",
                "timeout": 30,
                "max_requests_per_minute": 60
            },
            "version": {
                "config_version": "1.0.0",
                "last_updated": datetime.now().isoformat()
            }
        }

    def _load_config(self) -> None:
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                self.config = self._merge_configs(self.config, loaded_config)
                print(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                print(f"Error loading configuration: {str(e)}")
                print("Using default configuration")
        else:
            print(f"Configuration file {self.config_file} not found. Using default configuration.")

    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        merged = default.copy()
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def save_config(self, file_path: Optional[str] = None) -> bool:
        save_path = file_path or self.config_file
        try:
            self.config["version"]["last_updated"] = datetime.now().isoformat()
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            print(f"Configuration saved to {save_path}")
            return True
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")
            return False

    def get_config(self, key: Optional[str] = None) -> Any:
        if key is None:
            return self.config
        
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value

    def set_config(self, key: str, value: Any) -> bool:
        try:
            keys = key.split('.')
            config = self.config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
            print(f"Set {key} to {value}")
            return True
        except Exception as e:
            print(f"Error setting configuration: {str(e)}")
            return False

    def validate_config(self) -> bool:
        try:
            temperature = self.get_config("generation.temperature")
            if temperature < 0 or temperature > 10:
                print("Invalid temperature value. Must be between 0 and 10.")
                return False
            
            top_p = self.get_config("generation.top_p")
            if top_p < 0 or top_p > 1:
                print("Invalid top_p value. Must be between 0 and 1.")
                return False
            
            top_k = self.get_config("generation.top_k")
            if top_k < 0:
                print("Invalid top_k value. Must be non-negative.")
                return False
            
            max_new_tokens = self.get_config("generation.max_new_tokens")
            if max_new_tokens < 1 or max_new_tokens > 10000:
                print("Invalid max_new_tokens value. Must be between 1 and 10000.")
                return False
            
            window_size = self.get_config("context.window_size")
            if window_size < 1:
                print("Invalid window_size value. Must be positive.")
                return False
            
            device = self.get_config("resources.device")
            valid_devices = ["auto", "cpu", "cuda", "mps"]
            if device not in valid_devices:
                print(f"Invalid device value. Must be one of: {', '.join(valid_devices)}")
                return False
            
            print("Configuration validation passed")
            return True
        except Exception as e:
            print(f"Error validating configuration: {str(e)}")
            return False

    def reset_config(self, section: Optional[str] = None) -> bool:
        try:
            default_config = self._get_default_config()
            if section is None:
                self.config = default_config
                print("All configuration reset to default")
            else:
                if section in self.config and section in default_config:
                    self.config[section] = default_config[section]
                    print(f"Section {section} reset to default")
                else:
                    print(f"Section {section} not found in configuration")
                    return False
            return True
        except Exception as e:
            print(f"Error resetting configuration: {str(e)}")
            return False

    def export_config(self, format: str = "json") -> Optional[str]:
        try:
            if format == "json":
                return json.dumps(self.config, ensure_ascii=False, indent=2)
            elif format == "yaml":
                return yaml.dump(self.config, default_flow_style=False, allow_unicode=True)
            elif format == "ini":
                config_parser = configparser.ConfigParser()
                for section, values in self.config.items():
                    if isinstance(values, dict):
                        config_parser[section] = {k: str(v) for k, v in values.items()}
                output = []
                config_parser.write(output)
                return ''.join(output)
            else:
                print(f"Unsupported export format: {format}")
                return None
        except Exception as e:
            print(f"Error exporting configuration: {str(e)}")
            return None

    def import_config(self, config_str: str, format: str = "json") -> bool:
        try:
            if format == "json":
                imported_config = json.loads(config_str)
            elif format == "yaml":
                imported_config = yaml.safe_load(config_str)
            elif format == "ini":
                config_parser = configparser.ConfigParser()
                config_parser.read_string(config_str)
                imported_config = {}
                for section in config_parser.sections():
                    imported_config[section] = dict(config_parser[section])
            else:
                print(f"Unsupported import format: {format}")
                return False
            
            self.config = self._merge_configs(self.config, imported_config)
            print("Configuration imported successfully")
            return True
        except Exception as e:
            print(f"Error importing configuration: {str(e)}")
            return False

    def get_parameter_range(self, parameter: str) -> Optional[List[float]]:
        ranges = {
            "temperature": [0.1, 2.0],
            "top_p": [0.1, 1.0],
            "top_k": [1, 100],
            "max_new_tokens": [1, 10000],
            "window_size": [1, 4096],
            "repetition_penalty": [0.1, 2.0],
            "length_penalty": [0.1, 2.0]
        }
        return ranges.get(parameter)

    def optimize_for_speed(self) -> bool:
        try:
            speed_config = {
                "generation": {
                    "temperature": 0.1,
                    "top_k": 1,
                    "do_sample": False
                },
                "resources": {
                    "use_mixed_precision": True,
                    "use_fused_attention": True
                }
            }
            self.config = self._merge_configs(self.config, speed_config)
            print("Configuration optimized for speed")
            return True
        except Exception as e:
            print(f"Error optimizing configuration for speed: {str(e)}")
            return False

    def optimize_for_quality(self) -> bool:
        try:
            quality_config = {
                "generation": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "do_sample": True
                },
                "context": {
                    "max_length": 2048,
                    "window_size": 1024
                }
            }
            self.config = self._merge_configs(self.config, quality_config)
            print("Configuration optimized for quality")
            return True
        except Exception as e:
            print(f"Error optimizing configuration for quality: {str(e)}")
            return False

    def optimize_for_memory(self) -> bool:
        try:
            memory_config = {
                "context": {
                    "max_length": 512,
                    "window_size": 256,
                    "enable_caching": False
                },
                "resources": {
                    "batch_size": 1,
                    "use_mixed_precision": True
                }
            }
            self.config = self._merge_configs(self.config, memory_config)
            print("Configuration optimized for memory")
            return True
        except Exception as e:
            print(f"Error optimizing configuration for memory: {str(e)}")
            return False

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "model": self.config.get("model", {}).get("name", "N/A"),
            "generation": {
                "temperature": self.config.get("generation", {}).get("temperature", 0.7),
                "max_new_tokens": self.config.get("generation", {}).get("max_new_tokens", 100),
                "sampling_strategy": self.config.get("sampling", {}).get("strategy", "top_p")
            },
            "context": {
                "max_length": self.config.get("context", {}).get("max_length", 1024),
                "window_size": self.config.get("context", {}).get("window_size", 512)
            },
            "resources": {
                "device": self.config.get("resources", {}).get("device", "auto"),
                "use_mixed_precision": self.config.get("resources", {}).get("use_mixed_precision", True)
            },
            "version": self.config.get("version", {}).get("config_version", "N/A"),
            "last_updated": self.config.get("version", {}).get("last_updated", "N/A")
        }
        return summary

    def print_summary(self) -> None:
        summary = self.get_summary()
        print("=" * 60)
        print("AI Model Configuration Summary")
        print("=" * 60)
        print(f"Model: {summary['model']}")
        print("\nGeneration Parameters:")
        print(f"  Temperature: {summary['generation']['temperature']}")
        print(f"  Max New Tokens: {summary['generation']['max_new_tokens']}")
        print(f"  Sampling Strategy: {summary['generation']['sampling_strategy']}")
        print("\nContext Parameters:")
        print(f"  Max Length: {summary['context']['max_length']}")
        print(f"  Window Size: {summary['context']['window_size']}")
        print("\nResources:")
        print(f"  Device: {summary['resources']['device']}")
        print(f"  Mixed Precision: {summary['resources']['use_mixed_precision']}")
        print("\nVersion:")
        print(f"  Config Version: {summary['version']}")
        print(f"  Last Updated: {summary['last_updated']}")
        print("=" * 60)


class ConfigPresets:
    @staticmethod
    def get_preset(name: str) -> Dict[str, Any]:
        presets = {
            "default": {},
            "speed": {
                "generation": {
                    "temperature": 0.1,
                    "top_k": 1,
                    "do_sample": False
                },
                "resources": {
                    "use_mixed_precision": True,
                    "use_fused_attention": True
                }
            },
            "quality": {
                "generation": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "do_sample": True
                },
                "context": {
                    "max_length": 2048,
                    "window_size": 1024
                }
            },
            "memory": {
                "context": {
                    "max_length": 512,
                    "window_size": 256,
                    "enable_caching": False
                },
                "resources": {
                    "batch_size": 1,
                    "use_mixed_precision": True
                }
            },
            "creative": {
                "generation": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "do_sample": True
                }
            },
            "precise": {
                "generation": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 10,
                    "do_sample": True
                }
            }
        }
        return presets.get(name, presets["default"])

    @staticmethod
    def list_presets() -> List[str]:
        return ["default", "speed", "quality", "memory", "creative", "precise"]


class ConfigRecommender:
    def __init__(self, history_file: str = "config_history.json"):
        self.history_file = history_file
        self.history = self._load_history()
    
    def _load_history(self) -> Dict[str, Any]:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading history: {str(e)}")
                return {}
        return {}
    
    def _save_history(self) -> None:
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving history: {str(e)}")
    
    def add_history(self, config: Dict[str, Any], performance: Dict[str, Any]) -> None:
        entry = {
            'config': config,
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        }
        
        if 'entries' not in self.history:
            self.history['entries'] = []
        
        self.history['entries'].append(entry)
        self._save_history()
    
    def recommend_config(self, task_type: str, performance_target: Dict[str, Any] = None) -> Dict[str, Any]:
        relevant_entries = []
        if 'entries' in self.history:
            for entry in self.history['entries']:
                if 'config' in entry and 'task_type' in entry['config']:
                    if entry['config']['task_type'] == task_type:
                        relevant_entries.append(entry)
        
        if not relevant_entries:
            return self._get_default_config(task_type)
        
        if performance_target:
            def score_entry(entry):
                score = 0
                for key, target in performance_target.items():
                    if key in entry['performance']:
                        if key == 'latency':
                            score += 1.0 / (entry['performance'][key] / target)
                        else:
                            score += entry['performance'][key] / target
                return score
            
            relevant_entries.sort(key=score_entry, reverse=True)
        else:
            def comprehensive_score(entry):
                score = 0
                if 'accuracy' in entry['performance']:
                    score += entry['performance']['accuracy'] * 0.4
                if 'latency' in entry['performance']:
                    score += (1.0 / entry['performance']['latency']) * 0.3
                if 'memory_usage' in entry['performance']:
                    score += (1.0 / entry['performance']['memory_usage']) * 0.3
                return score
            
            relevant_entries.sort(key=comprehensive_score, reverse=True)
        
        if relevant_entries:
            best_entry = relevant_entries[0]
            print(f"Recommended configuration based on historical data: {best_entry['config']}")
            return best_entry['config']
        
        return self._get_default_config(task_type)
    
    def _get_default_config(self, task_type: str) -> Dict[str, Any]:
        default_configs = {
            'text_generation': {
                'generation': {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'max_new_tokens': 500
                }
            },
            'translation': {
                'generation': {
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'top_k': 10,
                    'max_new_tokens': 1000
                }
            },
            'summarization': {
                'generation': {
                    'temperature': 0.5,
                    'top_p': 0.9,
                    'top_k': 20,
                    'max_new_tokens': 300
                }
            },
            'question_answering': {
                'generation': {
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'top_k': 5,
                    'max_new_tokens': 200
                }
            }
        }
        
        return default_configs.get(task_type, default_configs['text_generation'])
    
    def auto_tune(self, model, task_type: str, eval_fn, max_trials: int = 10) -> Dict[str, Any]:
        param_space = {
            'temperature': [0.1, 0.3, 0.5, 0.7, 1.0],
            'top_p': [0.7, 0.8, 0.9, 0.95],
            'top_k': [5, 10, 20, 50],
            'max_new_tokens': [100, 300, 500, 1000]
        }
        
        best_config = None
        best_performance = None
        
        import random
        for trial in range(max_trials):
            config = {
                'generation': {
                    'temperature': random.choice(param_space['temperature']),
                    'top_p': random.choice(param_space['top_p']),
                    'top_k': random.choice(param_space['top_k']),
                    'max_new_tokens': random.choice(param_space['max_new_tokens']),
                    'do_sample': True
                },
                'task_type': task_type
            }
            
            try:
                performance = eval_fn(model, config)
                
                print(f"Trial {trial+1}/{max_trials}: {performance}")
                
                if best_performance is None or self._is_better(performance, best_performance, task_type):
                    best_config = config
                    best_performance = performance
                    print(f"New best configuration: {config}")
            except Exception as e:
                print(f"Error evaluating configuration: {str(e)}")
        
        if best_config and best_performance:
            self.add_history(best_config, best_performance)
            print(f"Auto-tune completed. Best configuration: {best_config}")
        
        return best_config
    
    def _is_better(self, performance1: Dict[str, Any], performance2: Dict[str, Any], task_type: str) -> bool:
        if task_type == 'text_generation':
            score1 = 0
            score2 = 0
            
            if 'bleu' in performance1 and 'bleu' in performance2:
                score1 += performance1['bleu'] * 0.6
                score2 += performance2['bleu'] * 0.6
            if 'latency' in performance1 and 'latency' in performance2:
                score1 += (1.0 / performance1['latency']) * 0.4
                score2 += (1.0 / performance2['latency']) * 0.4
            
            return score1 > score2
        elif task_type == 'translation':
            return performance1.get('bleu', 0) > performance2.get('bleu', 0)
        elif task_type == 'summarization':
            return performance1.get('rouge', 0) > performance2.get('rouge', 0)
        else:
            score1 = 0
            score2 = 0
            
            if 'accuracy' in performance1 and 'accuracy' in performance2:
                score1 += performance1['accuracy'] * 0.5
                score2 += performance2['accuracy'] * 0.5
            if 'latency' in performance1 and 'latency' in performance2:
                score1 += (1.0 / performance1['latency']) * 0.5
                score2 += (1.0 / performance2['latency']) * 0.5
            
            return score1 > score2