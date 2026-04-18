import json
import os
import time


class AIConfigManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self._load_config()
        self.original_config = self.config.copy()
        self.last_modified_time = (
            os.path.getmtime(self.config_file)
            if os.path.exists(self.config_file)
            else 0
        )
        self.config_version = 1

    def _load_config(self):
        try:
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(
                    f"Configuration file {self.config_file} not found"
                )

            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {}

    def get_config(self):
        self._check_for_config_changes()
        return self.config

    def _check_for_config_changes(self):
        if os.path.exists(self.config_file):
            current_modified_time = os.path.getmtime(self.config_file)
            if current_modified_time > self.last_modified_time:
                print(
                    f"Configuration file {self.config_file} has changed, reloading..."
                )
                self.config = self._load_config()
                self.last_modified_time = current_modified_time
                self.config_version += 1
                print(f"Configuration reloaded, version: {self.config_version}")

    def set_config(self, config):
        self.config = config
        self.config_version += 1

    def get_env_config(self, env="default"):
        self._check_for_config_changes()

        if "environments" in self.original_config and env in self.original_config["environments"]:
            base_config = self.original_config.copy()
            if "environments" in base_config:
                del base_config["environments"]
            env_config = self.original_config["environments"][env]
            return self.merge_configs(base_config, env_config)
        return self.config

    def switch_env(self, env="default"):
        env_config = self.get_env_config(env)
        if env_config != self.config:
            self.config = env_config
            self.config_version += 1
            print(
                f"Switched to environment: {env}, configuration version: {self.config_version}"
            )
            return True
        return False

    def validate_env_config(self, env="default"):
        env_config = self.get_env_config(env)

        required_fields = ["model", "training", "evaluation"]
        for field in required_fields:
            if field not in env_config:
                print(f"Missing required field: {field} in environment {env}")
                return False

        model_fields = [
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "random_state",
        ]
        for field in model_fields:
            if field not in env_config["model"]:
                print(f"Missing model field: {field} in environment {env}")
                return False

        training_fields = ["batch_size", "epochs", "learning_rate"]
        for field in training_fields:
            if field not in env_config["training"]:
                print(f"Missing training field: {field} in environment {env}")
                return False

        evaluation_fields = ["metrics", "validation_split"]
        for field in evaluation_fields:
            if field not in env_config["evaluation"]:
                print(f"Missing evaluation field: {field} in environment {env}")
                return False

        return True

    def list_environments(self):
        self._check_for_config_changes()
        if "environments" in self.config:
            return list(self.config["environments"].keys())
        return []

    def save_config(self):
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            print(f"Configuration saved to {self.config_file}")
            self.last_modified_time = os.path.getmtime(self.config_file)
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def export_config(self, export_path):
        try:
            self._check_for_config_changes()
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            print(f"Configuration exported to {export_path}")
        except Exception as e:
            print(f"Error exporting configuration: {e}")

    def import_config(self, import_path):
        try:
            with open(import_path, "r", encoding="utf-8") as f:
                imported_config = json.load(f)
            self.config = imported_config
            self.config_version += 1
            self.save_config()
            print(f"Configuration imported from {import_path}")
        except Exception as e:
            print(f"Error importing configuration: {e}")

    def backup_config(self, backup_dir=None):
        try:
            if backup_dir is None:
                backup_dir = os.path.dirname(self.config_file)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"config_backup_{timestamp}.json"
            backup_path = os.path.join(backup_dir, backup_filename)

            self.export_config(backup_path)
            print(f"Configuration backed up to {backup_path}")
            return backup_path
        except Exception as e:
            print(f"Error backing up configuration: {e}")
            return None

    def restore_config(self, backup_path):
        try:
            self.import_config(backup_path)
            print(f"Configuration restored from {backup_path}")
        except Exception as e:
            print(f"Error restoring configuration: {e}")

    def update_config(self, updates):
        def deep_update(target, source):
            for key, value in source.items():
                if (
                    key in target
                    and isinstance(target[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_update(target[key], value)
                else:
                    target[key] = value

        deep_update(self.config, updates)
        self.save_config()

    def validate_config(self):
        required_fields = ["model", "training", "evaluation"]
        for field in required_fields:
            if field not in self.config:
                print(f"Missing required field: {field}")
                return False

        model_fields = [
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "random_state",
        ]
        for field in model_fields:
            if field not in self.config["model"]:
                print(f"Missing model field: {field}")
                return False

        training_fields = ["batch_size", "epochs", "learning_rate"]
        for field in training_fields:
            if field not in self.config["training"]:
                print(f"Missing training field: {field}")
                return False

        evaluation_fields = ["metrics", "validation_split"]
        for field in evaluation_fields:
            if field not in self.config["evaluation"]:
                print(f"Missing evaluation field: {field}")
                return False

        return True

    def get(self, key, default=None):
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key, value):
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        self.save_config()

    def load_from_env(self, prefix="AI_"):
        import os

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace("_", ".")
                try:
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif "." in value and all(c.isdigit() or c == "." for c in value):
                        value = float(value)
                    elif value.startswith("[") and value.endswith("]"):
                        value = json.loads(value)
                    elif value.startswith("{") and value.endswith("}"):
                        value = json.loads(value)
                except Exception:
                    pass

                self.set(config_key, value)

    def merge_configs(self, base_config, override_config):
        merged = base_config.copy()

        for key, value in override_config.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged


