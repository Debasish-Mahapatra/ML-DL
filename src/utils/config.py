"""
Configuration management utilities.
"""

import os
import yaml
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_name: str) -> DictConfig:
        """Load configuration from YAML file."""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        return OmegaConf.create(config_dict)
    
    def load_all_configs(self) -> DictConfig:
        """Load all configuration files and merge them."""
        configs = {}
        
        # Load main configs
        for config_file in ["model_config", "data_config", "training_config"]:
            try:
                config_data = self.load_config(config_file)
                # Extract the main section (model, data, training)
                if config_file == "model_config" and "model" in config_data:
                    configs["model"] = config_data["model"]
                elif config_file == "data_config" and "data" in config_data:
                    configs["data"] = config_data["data"]
                elif config_file == "training_config" and "training" in config_data:
                    configs["training"] = config_data["training"]
                else:
                    # If no nested structure, use the whole config
                    section_name = config_file.replace("_config", "")
                    configs[section_name] = config_data
            except FileNotFoundError:
                print(f"Warning: {config_file}.yaml not found, using defaults")
                
        return OmegaConf.create(configs)
    
    def save_config(self, config: DictConfig, save_path: str):
        """Save configuration to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w") as f:
            OmegaConf.save(config, f)
    
    def validate_config(self, config: DictConfig) -> bool:
        """Validate configuration parameters."""
        # Add validation logic here
        required_keys = [
            "data.root_dir",
            "model.name", 
            "training.max_epochs"
        ]
        
        for key in required_keys:
            if not OmegaConf.select(config, key):
                raise ValueError(f"Required configuration key missing: {key}")
                
        return True

def get_config(config_dir: str = "config") -> DictConfig:
    """Convenience function to load all configs."""
    manager = ConfigManager(config_dir)
    config = manager.load_all_configs()
    manager.validate_config(config)
    return config
