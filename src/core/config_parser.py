import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigParser:
    """Simple configuration parser for YAML configs with CLI overrides"""
    
    def __init__(self, default_config_path: str = "config/inference_config.yaml"):
        self.default_config_path = default_config_path
        self.config = {}
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = config_path or self.default_config_path
        config_path_obj = Path(config_file)
        
        if not config_path_obj.exists():
            print(f"Config file not found: {config_file}")
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        return self.config
    
    def get(self, section: str, key: str, default=None):
        """Get configuration value with dot notation support"""
        return self.config.get(section, {}).get(key, default)
    
    def print_config(self):
        """Print current configuration"""
        print("Current Configuration:")
        print("=" * 40)
        print(yaml.dump(self.config, default_flow_style=False, indent=2))
        print("=" * 40)
