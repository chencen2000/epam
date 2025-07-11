import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union


class ConfigParser:
    """Enhanced configuration parser for YAML configs with nested access support"""
    
    def __init__(self, default_config_path: str = "config/training_config.yaml"):
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
    
    def get(self, section: str, key: str = None, default=None) -> Any:
        """
        Get configuration value with support for nested access
        
        Args:
            section: Top-level section (e.g., 'training', 'model')
            key: Optional key within section (e.g., 'batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if key is None:
            return self.config.get(section, default)
        return self.config.get(section, {}).get(key, default)
    
    def get_nested(self, path: str, default=None) -> Any:
        """
        Get configuration value using dot notation path
        
        Args:
            path: Dot-separated path (e.g., 'training.batch_size', 'model.architecture')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.config.get(section, {})
    
    def update(self, section: str, key: str, value: Any) -> None:
        """Update configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def print_config(self):
        """Print current configuration"""
        print("Current Configuration:")
        print("=" * 40)
        print(yaml.dump(self.config, default_flow_style=False, indent=2))
        print("=" * 40)
