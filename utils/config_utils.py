"""Utility functions for handling configuration.

This module provides utility functions for loading, printing, and managing
configuration files and settings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
            The file should contain dataset, model, training, and logging sections.
    
    Returns:
        dict: Loaded configuration dictionary.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the config file is invalid.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data_config(config_path: str, dataset_name: str) -> Dict[str, Any]:
    """Load dataset configuration from the data config file.
    
    Args:
        config_path (str): Path to the data configuration file.
        dataset_name (str): Name of the dataset to load configuration for.
    
    Returns:
        dict: Dataset configuration dictionary.
        
    Raises:
        ValueError: If the dataset is not found in the config file.
    """
    with open(config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    if dataset_name not in data_config['datasets']:
        raise ValueError(f"Dataset {dataset_name} not found in {config_path}")
    
    return data_config['datasets'][dataset_name]

def print_config(title: str, config: dict):
    """Print configuration in a formatted way.
    
    Args:
        title (str): Title of the configuration section
        config (dict): Configuration dictionary to print
    """
    print("\n" + "="*50)
    print(f"{title}:")
    print("="*50)
    
    def _print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("\n" + " "*indent + f"{key}:")
                _print_dict(value, indent + 2)
            else:
                print(" "*indent + f"{key}: {value}")
    
    _print_dict(config)
    print("="*50 + "\n") 