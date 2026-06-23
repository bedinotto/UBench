"""
Model Registry
==============
Dynamically load and register model architectures.
"""

from typing import Type, Dict, Any
import torch.nn as nn

_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str):
    """Decorator to register a model class"""
    def decorator(cls: Type[nn.Module]):
        _MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def get_registered_models() -> list:
    """Return a list of registered model names"""
    return list(_MODEL_REGISTRY.keys())

def create_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name"""
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](**kwargs)
