"""Dataset package for diffusion models.

This package contains dataset implementations and utilities for loading
and preprocessing data for diffusion models.
"""

from .mnist_loader import MNISTDataset
from .cifar10_loader import CIFAR10Dataset
from .celeba_loader import CelebADataset
from .dataset_utils import get_dataset, create_transforms

# Registry of available datasets
DATASET_REGISTRY = {
    'mnist': MNISTDataset,
    'cifar10': CIFAR10Dataset,
    'celeba': CelebADataset
}

__all__ = [
    'DATASET_REGISTRY',
    'get_dataset',
    'create_transforms',
    'MNISTDataset',
    'CIFAR10Dataset',
    'CelebADataset'
] 