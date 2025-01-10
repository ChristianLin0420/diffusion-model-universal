"""Dataset Registry.

This module contains the registry of available datasets for the diffusion models.
It is separated to avoid circular imports.
"""

from .mnist_loader import MNISTDataset
from .cifar10_loader import CIFAR10Dataset
from .celeba_loader import CelebADataset

# Registry of available datasets
DATASET_REGISTRY = {
    'mnist': MNISTDataset,
    'cifar10': CIFAR10Dataset,
    'celeba': CelebADataset
} 