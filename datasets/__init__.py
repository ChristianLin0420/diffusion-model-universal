from .mnist_loader import MNISTDataset
from .cifar10_loader import CIFAR10Dataset
from .celeba_loader import CelebADataset

DATASET_REGISTRY = {
    'mnist': MNISTDataset,
    'cifar10': CIFAR10Dataset,
    'celeba': CelebADataset
}

__all__ = [
    'MNISTDataset',
    'CIFAR10Dataset',
    'CelebADataset',
    'DATASET_REGISTRY'
] 