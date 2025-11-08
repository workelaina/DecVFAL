import torch
from torchvision import datasets

DATA_DIR = '../data/'

datasets.MNIST(DATA_DIR, download=True, train=True)
datasets.MNIST(DATA_DIR, download=True, train=False)
datasets.CIFAR10(DATA_DIR, download=True, train=True)
datasets.CIFAR10(DATA_DIR, download=True, train=False)
datasets.CIFAR100(DATA_DIR, download=True, train=True)
datasets.CIFAR100(DATA_DIR, download=True, train=False)
# datasets.FER2013(DATA_DIR, train=True)
