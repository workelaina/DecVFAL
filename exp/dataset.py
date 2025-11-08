import os
from typing import Iterable, Tuple, Literal
from PIL import Image

import torch
from torch import Tensor
import torch.utils
from torch.utils.data import DataLoader, Dataset
import torch.utils.data
from torchvision import datasets, transforms

DATA_DIR = '../data/'


class DfDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
        return x, y


class Xray(Dataset):
    def __init__(self, transform):
        self.data_dir = os.path.join(DATA_DIR, 'xray')
        self.transform = transform
        self.samples = []

        for fname in os.listdir(self.data_dir):
            if fname.endswith('.png'):
                label = 1 if '_y.png' in fname else 0
                path = os.path.join(self.data_dir, fname)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y = self.samples[idx]
        x = Image.open(img_path)
        x = self.transform(x)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
        return x, y


def get_dataset(name: str, flat: bool = False):
    if name == 'mnist':
        _l = [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]
        if flat:
            _l.append(transforms.Lambda(torch.flatten))
        trainset = datasets.MNIST(
            DATA_DIR, download=True, train=True,
            transform=transforms.Compose(_l)
        )
        testset = datasets.MNIST(
            DATA_DIR, download=True, train=False,
            transform=transforms.Compose(_l)
        )
    elif name == 'xray':
        if flat:
            trainset = Xray(transform=transforms.Compose([
                transforms.Grayscale(),
                # transforms.Resize((64, 64)),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Lambda(torch.flatten)
            ]))
            testset = Xray(transform=transforms.Compose([
                transforms.Grayscale(),
                # transforms.Resize((64, 64)),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Lambda(torch.flatten)
            ]))
        else:
            trainset = Xray(transforms.Compose([
                transforms.Grayscale(),
                # transforms.Resize((64, 64)),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]))
            testset = Xray(transforms.Compose([
                transforms.Grayscale(),
                # transforms.Resize((64, 64)),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]))
    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.4914, 0.4822, 0.4465),
            #     (0.2023, 0.1994, 0.2010)
            # )
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.4914, 0.4822, 0.4465),
            #     (0.2023, 0.1994, 0.2010)
            # )
        ])
        trainset = datasets.CIFAR10(
            DATA_DIR, download=True, train=True,
            transform=transform_train
        )
        testset = datasets.CIFAR10(
            DATA_DIR, download=True, train=False,
            transform=transform_test
        )
    elif name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.4914, 0.4822, 0.4465),
            #     (0.2023, 0.1994, 0.2010)
            # )
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.4914, 0.4822, 0.4465),
            #     (0.2023, 0.1994, 0.2010)
            # )
        ])
        trainset = datasets.CIFAR100(
            DATA_DIR, download=True, train=True,
            transform=transform_train
        )
        testset = datasets.CIFAR100(
            DATA_DIR, download=True, train=False,
            transform=transform_test
        )
    elif name == 'fer':
        _l = [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]
        dataset = datasets.FER2013(
            DATA_DIR, download=True, train=True,
            transform=transforms.Compose(_l)
        )
        trsz = int(len(dataset) * .8)
        tesz = len(dataset) - trsz
        trainset, testset = torch.utils.data.random_split(
            dataset,
            [trsz, tesz]
        )
    elif name == 'criteo':
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        # from sklearn.preprocessing import StandardScaler, OneHotEncoder
        # from sklearn.compose import ColumnTransformer

        file_path = os.path.join(DATA_DIR, 'criteo.csv')
        data = pd.read_csv(file_path)
        labels = data.iloc[:, -1].values
        features = MinMaxScaler((0, 1)).fit_transform(data.iloc[:, :-1])
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=0.2, stratify=labels
        )
        trainset = DfDataset(X_train, y_train)
        testset = DfDataset(X_test, y_test)
    elif name == 'covertype':
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        file_path = os.path.join(DATA_DIR, 'covertype.csv')
        data = pd.read_csv(file_path)
        labels = data.iloc[:, -1].values - 1
        features = MinMaxScaler((0, 1)).fit_transform(data.iloc[:, :-1])
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=0.2, stratify=labels
        )
        trainset = DfDataset(X_train, y_train)
        testset = DfDataset(X_test, y_test)
    else:
        raise ValueError(name)
    return trainset, testset


class Datasets:
    def __init__(
        self,
        name: Literal[
            'mnist', 'cifar10', 'cifar100',
            'fer', 'criteo', 'covertype', 'xray'
        ],
        batch_size: Tuple[int, int],
        num_workers: int,
        shuffle: bool = True,
        flat: bool = False
    ) -> None:
        assert name in [
            'mnist', 'cifar10', 'cifar100',
            'fer', 'criteo', 'covertype', 'xray'
        ]
        self.name = name
        self.batch_size = batch_size
        trainset, testset = get_dataset(name, flat)
        trainbs, testbs = batch_size
        self.trainloader: Iterable[Tuple[Tensor, Tensor]] = DataLoader(
            trainset,
            batch_size=trainbs,
            shuffle=shuffle,
            num_workers=num_workers
        )
        self.testloader: Iterable[Tuple[Tensor, Tensor]] = DataLoader(
            testset,
            batch_size=testbs,
            shuffle=shuffle,
            num_workers=num_workers
        )


if __name__ == '__main__':
    get_dataset('mnist')
    # get_dataset('cifar10')
