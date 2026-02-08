import torch
from torchvision import datasets, transforms
from pathlib import Path
from typing import cast

class BinaryMNIST:
    def __init__(self, data_path: str = '/data', index0: int = 0, index1: int = 1, image_size: int = 28):
        self.data_path = data_path
        self.index0 = index0
        self.index1 = index1
        self.image_size = image_size

        self.transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.Grayscale(),
                    transforms.ToTensor()
        ])

        self.noisy_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomRotation(20),
                    transforms.RandomApply([transforms.GaussianBlur(5)], p=0.8),
                    transforms.RandomAffine(0, translate=(0.05, 0.05)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,))
        ])

        self.noisy_infer_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,))
        ])

        self.mnist_train = datasets.MNIST(self.data_path, train=True, download=True, transform=self.transform)
        self.mnist_test = datasets.MNIST(self.data_path, train=False, download=True, transform=self.transform)
        self.mnist_test_noisy = datasets.MNIST(self.data_path, train=False, download=True, transform=self.noisy_infer_transform)

        for dataset in (self.mnist_train, self.mnist_test, self.mnist_test_noisy):
            indices = indices = (dataset.targets == self.index0) | (dataset.targets == self.index1) 
            dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices]
            dataset.targets = torch.where(dataset.targets == self.index0, torch.tensor(0), torch.tensor(1))

    def get_image_size(self):
        return self.image_size
    
    def get_index0(self):
        return self.index0
    
    def get_index1(self):
        return self.index1

    def get_train_data(self):
        return self.mnist_train
    
    def get_test_data(self):
        return self.mnist_test
    
    def get_noisy_test_data(self):
        return self.mnist_test_noisy
    
    def set_transform(self, transform: transforms.Compose):
        self.transform = transform
        self.mnist_train.transform = transform
        self.mnist_test.transform = transform

    def set_noisy_transform(self, transform: transforms.Compose):
        self.noisy_transform = transform
        self.mnist_test_noisy.transform = transform

    def save_train_data(self) -> None:
        images = torch.stack([img for img, _ in self.mnist_train])
        labels = torch.tensor([label for _, label in self.mnist_train], dtype=torch.long)

        file_path: Path = Path(f"{self.data_path}/mnist_train_set_{self.index0}_{self.index1}.pt")
        torch.save({'images': images, 'labels': labels}, str(file_path)) # type: ignore
        print(f"✅ MNIST train set saved to: {file_path}")

    def save_test_data(self) -> None:
        images = torch.stack([img for img, _ in self.mnist_test])
        labels = torch.tensor([label for _, label in self.mnist_test], dtype=torch.long)

        file_path: Path = Path(f"{self.data_path}/mnist_test_set_{self.index0}_{self.index1}.pt")
        torch.save({'images': images, 'labels': labels}, str(file_path)) # type: ignore
        print(f"✅ MNIST test set saved to: {file_path}")