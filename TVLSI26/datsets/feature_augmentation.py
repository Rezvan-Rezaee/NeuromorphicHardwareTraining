from skimage.feature import hog
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

from TVLSI26.configs.config import modelConstants


def normalize_features(x):
    """Min-max normalize each feature vector independently."""
    min_vals = x.min(axis=1, keepdims=True)
    max_vals = x.max(axis=1, keepdims=True)
    return (x - min_vals) / (max_vals - min_vals + 1e-8)

def generate_hog_features(mnist_train, mnist_test, index0, index1, pixels_cell, FULL_MNIST = False):
    def process(dataset):
        data, targets = [], []
        for x, y in dataset:
            y = y.item() if isinstance(y, torch.Tensor) else y
            if FULL_MNIST or y in [index0, index1]:
                #print(x.shape)
                image = x.squeeze().numpy()  # [28, 28]
                features = hog(image, pixels_per_cell=(pixels_cell, pixels_cell), cells_per_block=(2, 2), visualize=False)
                data.append(features)

                if FULL_MNIST:
                    targets.append(y)
                else:
                    targets.append(0 if y == index0 else 1) 
        return np.array(data), np.array(targets)

    X_train, y_train = process(mnist_train)
    X_test, y_test = process(mnist_test)

    # Apply per-sample normalization
    X_train = normalize_features(X_train)
    X_test = normalize_features(X_test)

    return X_train, X_test, y_train, y_test

def generate_pca_features(mnist_train, mnist_test, index0 , index1, n_components=100, FULL_MNIST = False):
    def process(dataset):
        data, targets = [], []
        for x, y in dataset:
            y = y.item() if isinstance(y, torch.Tensor) else y

            if FULL_MNIST or y in [index0, index1]:
                data.append(x.numpy())

                if FULL_MNIST:
                    targets.append(y)  # Keep original digit label
                else:
                    targets.append(0 if y == index0 else 1)  # Binary classification
        return np.array(data), np.array(targets)

    # Process data
    X_train, y_train = process(mnist_train)
    X_test, y_test = process(mnist_test)

    # PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Normalize per sample
    X_train_norm = normalize_features(X_train_pca)
    X_test_norm = normalize_features(X_test_pca)

    return X_train_norm, X_test_norm, y_train, y_test

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon

def generate_autoencoder_features(mnist_train, mnist_test, index0, index1, device, latent_dim=100, epochs=20, batch_size=64, FULL_MNIST = False):
    def process(dataset):
        data, targets = [], []
        for x, y in dataset:
            y = y.item() if isinstance(y, torch.Tensor) else y

            if FULL_MNIST or y in [index0, index1]:
                data.append(x.numpy())

                if FULL_MNIST:
                    targets.append(y)  # Keep original digit label
                else:
                    targets.append(0 if y == index0 else 1)  # Binary classification
        return np.array(data), np.array(targets)

    X_train, y_train = process(mnist_train)
    X_test, y_test = process(mnist_test)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    # Train autoencoder
    model = Autoencoder(input_dim=784, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            x = batch[0].to(device)
            _, recon = model(x)
            loss = loss_fn(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Inference
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        Z_train = model.encoder(X_train_tensor).cpu().numpy()
        Z_test = model.encoder(X_test_tensor).cpu().numpy()

    # Normalize latent features
    Z_train = normalize_features(Z_train)
    Z_test = normalize_features(Z_test)

    return Z_train, Z_test, y_train, y_test

class augmentedDataset(torch.utils.data.Dataset):
    def __init__(self, index0, index1, device, augmentation_type = "HOG", FULL_MNIST = True):
        flatten = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 -> 784
        ])

        tensor = transforms.Compose([
            transforms.ToTensor()
        ])

        hog_pixel_cell = 10

        if augmentation_type == "HOG":
            mnist_train = datasets.MNIST(modelConstants.data_path, train=True, download=True, transform=tensor)
            mnist_test = datasets.MNIST(modelConstants.data_path, train=False, download=True, transform=tensor)
            X_train , X_test, y_train, y_test = generate_hog_features(mnist_train, mnist_test, index0, index1, hog_pixel_cell, FULL_MNIST = FULL_MNIST) 
        elif augmentation_type == "PCA":
            mnist_train = datasets.MNIST(modelConstants.data_path, train=True, download=True, transform=flatten)
            mnist_test = datasets.MNIST(modelConstants.data_path, train=False, download=True, transform=flatten)
            X_train , X_test, y_train, y_test = generate_pca_features(mnist_train, mnist_test, index0, index1, 200, FULL_MNIST = FULL_MNIST)
        elif augmentation_type == "AUTO_ENCODER":
            mnist_train = datasets.MNIST(modelConstants.data_path, train=True, download=True, transform=flatten)
            mnist_test = datasets.MNIST(modelConstants.data_path, train=False, download=True, transform=flatten)
            X_train , X_test, y_train, y_test = generate_autoencoder_features(mnist_train, mnist_test, index0, index1, device, latent_dim = 400, epochs=50, FULL_MNIST = FULL_MNIST)

        mnist_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
        mnist_test = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))

        self.train_data = mnist_train
        self.test_data = mnist_test
        self.features = X_train
        self.labels = y_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def get_train_data(self):
        return self.train_data
    
    def get_test_data(self):
        return self.test_data