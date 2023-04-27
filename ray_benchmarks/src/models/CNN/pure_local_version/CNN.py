# Imports
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import nn  # All neural network modules
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch.utils.data import (
    DataLoader, Dataset,
)  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!

from ray_benchmarks.src.data.MNISTs.fashion_mnist_dataset import FashionMNISTDataset, concat_csvs

PYTORCH_MNIST_DATASET_PATH = "../../../../data/external/pytorch_MNIST_dataset/"
KAGGLE_MNIST_DATASET_PATH = "../../../../data/external/Fashion_MNIST/"


# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


def train_epoch(
        model: torch.nn,
        train_loader: DataLoader,
        loss_func: nn.CrossEntropyLoss,
        optimizer: optim.Adam,
        device: str,
        should_use_tqdm=False,
):
    if should_use_tqdm:
        train_loader = tqdm(train_loader)

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = loss_func(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


def get_trained_model(
        train_loader: DataLoader,
        in_channels: int = 1,
        num_classes: int = 10,
        learning_rate: float = 3e-4,
        num_epochs: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        should_use_tqdm=False,
) -> CNN:
    if should_use_tqdm:
        train_loader = tqdm(train_loader)

    model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
    model.train()

    # Loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(num_epochs):
        train_epoch(
            model=model,
            train_loader=train_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            device=device
        )

    return model


def time_main_computation(
        train_dataset: Dataset = None,
        test_dataset: Dataset = None,
) -> tuple[float, float, float, float]:
    before_dataset_init = datetime.now().timestamp()
    batch_size = 64
    device = "cpu"
    if not train_dataset:
        train_dataset = datasets.MNIST(
            root=PYTORCH_MNIST_DATASET_PATH,
            train=True,
            transform=transforms.ToTensor(),
            download=True
        )
    if not test_dataset:
        test_dataset = datasets.MNIST(
            root=PYTORCH_MNIST_DATASET_PATH,
            train=False,
            transform=transforms.ToTensor(),
            download=True
        )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    after_dataset_init = datetime.now().timestamp()

    model = get_trained_model(train_loader=train_loader, device=device)
    after_training = datetime.now().timestamp()

    accuracy = check_accuracy(test_loader, model=model, device=device)
    after_evaluation = datetime.now().timestamp()

    dataset_init_time = after_dataset_init - before_dataset_init
    model_train_time = after_training - after_dataset_init
    model_eval_time = after_evaluation - after_training
    return dataset_init_time, model_train_time, model_eval_time, float(accuracy)


def check_accuracy(
        loader: DataLoader,
        model: CNN,
        device: str,
        should_use_tqdm=False,
        # Zmienić na ustawianie device na ustawiane na początku - bez wartości domyślnej.
) -> float:
    if should_use_tqdm:
        loader = tqdm(loader)

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)  # Dodaj typ tej zmiennej
            y = y.to(device=device)  # Dodaj typ tej zmiennej

            scores = model(x)
            _, predictions = scores.max(1)  # Muszę sprawdzić co to ten .max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


def main(mnist_path: str = KAGGLE_MNIST_DATASET_PATH):
    train_MNIST_dataset = FashionMNISTDataset(
        concat_csvs(
            f"{mnist_path}fashion-mnist_train_1.csv",
            f"{mnist_path}fashion-mnist_train_2.csv"
        )
    )
    test_MNIST_dataset = FashionMNISTDataset(pd.read_csv(f"{mnist_path}fashion-mnist_test.csv"))
    dataset_init_t, model_train_t, model_eval_t, accuracy_ = time_main_computation(
        train_dataset=train_MNIST_dataset,
        test_dataset=test_MNIST_dataset,
    )
    print(dataset_init_t, model_train_t, model_eval_t, accuracy_)


if __name__ == "__main__":
    main()
