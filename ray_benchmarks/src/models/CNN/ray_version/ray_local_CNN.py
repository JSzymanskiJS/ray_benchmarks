import argparse
import os.path
from typing import Dict
from datetime import datetime

import pandas as pd
import ray
import torch
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import ray_benchmarks.src.models.CNN.pure_local_version.CNN as cnn
from ray_benchmarks.src.data.MNISTs.fashion_mnist_dataset import FashionMNISTDataset, concat_csvs
from ray_benchmarks.src.visualization.data_convention import test_result

FASHION_MNIST_PATH = "./../../../../data/external/Fashion_MNIST/"
ray_local_test_result = test_result.copy()


def train_epoch(
        loss_func: nn.CrossEntropyLoss,
        optimizer: optim.Optimizer,
        model: cnn.CNN,
        train_loader: DataLoader,
        should_use_tqdm=False,
):
    if should_use_tqdm:
        train_loader = tqdm(train_loader)

    for data, targets in train_loader:
        # forward
        scores = model(data)
        loss = loss_func(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


def train_loop(
        model: cnn.CNN,
        loss_func: nn.CrossEntropyLoss,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        num_epochs: int = 3,
        should_use_tqdm=False,
):
    for _ in range(num_epochs):
        train_epoch(
            loss_func=loss_func,
            optimizer=optimizer,
            model=model,
            train_loader=train_loader,
            should_use_tqdm=should_use_tqdm
        )


def check_accuracy(
        model: cnn.CNN,
        test_loader: DataLoader,
        should_use_tqdm=False,
) -> float:
    if should_use_tqdm:
        test_loader = tqdm(test_loader)

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


def train_model(config: Dict):
    dataset_abs_path = config['dataset_abs_path']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    should_use_tqdm = config['should_use_tqdm']

    if not should_use_tqdm:
        should_use_tqdm = False

    model = cnn.CNN()
    model.train()

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    before_dataset_init = datetime.now().timestamp()
    train_loader = DataLoader(
        FashionMNISTDataset(
            concat_csvs(
                f"{dataset_abs_path}/fashion-mnist_train_1.csv",
                f"{dataset_abs_path}/fashion-mnist_train_2.csv",
            )
        ),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        FashionMNISTDataset(
            pd.read_csv(f"{dataset_abs_path}/fashion-mnist_test.csv")
        ),
        batch_size=batch_size,
        shuffle=True
    )
    after_dataset_init = datetime.now().timestamp()
    ray_local_test_result['dataset_init_time'] = after_dataset_init - before_dataset_init

    train_loop(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        num_epochs=num_epochs,
        should_use_tqdm=should_use_tqdm
    )
    after_training = datetime.now().timestamp()
    ray_local_test_result['model_train_time'] = after_training - after_dataset_init

    ray_local_test_result['model_accuracy'] = check_accuracy(model=model, test_loader=test_loader)
    after_evaluation = datetime.now().timestamp()
    ray_local_test_result['model_eval_time'] = after_evaluation - after_training

def trainer_activation(dataset_abs_path: str, num_workers: int = 2, use_gpu=False):
    trainer = TorchTrainer(
        train_loop_per_worker=train_model,
        train_loop_config={
            "dataset_abs_path": dataset_abs_path,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "num_epochs": 3,
            "should_use_tqdm": False,
        },
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),

    )
    result = trainer.fit()
    print("result")
    print(result)


def main(smoke_test=True, address=None, num_workers=None, use_gpu=None):
    fashion_MNIST_path = os.path.abspath(FASHION_MNIST_PATH)

    if smoke_test:
        # 2 workers + 1 for trainer.
        ray.init()
        trainer_activation(fashion_MNIST_path)
    else:
        ray.init(address=address)
        trainer_activation(fashion_MNIST_path, num_workers=num_workers, use_gpu=use_gpu)


if __name__ == "__main__":
    fashion_MNIST_path = str(os.path.abspath(FASHION_MNIST_PATH))
    print(fashion_MNIST_path)
    cwd = str(os.getcwd())
    print(cwd)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", required=False, type=str, help="the address to use for Ray"
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=0,
        help="Sets number of workers for training.",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=False, help="Enables GPU training"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Finish quickly for testing.",
    )

    args, _ = parser.parse_known_args()

    main(
        address=args.address,
        num_workers=args.num_workers,
        use_gpu=args.use_gpu
    )
