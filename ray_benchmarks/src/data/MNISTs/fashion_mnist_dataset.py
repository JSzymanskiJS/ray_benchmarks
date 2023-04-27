import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class FashionMNISTDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform=None):
        self.labels = torch.tensor(data['label'].values)
        self.images = torch.reshape(
            torch.tensor(data.drop(columns=['label']).values, dtype=torch.float),
            (-1, 1, 28, 28),
        )
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]


def concatenate_datasets_and_save(train_csv_path: str, test_csv_path: str, full_csv_path: str):
    train_csv = pd.read_csv(train_csv_path)
    test_csv = pd.read_csv(test_csv_path)
    pd.concat([train_csv, test_csv], axis=0).to_csv(full_csv_path)


def concat_csvs(train_csv_path: str, test_csv_path: str) -> pd.DataFrame:
    print(os.path.abspath(train_csv_path))
    print(str(os.path.abspath(train_csv_path)))
    print(train_csv_path)

    train_csv = pd.read_csv(train_csv_path)
    test_csv = pd.read_csv(test_csv_path)
    return pd.concat([train_csv, test_csv], axis=0)



def split_datasets_and_save(path_to_dataset: str, full_csv_name: str, first_half_csv_name: str,
                            second_half_csv_name: str):
    full_csv = pd.read_csv(path_to_dataset + full_csv_name)
    full_csv[:int(len(full_csv) / 2)].to_csv(path_to_dataset + first_half_csv_name, index=False)
    full_csv[int(len(full_csv) / 2):].to_csv(path_to_dataset + second_half_csv_name, index=False)


if __name__ == "__main__":
    print("Wrong main file.")