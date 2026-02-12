
# use torch sampler https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.random_split
import os
import glob
import torch
import json
import numpy as np

from torch.utils.data import random_split, DataLoader
from utility.utils import MRI_Dataset

def combine_from_npy(data_path: str = "data/BraTS2020_npy/data") -> tuple:

    '''Load the dataset from numpy into main memory'''
    features = np.ndarray(shape=(369, 128, 128, 128, 3), dtype=np.float32)
    labels = np.ndarray(shape=(369, 128, 128, 128, 4), dtype=np.float32)

    for idx in range(features.shape[0]):    

        combined_path = (data_path + f"/{idx}/combined_scan.npy")
        mask_path = (data_path + f"/{idx}/mask.npy")

        features[idx] = np.load(combined_path)
        labels[idx] = np.load(mask_path)

    return (features, labels)

def train_test_split(features, labels) -> tuple:

    with open(file="config.json", encoding="utf-8", mode="r") as file:
        
        params = json.load(file)

    train_ratio = params["train ratio"]
    batch_size = params["batch size"]

    train_size = int(len(features) * train_ratio)
    test_size = len(features) - train_size

    labels = torch.tensor(labels, dtype=torch.float32)
    features = torch.tensor(features, dtype=torch.float32)

    dataset = MRI_Dataset(features, labels)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    del dataset # save memory

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

    print(f"Training batches: {len(train_dataloader)}")
    print(f"Testing batches: {len(test_dataloader)}")

    return (train_dataloader, test_dataloader)