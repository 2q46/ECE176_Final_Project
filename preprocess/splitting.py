
# use torch sampler https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.random_split
import os
import glob
import tqdm
import numpy as np

def combine_from_npy(data_path: str = "data/BraTS2020_npy/data") -> tuple:

    '''Load the dataset from numpy into main memory'''
    features = np.ndarray(shape=(369, 128, 128, 128, 3))
    labels = np.ndarray(shape=(369, 128, 128, 128, 4))

    for idx in range(features.shape[0]):    

        combined_path = (data_path + f"/{idx}/combined_scan.npy")
        mask_path = (data_path + f"/{idx}/mask.npy")

        features[idx] = np.load(combined_path)
        labels[idx] = np.load(mask_path)

    return (features, labels)

def train_test_split(train_ratio: float = 0.8, test_ratio: float = 0.2):

    pass