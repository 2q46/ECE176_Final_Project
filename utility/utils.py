import json
import torch

import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class MRI_Dataset(Dataset):

    def __init__(self, in_features: torch.Tensor, out_features: torch.Tensor, device: torch.device) -> None:

        self.in_features = in_features.to(device=device)
        self.out_features = out_features.to(device=device)

    def __getitem__(self, index) -> torch.Tensor:

        return self.in_features[index], self.out_features[index]
    
    def __len__(self) -> int:

        return self.in_features.shape[0]       

def compute_accuracy():

    pass

def compute_dice_loss():

    pass

def train_loop(model, train_dataloader : DataLoader, loss_fn, optimizer: optim) -> float:

    total_loss = 0.0
    size = len(train_dataloader)

    for batch, (x, y) in tqdm(enumerate(train_dataloader), desc="Training Loop"):
        
        print(f"On batch {batch}")

        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
    
    return total_loss/size

def inference_loop(model, test_dataloader : DataLoader, loss_fn):

    total_loss = 0.0
    size = len(test_dataloader)

    for batch, (x, y) in tqdm(enumerate(test_dataloader), desc="Evaluation Loop"):
        
        print(f"On batch {batch}")

        with torch.no_grad():

            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()

    return total_loss/size