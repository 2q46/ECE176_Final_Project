import json
import torch

import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class MRI_Dataset(Dataset):

    def __init__(self, in_features: torch.Tensor, out_features: torch.Tensor, device: torch.device) -> None:

        self.in_features = in_features
        self.out_features = out_features

    def __getitem__(self, index) -> torch.Tensor:

        return self.in_features[index], self.out_features[index]
    
    def __len__(self) -> int:

        return self.in_features.shape[0]       


def train_loop(model, train_dataloader : DataLoader, loss_fn, optimizer: optim, device) -> float:

    total_loss = 0.0
    size = len(train_dataloader)

    for batch, (x, y) in tqdm(enumerate(train_dataloader), desc="Training Loop"):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            
            pred = model(x)
            loss = loss_fn(pred, y)

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # print(f"Batch {batch} Loss {batch_loss}")

    
    return total_loss/size

def inference_loop(model, test_dataloader : DataLoader, loss_fn, device):

    total_loss = 0.0
    size = len(test_dataloader)

    for batch, (x, y) in tqdm(enumerate(test_dataloader), desc="Evaluation Loop"):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        with torch.no_grad():

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
              
                pred = model(x)
                loss = loss_fn(pred, y)
            
            total_loss += loss.item()

        # print(f"Batch {batch} Loss {batch_loss}")
        
    return total_loss/size