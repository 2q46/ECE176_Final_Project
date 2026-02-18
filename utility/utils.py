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

def init_weights(model):

    if isinstance(model, nn.Conv3d) or isinstance(model, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            nn.init.zeros_(model.bias)

def train_loop(model, train_dataloader: DataLoader, loss_fn, optimizer: optim, device, accumulation_steps: int = 4) -> float:

    total_loss = 0.0
    size = len(train_dataloader)

    optimizer.zero_grad()

    for batch, (x, y) in tqdm(enumerate(train_dataloader), desc="Training Loop"):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x)
            loss = loss_fn(pred, y)
            loss = loss / accumulation_steps
            total_loss += loss.item() * accumulation_steps
            loss.backward()

        if (batch + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Handle any remaining batches
    if (batch + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / size

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
                        
    return total_loss/size