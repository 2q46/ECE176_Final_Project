import os
import glob
import random
import nibabel as nib
import numpy as np

import torch
import torch.nn.functional as F 

from nilearn import plotting
from typing import Optional
from sklearn.preprocessing import StandardScaler

train_data_path = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
val_data_path = "data/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"

def visualise_scan_data(data_path : Optional[str], is_train : bool, title="3D Scan") -> None:
    
    root_path = train_data_path if is_train else val_data_path

    plotting.plot_img(img=root_path + data_path, title=title)
    plotting.show()

def to_one_hot(array : np.ndarray, num_classes : int) -> np.ndarray:

    tensor = torch.tensor(array, dtype=torch.int64)
    one_hot_tensor = F.one_hot(tensor, num_classes=num_classes)

    return np.array(one_hot_tensor).astype(np.uint8)

def visualise_datapoint(dir_path: Optional[str], is_train : bool, title="3D Scan")-> None:
    '''
        dir_path : in the form /BraTS20_{train/val}_{number}
    '''

    scans = ["_flair.nii", "_seg.nii", "_t1.nii", "_t1ce.nii", "_t2.nii"]
    
    for scan in scans:
        visualise_scan_data(data_path=(dir_path + dir_path + scan), is_train=is_train, title=scan)

def convert_all_npy(storage_location="data/BraTS2020_npy", cropping=True):
        
    t1ce_paths = sorted(glob.glob(train_data_path + "/*/*t1ce.nii"))
    t2_paths = sorted(glob.glob(train_data_path + "/*/*t2.nii"))
    flair_paths = sorted(glob.glob(train_data_path + "/*/*flair.nii"))
    mask_paths = sorted(glob.glob(train_data_path + "/*/*seg.nii"))

    print(f"{len(t1ce_paths)} datapoints found")
    
    scaler = StandardScaler()

    for idx, (t1ce, t2, flair, mask) in enumerate(zip(t1ce_paths, t2_paths, flair_paths, mask_paths)):
            
        t1ce_img = nib.load(t1ce).get_fdata()
        t1ce_img = scaler.fit_transform(t1ce_img.reshape(-1, t1ce_img.shape[-1])).reshape(t1ce_img.shape)

        t2_img = nib.load(t2).get_fdata()
        t2_img = scaler.fit_transform(t2_img.reshape(-1, t2_img.shape[-1])).reshape(t2_img.shape)

        flair_img = nib.load(flair).get_fdata()
        flair_img = scaler.fit_transform(flair_img.reshape(-1, flair_img.shape[-1])).reshape(flair_img.shape)

        mask_img = nib.load(mask).get_fdata().astype(np.uint8)
        mask_img[mask_img==4] = 3

        # one-hot encoding each of the segmentation labels

        mask_img = to_one_hot(mask_img, 4) # 0 1 2 and 3

        combined_img = np.stack([t1ce_img, t2_img, flair_img], axis=3)
        
        if cropping:

            combined_img = combined_img[56:184, 56:184, 13:141]
            mask_img = mask_img[56:184, 56:184, 13:141]

        out_dir = os.path.join(storage_location, "data", str(idx))
        os.makedirs(out_dir, exist_ok=True)
        
        np.save(f"{storage_location}/data/{idx}/combined_scan.npy", combined_img)
        np.save(f"{storage_location}/data/{idx}/mask.npy", mask_img)

        saved_mask = np.load(f"{storage_location}/data/{idx}/mask.npy")
        saved_combined = np.load(f"{storage_location}/data/{idx}/combined_scan.npy")

        assert saved_combined.all() == combined_img.all(), "Saved combined img incorrectly"
        assert saved_mask.all() == mask_img.all(), "Saved mask img incorrectly"

    print("Saving completed.")
