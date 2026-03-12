import json

import torch
import numpy as np
import monai
from sklearn.model_selection import KFold
from monai.data import decollate_batch, DataLoader
from monai.transforms import (
    Activations, 
    AsDiscrete, 
    Compose, 
    LoadImaged, 
    RandRotate90d, 
    Resized,
    ToTensord,
    EnsureTyped,
    ScaleIntensityd)


class LungCancerDataset():
    def __init__(self, fold, data_dir):
        self.fold = fold
        self.data_dir = data_dir
        self.data = []
        self.train_transforms = None
        self.val_transforms = None
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)

        self.prepare_data()
        self.setup()

    def prepare_data(self):
        with open(f"{self.data_dir}/data.json", "r") as f:
            data_list = json.load(f)
        for i, (train_index, val_index) in enumerate(self.kf.split(data_list)):
            train_keys = np.array(data_list)[train_index]
            val_keys = np.array(data_list)[val_index]
            self.data.append({"train": list(train_keys), "val": list(val_keys)})

    def setup(self):
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["img"], ensure_channel_first=True),
                # EnsureTyped(keys=["img"], track_meta=False),
                ScaleIntensityd(keys=["img"]),
                # Resized(keys=["img"], spatial_size=[48, 256, 256]),
                # Resized(keys=["img"], spatial_size=[224, 224, 224]),
                Resized(keys=["img"], spatial_size=[256, 256, 256]),
            ]
        )
        self.val_transforms = Compose(
            [
                LoadImaged(keys=["img"], ensure_channel_first=True),
                # EnsureTyped(keys=["img"], track_meta=False),
                ScaleIntensityd(keys=["img"]),
                # Resized(keys=["img"], spatial_size=[48, 256, 256]),
                # Resized(keys=["img"], spatial_size=[224, 224, 224]),
                Resized(keys=["img"], spatial_size=[256, 256, 256]),
            ]
        )

    def get_train_dataset(self):
        return monai.data.Dataset(data=self.data[self.fold]["train"], transform=self.train_transforms)
    
    def get_val_dataset(self):
        return monai.data.Dataset(data=self.data[self.fold]["val"], transform=self.val_transforms)
    

if __name__ == "__main__":
    data_dir = "./mRSdata"
    dataset = LungCancerDataset(fold=0, data_dir=data_dir)
    train_ds = dataset.get_train_dataset()
    print(len(train_ds))
    print(train_ds[0]["img"].shape)
