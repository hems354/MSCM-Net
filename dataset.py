import json
import random
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
     RandFlipd,
      RandAffined, 
    RandGaussianNoised,
    RandScaleIntensityd,
    Transform,
    # Rand
    ScaleIntensityd)
from monai.utils import set_determinism
set_determinism(seed=42)


class CustomRandAugmentd(Transform):
    def __init__(self, keys, prob=0.5):
        super().__init__()
        self.keys = keys
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            img = d[key]  # shape: C, D, H, W
            x = img.squeeze(0).cpu().numpy()  # D, H, W

            # ========== random flip ==========
            if random.random() < self.prob:
                axis = random.choice([0,1,2])
                x = np.flip(x, axis=axis).copy()

            # ========== random rotate ==========
            if random.random() < self.prob:
                axes = random.choice([(0,1), (0,2), (1,2)])
                k = random.choice([1, -1])
                x = np.rot90(x, k=k, axes=axes).copy()

            # ========== random  intensity ==========
            if random.random() < self.prob:
                scale = 1.0 + random.uniform(-0.1, 0.1)
                x = x * scale

            # ========== random gaussian ==========
            if random.random() < self.prob:
                noise = np.random.normal(0, 0.01, size=x.shape)
                x = x + noise

            # 转回 tensor
            d[key] = x

        return d

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
                ScaleIntensityd(keys=["img"]),
                Resized(keys=["img"], spatial_size=[256, 256, 256]),
                CustomRandAugmentd(keys=["img"], prob=0.5)
            ],
        )


        self.val_transforms = Compose(
            [
                LoadImaged(keys=["img"], ensure_channel_first=True),
                ScaleIntensityd(keys=["img"]),
                Resized(keys=["img"], spatial_size=[256, 256, 256]),
            ]
        )

    def get_train_dataset(self):
        return monai.data.Dataset(data=self.data[self.fold]["train"], transform=self.train_transforms)
    
    def get_val_dataset(self):
        return monai.data.Dataset(data=self.data[self.fold]["val"], transform=self.val_transforms)
    

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = "./mRSdata"
    dataset = LungCancerDataset(fold=0, data_dir=data_dir)
    train_ds = dataset.get_train_dataset()
    val_ds = dataset.get_val_dataset()

    print("训练集样本数量:", len(train_ds))
    print("验证集样本数量:", len(val_ds))

    # ===================== 方法1：直接 for 遍历 dataset（推荐） =====================
    print("\n===== 直接遍历训练集 =====")
    for idx, data in enumerate(train_ds):
        img = data["img"]  # 取出图像 tensor
        print(f"样本 {idx} | shape: {img.shape} | 取值范围: [{img.min():.4f}, {img.max():.4f}]")
        
        # 只看前3个，避免刷屏
        if idx >= 2:
            break

    # ===================== 方法2：构建 DataLoader 批量读取（训练用） =====================
    print("\n===== DataLoader 批量读取 =====")
    train_loader = DataLoader(
        train_ds,
        batch_size=2,    # 批次大小
        shuffle=True,    # 训练必须打乱
        num_workers=0    # Windows设0，Linux可设2/4
    )

    # 遍历 dataloader
    for batch_idx, batch_data in enumerate(train_loader):
        batch_img = batch_data["img"]  # shape: [B, C, D, H, W]
        print(f"批次 {batch_idx} | batch shape: {batch_img.shape}")
        
        if batch_idx >= 1:
            break
