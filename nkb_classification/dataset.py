import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A


class Transforms:
    def __init__(
        self,
        transforms: A.Compose,
    ) -> None:
        
        self.transforms = transforms

    def __call__(
        self,
        img,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        
        return self.transforms(image=np.array(img))['image']


class CropsClassificationDataset(Dataset):
    def __init__(self, root: str, label_names: list, transform=None):
        self.root = root
        self.labels = pd.read_csv(Path(root, 'labels.csv'))
        self.label_to_idx = {label: idx for idx, label in enumerate(label_names)}
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.labels.loc[idx, 'img_name']
        img_path = str(Path(self.root, 'images', img_name))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.label_to_idx[self.labels.loc[idx, 'label']]
        if self.transform is not None:
            return self.transform(img), label
        return img, label


def get_dataset(data: dict, pipeline: dict) -> DataLoader:
    transform = Transforms(pipeline)
    dataset = CropsClassificationDataset(
        data['root'],
        data['label_names'],
        transform
    )
    loader = DataLoader(dataset, batch_size=data['batch_size'], 
                        shuffle=data['shuffle'], num_workers=data['num_workers'], pin_memory=True)
    return loader
