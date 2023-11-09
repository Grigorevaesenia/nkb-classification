import os
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import cv2
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
    

class InferDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.img_files = sorted(os.listdir(os.path.join(root, 'images')))
        
        crop_files = sorted(os.listdir(os.path.join(root, 'labels')))
        self.crops = []
        for filename in crop_files:
            with open(os.path.join(root, 'labels', filename), 'r') as crp:
                self.crops.append(crp.read().split()[:4])

        img_files_ = np.array([*map(lambda x: x.split('.')[0], self.img_files)])
        crop_files_ = np.array([*map(lambda x: x.split('.')[0], crop_files)])
        assert (img_files_ != crop_files_).sum() == 0, 'Filenames do not match'

        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.root, 'images', img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop = np.array(self.crops[idx]).astype(np.int32)
        xlt, ylt, xrb, yrb = crop
        img = img[ylt:yrb, xlt: xrb]
        if self.transform is not None:
            return self.transform(img), crop, img_path
        return img, crop, img_path


def get_dataset(data: dict, pipeline: dict) -> DataLoader:
    transform = Transforms(pipeline)
    dataset = CropsClassificationDataset(
        data['root'],
        data['label_names'],
        transform
    )
    loader = DataLoader(dataset,
                        batch_size=data['batch_size'], 
                        shuffle=data['shuffle'],
                        num_workers=data['num_workers'],
                        pin_memory=True)
    return loader


def get_inference_dataset(data: dict, pipeline: dict) -> DataLoader:
    transform = Transforms(pipeline)
    dataset = InferDataset(
        data['root'],
        transform
    )
    loader = DataLoader(dataset,
                        batch_size=data['batch_size'], 
                        shuffle=data['shuffle'],
                        num_workers=data['num_workers'],
                        pin_memory=True)
    return loader
