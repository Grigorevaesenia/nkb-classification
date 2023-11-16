from typing import Any, Union

from pathlib import Path

import numpy as np
import pandas as pd

import torch

import argparse
from tqdm import tqdm

from nkb_classification.dataset import get_inference_dataset
from nkb_classification.model import get_model
from nkb_classification.utils import read_py_config


@torch.no_grad()
def inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader, 
    save_path: str,
    device: Union[torch.device, str]
) -> None:
    columns = ['path', 'xlt', 'ylt', 'xrb', 'yrb', 'label']
    inference_annotations = pd.DataFrame(columns=columns)

    model.eval()

    for imgs, crops, img_paths in tqdm(loader, leave=False, desc='Inference'):
        imgs = imgs.float().to(device)
        preds = model(imgs)
        batch_annotations = []
        batch_annotations.append(
            img_paths
        )
        batch_annotations.append(
            crops
            .cpu()
            .numpy()
            .T
        )
        batch_annotations.append(
            preds
            .argmax(dim=1)
            .cpu()
            .numpy()
        )
        batch_annotations = np.vstack(batch_annotations).T
        inference_annotations = pd.concat([
            inference_annotations,
            pd.DataFrame(batch_annotations,
                         columns=columns)
        ])
    inference_annotations.to_csv(Path(save_path, 'inference_annotations.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description='Inference arguments')
    parser.add_argument('-cfg', '--config', help='Config file path', type=str, default='', required=True)
    args = parser.parse_args()

    cfg_file = args.config
    exec(read_py_config(cfg_file), globals(), globals())

    # get dataloader
    data_loader = get_inference_dataset(cfg.inference_data, cfg.inference_pipeline)
    device = torch.device(cfg.device)

    # get model
    label_names = cfg.label_names
    model = get_model(cfg.model, label_names, device, compile=cfg.compile)

    # load weights
    model.load_state_dict(torch.load(cfg.model['checkpoint'], map_location='cpu'))
    model.to(device)

    save_path = Path(cfg.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    inference(model, data_loader, save_path, device)

if __name__ == '__main__':
    main()
