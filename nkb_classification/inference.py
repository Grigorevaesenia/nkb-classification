from pathlib import Path
from os.path import split
import sys

import numpy as np
import pandas as pd

import torch

import argparse
from tqdm import tqdm

from nkb_classification.utils import get_inference_dataset


def inference(model, loader, 
              save_path, device):
    columns = loader.dataset.target_names.copy()
    columns.append('path')
    inference_annotations = pd.DataFrame(columns=columns)
    model.eval()
    Softmax = torch.nn.Softmax(dim=-1)
    with torch.no_grad():
        for imgs, img_paths in tqdm(loader, leave=False):
            imgs = imgs.float().to(device)
            preds = model(imgs)
            batch_annotations = []
            for target_name in loader.dataset.target_names:
                pred = preds[target_name]
                pred = list(Softmax(pred).argmax(dim=1).detach().cpu().numpy())
                pred = [loader.dataset.idx_to_class[target_name][idx] for idx in pred]
                batch_annotations.append(pred)
            batch_annotations.append(list(img_paths))
            batch_annotations = np.vstack(batch_annotations).T
            inference_annotations = pd.concat([inference_annotations,
                                            pd.DataFrame(batch_annotations,
                                                            columns=columns)])
    inference_annotations.to_csv(Path(save_path, 'inference_annotations.csv'), index=False)


def read_py_config(path):
    path = Path(path)
    sys.path.append(str(path.parent))
    line = f'import {path.stem} as cfg'
    return line


def main():
    parser = argparse.ArgumentParser(description='Inference arguments')
    parser.add_argument('-cfg', '--config', help='Config file path', type=str, default='', required=True)
    args = parser.parse_args()
    cfg_file = args.config
    exec(read_py_config(cfg_file), globals(), globals())
    data_loader = get_inference_dataset(cfg.inference_data, cfg.inference_pipeline)
    device = torch.device(cfg.device)
    model = torch.load(cfg.model['checkpoint']).to(torch.device(device))
    save_path = Path(cfg.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    inference(model, data_loader, save_path, device)

if __name__ == '__main__':
    main()
