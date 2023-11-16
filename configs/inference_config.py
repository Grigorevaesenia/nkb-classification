import albumentations as A
from albumentations.pytorch import ToTensorV2
from os.path import split
import cv2

compile = False

device = 'cuda:1'

save_path = '/home/denis/nkbtech/inference/crop_classification/./2023-11-15T15_21_46'

label_names = ['true', 'false']

img_size = 224

inference_pipeline = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])

inference_data = {
    'root': '/home/alexander/nkbtech/petsearch/repos/ps_db_writer/logs/debug/./2023-11-15T15_21_46',
    'shuffle': False,
    'batch_size': 64,
    'num_workers': 8,
}

model = {
    'model': 'mobilenetv3_large_100',
    'pretrained': False,
    'backbone_dropout': 0.0,
    'classifier_dropout': 0.0,
    'checkpoint': '/home/denis/src/project/models/false_positive_classification/ds3_mobilenetv3_large_100_focal_v4/last.pth'
}
