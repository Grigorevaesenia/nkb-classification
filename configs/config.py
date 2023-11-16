import albumentations as A
from albumentations.pytorch import ToTensorV2
from os.path import split
import cv2

show_full_current_loss_in_terminal = False

compile = False # Is not working correctly yet, so set to False
log_gradients = False
n_epochs = 60+1
device = 'cuda:1'
enable_mixed_presicion = True
enable_gradient_scaler = True

# true, false
label_names = [0, 1]

model_path = '/home/denis/src/project/models/false_positive_classification/ds3_mobilenetv3_large_100_focal_v4'

experiment = {
    'api_key_path': '/home/denis/nkbtech/nkb_classification/configs/comet_api_key.txt',
    'project_name': 'PetSearch_False_Positive',
    'workspace': 'dentikka',
    'auto_metric_logging': False,
    'name': split(model_path)[-1],
}

img_size = 224

train_pipeline = A.Compose([
    A.Resize(img_size, img_size),
    A.MotionBlur(blur_limit=21,
                 allow_shifted=True,
                 p=0.5),   
    A.CoarseDropout(
        max_holes=8,
        min_holes=2,
        max_height=0.3,
        min_height=0.1,
        max_width=0.3,
        min_width=0.1,
        fill_value=[0, 0.5, 1],
        p=1.,
    ),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])

# val_pipeline = train_pipeline
val_pipeline = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])

train_data = {
    'type': 'CropsClassificationDataset',
    'root': '/home/denis/nkbtech/data/false_positive_classification/true_and_false_cam_crops_v3/train/',
    'label_names': label_names,
    'shuffle': True,
    'weighted_sampling': True,
    'batch_size': 64,
    'num_workers': 8,
}   

val_data = {
    'type': 'CropsClassificationDataset',
    'root': '/home/denis/nkbtech/data/false_positive_classification/true_and_false_cam_crops_v3/val/',
    'label_names': label_names,
    'shuffle': True,
    'weighted_sampling': False,
    'batch_size': 64,
    'num_workers': 8,
}

model = {
    'model': 'mobilenetv3_large_100',
    'pretrained': True,
    'backbone_dropout': 0.5,
    'classifier_dropout': 0.5
}

optimizer = {
    'type': 'nadam',
    'lr': 1e-5,
    'weight_decay': 0.001,
    'backbone_lr': 1e-5,
    'classifier_lr': 1e-3,
}

lr_policy = {
    'type': 'multistep',
    'steps': [0, 20],
    'gamma': 0.1,
}

criterion = {
    'type': 'FocalLoss',
    'gamma': 2.0
}
