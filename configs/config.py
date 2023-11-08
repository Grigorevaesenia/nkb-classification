import albumentations as A
from albumentations.pytorch import ToTensorV2
from os.path import split
import cv2

show_full_current_loss_in_terminal = False

compile = False # Is not working correctly yet, so set to False
log_gradients = False
n_epochs = 10 + 1
device = 'cuda:1'
enable_mixed_presicion = True
enable_gradient_scaler = True

label_names = ['true', 'random']

model_path = '/home/denis/src/project/models/false_positive_classification/mobilenetv3_large_100_dummy'

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
    # A.MotionBlur(blur_limit=3,
    #              allow_shifted=True,
    #              p=0.5),
    # A.RandomBrightnessContrast(
    #     brightness_limit=(-0.2, 0.2),
    #     contrast_limit=(0.1, -0.5),
    #     p=0.5,
    # ),
    # A.HueSaturationValue(hue_shift_limit=0, 
    #                     sat_shift_limit=10, 
    #                     val_shift_limit=50,
    #                     p=0.5),
    # A.RandomShadow(p=0.5),
    # A.RandomFog(fog_coef_lower=0.3, 
    #             fog_coef_upper=0.5, 
    #             alpha_coef=0.28,
    #             p=0.5),
    # A.RandomRain(p=0.5),
    # A.CoarseDropout(
    #     max_holes=4,
    #     min_holes=1,
    #     max_height=0.2,
    #     min_height=0.05,
    #     max_width=0.2,
    #     min_width=0.05,
    #     fill_value=[0, 0.5, 1],
    #     p=0.5,
    # ),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])

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
    'root': '/home/denis/nkbtech/data/false_positive_classification/true_and_random_cam_crops_v1/train',
    'label_names': label_names,
    'shuffle': True,
    'batch_size': 64,
    'num_workers': 8,
}   

val_data = {
    'type': 'CropsClassificationDataset',
    'root': '/home/denis/nkbtech/data/false_positive_classification/true_and_random_cam_crops_v1/val',
    'label_names': label_names,
    'shuffle': True,
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
    'backbone_lr': 1e-3,
    'classifier_lr': 1e-3,
}

lr_policy = {
    'type': 'multistep',
    'steps': [3, 6],
    'gamma': 0.1,
}

criterion = {
    'type': 'CrossEntropyLoss'
}
