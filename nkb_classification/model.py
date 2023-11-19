import torch
from torch import nn
import timm


class CropsClassificationModel(nn.Module):
    """
    A class to make a model consisting of an embedding model (backbone)
    and classifier
    Currently maintained architectures are:
        MobileNet, EfficientNet, ConvNext, ResNet, ViT
    """
    def __init__(self,
                 cfg_model: dict, 
                 label_names: list):
        super().__init__()
        self.emb_model = timm.create_model(cfg_model['model'], pretrained=cfg_model['pretrained'])
        if cfg_model['model'].startswith('mobilenet'):
            for name, child in self.emb_model.named_children():
                if name in ['conv_stem', 'bn1']:
                    for param in child.parameters():
                        param.requires_grad = False
                elif name == 'blocks':
                    for name_, child_ in child.named_children():
                        if name_ in ['0', '1', '2', '3']:
                            for param in child_.parameters():
                                param.requires_grad = False
        self.emb_size = self.emb_model.num_features
        self.emb_model.reset_classifier(0)  # a simpler way to get emb_model from a timm model
        self.set_dropout(self.emb_model, cfg_model['backbone_dropout'])

        self.classifier = nn.Sequential(
            nn.Dropout(cfg_model['classifier_dropout']),
            nn.Linear(self.emb_size, len(label_names))
        )

    def forward(self, x):
        emb = self.emb_model(x)
        return self.classifier(emb)

    def set_backbone_state(self, state: str):
        for param in self.emb_model.parameters():
            if state == 'freeze':
                param.requires_grad = False
            elif state == 'unfreeze':
                param.requires_grad = True
    
    @staticmethod
    def set_dropout(model: nn.Module, drop_rate: float = 0.2) -> None:
        '''Set new `drop_rate` for model'''
        for child in model.children():
            if isinstance(child, torch.nn.Dropout):
                child.p = drop_rate
            CropsClassificationModel.set_dropout(child, drop_rate=drop_rate)
    

def get_model(cfg_model: dict,
              label_names: list,
              device: torch.device='cpu',
              compile: bool=True) -> CropsClassificationModel:
    model = CropsClassificationModel(cfg_model, label_names)

    model.to(device)
    if compile:
        model = torch.compile(model, dynamic=True)

    return model
