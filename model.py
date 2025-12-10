import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

def create_model(pretrained: bool = True, num_classes: int = 4) -> nn.Module:
    if pretrained:
        weights = EfficientNet_B3_Weights.DEFAULT
        model = efficientnet_b3(weights=weights)
    else:
        model = efficientnet_b3(weights=None)

    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model

def find_last_conv_layer(model: nn.Module) -> nn.Module:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last
