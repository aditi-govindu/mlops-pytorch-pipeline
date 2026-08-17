# Import modules.
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(architecture: str = 'resnet18', 
              num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    '''Function to get the model architecture.
    Args:
        architecture: Model architecture to use. Currently only resnet18 is supported.
        num_classes: Number of output classes for the model. Default is 10.
        pretrained: Whether to use a pretrained model or not. Default is True.
    '''
    if architecture == 'resnet18':
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f'Unsupported architecture: {architecture}')