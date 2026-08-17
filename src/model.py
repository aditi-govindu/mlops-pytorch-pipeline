import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes: int = 10, pretrained: bool = True, num_channels: int = 3) -> nn.Module:
    '''Function to get a ResNet18 model, optionally pretrained on ImageNet, 
    and adapted for the specified number of classes and input channels.'''

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    
    # Adapt first layer if single-channel input (e.g., FashionMNIST).
    if num_channels == 1:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    # Replace the final fully connected layer.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model