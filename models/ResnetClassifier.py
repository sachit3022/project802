import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

#channel shuffle permutations
#albumentations random jitter
# random array 0-255 map it to different number.
        
class Network(nn.Module):
    def __init__(self,freeze=True):
        super(Network, self).__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model  = torch.nn.Sequential(*list(self.base_model.children())[:-1])
        self.fc  = nn.Linear(512,10)
        if freeze:
            for param in self.base_model.parameters():
                param.requires_grad = False        
    def forward(self,X):
        return self.fc(torch.flatten(self.base_model(X), 1))
    def predict(self,X):
        return torch.argmax(self.forward(X),axis=1)

class BaseModel(nn.Module):
    pass 





class AdvNetwork(nn.Module):
    def __init__(self):
        super(AdvNetwork, self).__init__()
        self.fc  = nn.Linear(512,10)
        if freeze:
            for param in self.base_model.parameters():
                param.requires_grad = False        
    def forward(self,X):
        return self.fc(torch.flatten(self.base_model(X), 1))
    def predict(self,X):
        return torch.argmax(self.forward(X),axis=1)