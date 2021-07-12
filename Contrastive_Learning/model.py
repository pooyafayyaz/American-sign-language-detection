import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.models as models
from torch import nn

     
class Contrastive(torch.nn.Module):
  def __init__(self,mlp_hidden_size,projection_size):
  
    super(Contrastive, self).__init__()
    self.resnet = models.resnet50(pretrained=True) 
    out_feature_size = self.resnet.fc.in_features
    self.resnet.fc = nn.Identity()

    self.metric = nn.Sequential(
            nn.Linear(out_feature_size, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size))
    
  def forward(self, x):
  
    x = self.resnet(x)
    return self.metric(x)


