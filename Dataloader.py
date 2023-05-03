
#channel shuffle permutations
#albumentations random jitter
# random array 0-255 map it to different number.


#load the data.
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn 
import os,wget
from Config import Args,config
import random

Dir_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.isdir(Dir_path+"/"+config.DATA_DIR):
    os.mkdir(Dir_path+"/"+config.DATA_DIR)
    wget.download("https://drive.google.com/u/0/uc?id=11K-GmFD5cg3_KTtyBRkj9VBEnHl-hx_Q&export=download&confirm=t&uuid=4a570521-ee16-4f7b-ba39-bd335d53742f&at=ANzk5s470DAfmxouooYvWvjjuTIM:1682277833863",out = Dir_path+"/"+config.DATA_DIR) 



class BaseModel(nn.Module):
    def __init__(self,freeze=True):
        super(BaseModel, self).__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model  = torch.nn.Sequential(*list(self.base_model.children())[:-1])
        if freeze:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self,X):
        return torch.flatten(self.base_model(X), 1)
    def save_weights(self,X,y,s,loc="ResNetweights.pt"):
        based_model_weights = {"X":self(X),"y":y,"s":s}
        torch.save(based_model_weights,f"{Dir_path}/"+loc)
    def load_weights(self,loc="ResNetweights.pt"):
        return torch.load(f"{Dir_path}/"+loc)
    
        
class ResetWeigtsDataset(Dataset):
    def __init__(self,loc="ResNetweights.pt",type="train"):
        self.base_model = BaseModel()
        if not os.path.exists(f"{Dir_path}/"+loc):
            print("resnet model weights dont exists, Working on computing resnet features.")
            args_t = Args("train")
            u_dataset = WholeDataset(args_t)
            train_dataloader = DataLoader(u_dataset,shuffle=True,batch_size=len(u_dataset),num_workers=4,pin_memory=True,persistent_workers=True)
            X,l,y,s  = next(iter(train_dataloader))
            self.base_model.save_weights(X,y,s,loc)

        based_model_weights = self.base_model.load_weights(loc)
        self.X = based_model_weights["X"]
        self.y = based_model_weights["y"]
        self.s = based_model_weights["s"]


    def __getitem__(self,index):
        return self.X[index],self.y[index],self.s[index]

    def __len__(self):
        return self.X.shape[0]
    
    
class WholeDataset(Dataset):
    def __init__(self,option):
        self.data_split = option.data_split

        
        data_dic = np.load(f"{Dir_path}/{option.data_dir}/mnist_10color_jitter_var_{option.color_var:.3f}.npy",encoding='latin1',allow_pickle=True).item()
        if self.data_split == 'train':
            self.label = data_dic['train_label']
            self.image = data_dic['train_image']
        elif self.data_split == 'test':
            self.image = data_dic['test_image']
            self.label = data_dic['test_label']

        color_var = option.color_var
        self.color_std = color_var**0.5
        self.T = transforms.Compose([
                              transforms.ToTensor(),
                                    ])
        """
            transforms.Normalize((0.4914,0.4822,0.4465),
                        (0.2023,0.1994,0.2010)),
        """
        self.ToPIL = transforms.Compose([
                              transforms.ToPILImage(),
                              ])


    def __getitem__(self,index):
        label = torch.tensor(self.label[index],dtype=torch.long)
        image = self.image[index]

        image = self.ToPIL(image)

        label_image = image.resize((14,14), Image.Resampling.NEAREST) 

        label_image = torch.from_numpy(np.transpose(label_image,(2,0,1)))
        mask_image = torch.lt(label_image.float()-0.00001, 0.) * 255
        label_image = torch.div(label_image,32)
        label_image = label_image + mask_image
        label_image = label_image.long()
        c_bias = torch.tensor([(set(np.unique(label_image[0])) - set([255])).pop(),(set(np.unique(label_image[1])) - set([255])).pop(),(set(np.unique(label_image[2])) - set([255])).pop()],dtype=torch.float)
        return self.T(image), label_image, label, c_bias

    def __len__(self):
        return self.image.shape[0]
    
class ClassConditionalPermutation:
    def __init__(self,gamma):
        self.gamma = gamma
    def __call__(self,X,y):
        og_shape = X.shape
        X = X.flatten(start_dim=1)
        for c in np.unique(y):
            arr = np.eye(X[y==c].shape[0],dtype=np.float32)
            P = np.array([np.random.permutation(arr) if random.uniform(0,1)<self.gamma else arr for i  in range(X.shape[1]) ]) 
            X[y==c] = torch.tensor(np.einsum('ijk,ki -> ji', P, np.array(X)))
        X = X.reshape(*og_shape)
        return X
            

