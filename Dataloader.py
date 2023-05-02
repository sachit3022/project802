
#channel shuffle permutations
#albumentations random jitter
# random array 0-255 map it to different number.


#load the data.
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class WholeDataset(Dataset):
    def __init__(self,option):
        self.data_split = option.data_split
        data_dic = np.load(f"{option.data_dir}/mnist_10color_jitter_var_{option.color_var:.3f}.npy",encoding='latin1',allow_pickle=True).item()
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
                              transforms.Normalize((0.4914,0.4822,0.4465),
                                                   (0.2023,0.1994,0.2010)),
                                    ])

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