import wget,os,random
from tqdm import tqdm
import matplotlib.pyplot as plt

import wandb

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn as nn

import torchmetrics


from Dataloader import WholeDataset,ClassConditionalPermutation
from models.ResnetClassifier import Network



from PIL import Image

#change the environ here

DATA_DIR = "colored_mnist"
BATCH_SIZE = 1000
LOG  = "online" #dryrun #online

os.environ["WANDB_API_KEY"]= "13978bc398bdedd79f4db560bfb4b79e2db711b5"
wandb.login()
wandb.init(
    mode=LOG,
    project="Biased MNIST",
    config={
        "epochs": 100,
        "batch_size": 1000,
        "lr": 1e-2,
        "adversery_weight":False,
        "permutation":False
    })


class Args(object):
    def __init__(self,data_split):
        self.data_dir = DATA_DIR
        self.data_split = data_split
        self.color_var = 0.040
config = wandb.config


### ARGS END

if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
    wget.download("https://drive.google.com/u/0/uc?id=11K-GmFD5cg3_KTtyBRkj9VBEnHl-hx_Q&export=download&confirm=t&uuid=4a570521-ee16-4f7b-ba39-bd335d53742f&at=ANzk5s470DAfmxouooYvWvjjuTIM:1682277833863",out = args.data_dir) 



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda:5"
    else:
        device = "cpu"

    args_train = Args("train")
    train_dataset = WholeDataset(args_train)
    args_test  = Args("test")
    test_dataset = WholeDataset(args_test)

    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=BATCH_SIZE,num_workers=4,pin_memory=True,persistent_workers=True)
    test_dataloader = DataLoader(test_dataset,shuffle=False,batch_size=BATCH_SIZE,num_workers=4,pin_memory=True,persistent_workers=True)

    model = Network(False)
    loss= nn.CrossEntropyLoss()

    classCondPerm = ClassConditionalPermutation(0.2)

    optimizer = optim.Adam([param for name,param in model.named_parameters() if param.requires_grad], lr=config.lr)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

    if torch.cuda.is_available():
        model.to(device)

    #train loop 
    model.train()
    tracker ={"loss":[],"accuracy":[],"train_acc":[],"test_acc":[]}
    for e in tqdm(range(config.epochs)):
        l_sum = 0.0
        train_count = len(train_dataloader)
        test_count = len(test_dataloader)
        train_acc = 0
        test_acc = 0

        for batch_num,batch_data in enumerate(iter(train_dataloader)):
            X_t,l_t,y_t,s_t  = batch_data

            log = 0
            image_array = []
            X_t = classCondPerm(X_t,y_t)

            X_t,l_t,y_t,s_t = X_t.to(device),l_t.to(device),y_t.to(device),s_t.to(device)

            optimizer.zero_grad()
            y_logits = model(X_t)
            l = loss(y_logits,y_t)
            l.backward()
            optimizer.step()
            
            #metrics.
            with torch.no_grad():
                l_sum+=l
                y_hat = model.predict(X_t)
                train_acc += accuracy(y_t, y_hat)

    
        #test metrics
        for test_batch_data in iter(test_dataloader):
            X_v,l_v,y_v,s_v  = test_batch_data
            X_v,l_v,y_v,s_v = X_v.to(device),l_v.to(device),y_v.to(device),s_v.to(device)
            with torch.no_grad():
                y_hat = model.predict(X_v)
                test_acc+=accuracy(y_v, y_hat)
        
        metrics = {"test_acc":test_acc/test_count,"loss":l_sum/train_count,"train_acc":train_acc/train_count}
        
        wandb.log(metrics)