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
from Metrics import ComputeMetrics

from Dataloader import WholeDataset,ClassConditionalPermutation
from torchmetrics import ConfusionMatrix,Accuracy,Precision,Recall
from torchmetrics.classification import MulticlassF1Score
from models.ResnetClassifier import Network

from PIL import Image
from Config import Args,config


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    args_train = Args("train")
    train_dataset = WholeDataset(args_train)
    args_test  = Args("test")
    test_dataset = WholeDataset(args_test)

    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=args_train.BATCH_SIZE,num_workers=4,pin_memory=True,persistent_workers=True)
    test_dataloader = DataLoader(test_dataset,shuffle=False,batch_size=args_test.BATCH_SIZE,num_workers=4,pin_memory=True,persistent_workers=True)

    model = Network(False)
    
    loss= nn.CrossEntropyLoss()
    loss_adv= nn.CrossEntropyLoss(reduction=None)

    compute_metrics = ComputeMetrics(10)

    classCondPerm = ClassConditionalPermutation(0.2)

    optimizer = optim.Adam([param for name,param in model.named_parameters() if param.requires_grad], lr=config.lr)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    f1 = MulticlassF1Score( num_classes=10).to(device)
    acc = Accuracy(task="multiclass", num_classes=10).to(device)
    pre = Precision(task="multiclass", num_classes=10).to(device)
    rec = Recall(task="multiclass", num_classes=10).to(device)
    metrics_dict = {"f1":f1,"accuracy":acc,"precesion":pre,"recall":rec}

    

    if torch.cuda.is_available():
        model.to(device)

    #train loop 
    model.train()
    #tracker ={"loss":[],"accuracy":[],"train_acc":[],"test_acc":[],"f1":[],"recall":[],"precesion":[]}
    tracker = {}    
    
    for e in tqdm(range(config.epochs)):
        l_sum = 0.0
        train_count = len(train_dataloader)
        test_count = len(test_dataloader)
        train_acc = 0
        test_acc = 0

        for name, metric in metrics_dict.items():
            tracker[name] =0
            tracker["test_"+name] =0
            

        for batch_num,batch_data in enumerate(iter(train_dataloader)):
            X_t,l_t,y_t,s_t  = batch_data

            log = 0
            image_array = []
            #X_t = classCondPerm(X_t,y_t)

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
                for name, metric in metrics_dict.items():
                    tracker[name] += metric(y_hat,y_t)

    
        #test metrics
        for test_batch_data in iter(test_dataloader):
            X_v,l_v,y_v,s_v  = test_batch_data
            X_v,l_v,y_v,s_v = X_v.to(device),l_v.to(device),y_v.to(device),s_v.to(device)
            with torch.no_grad():
                y_hat = model.predict(X_v)    
                test_acc+=accuracy(y_v, y_hat)
                for name, metric in metrics_dict.items():
                    tracker["test_"+name] += metric(y_hat,y_v)
        
        metrics = {"test_acc":test_acc/test_count,"loss":l_sum/train_count,"train_acc":train_acc/train_count}
        for name, metric in tracker.items():
            
            if "test" in name:
                metrics[name] =tracker[name] / test_count
            else:
                metrics[name] =tracker[name] / train_count
        print(metrics)
            
        wandb.log(metrics)
    
