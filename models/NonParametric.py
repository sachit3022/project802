

import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from Dataloader import WholeDataset,BaseModel,ResetWeigtsDataset
from torch.utils.data import  DataLoader,SubsetRandomSampler
from sklearn.neighbors import KNeighborsClassifier
from Config import config,Args
from Metrics import ComputeMetrics
from sklearn.model_selection import KFold
from tqdm import tqdm
from models.DimRed import zca_whitening_matrix
import math
from sklearn.neighbors import KernelDensity


class KNN:
    def __init__(self,k=10,whitten=False):
        self.K = k
        self.whitten = whitten
        self.zca_whitening_matrix = None
        self.knn = KNeighborsClassifier(n_neighbors=k)
    def fit(self,X,y):
        if self.whitten:
            self.zca_whitening_matrix =zca_whitening_matrix(X)
            X = (self.zca_whitening_matrix  @ X.T).T
        self.knn.fit(X,y)
    def predict(self,X):
        if self.whitten:
            X = (self.zca_whitening_matrix  @ X.T).T
        y_test_pred = self.knn.predict(X) 
        return torch.tensor(y_test_pred)


gauss_dist = lambda x : np.exp(-(x.T @ x)/2)/((2*math.pi)**(1/2))

def parzen_window(X,h):
    phi_x = lambda xi: lambda x: gauss_dist((xi - x)/h) 
    return lambda x :sum([f(x) for f in np.apply_along_axis(phi_x,1,X)]) / X.shape[0] / h


class ParzenWindow:
    def __init__(self,h=1,whitten=False):
        self.h= h
        self.whitten= whitten
        self.zca_whitening_matrix = None
        self.pxw = []

    def fit(self,X,y):
        if self.whitten:
            self.zca_whitening_matrix = zca_whitening_matrix(X)
            X = (self.zca_whitening_matrix  @ X.T).T

        X = X.numpy()
        y = y.numpy()

        self.lf= [None for i in range(len(np.unique(y)))]
        for _c in np.unique(y):
            self.lf[_c]  =  KernelDensity(kernel='gaussian', bandwidth=self.h).fit(X[y==_c]).score_samples

    def predict(self,X):
        if self.whitten:
           X = (self.zca_whitening_matrix  @ X.T).T
        X = X.numpy()
        return torch.tensor(np.apply_along_axis(np.argmax, 1, np.column_stack([score_fn_c(X) for score_fn_c in self.lf])))


if __name__ == "__main__":

    #if torch.cuda.is_available():
    #    device = "cuda:5"
    #else:
    device = "cpu"
    compute_metrics = ComputeMetrics(10)

    train_dataset = ResetWeigtsDataset("ResnetFeaturesTrain.pt","train")
    test_dataset = ResetWeigtsDataset("ResnetFeaturesTest.pt","test")

    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size

    metrics ={}

    #change k to find the right balance between them.
    #hyper parameter search

    """

    # hyper parameter search
    for h in tqdm(np.logspace(-1, 1, 20)):
        train_split,valid_split = torch.utils.data.random_split(train_dataset, [train_size, test_size])

       
        X_t,y_t,s_t = train_split[:]
        X_v,y_v,s_v = valid_split[:] 
        
        pw = ParzenWindow(h=h)
        pw.fit(X_t,y_t)
        
        y_pred = pw.predict(X_v)
        metrics_dict = compute_metrics(y_pred,y_v)
        metrics[h] = metrics_dict
 
    torch.save(metrics,"pzn_metrics.pt")

    for k in tqdm(np.arange(1,100,5)):
        train_split,valid_split = torch.utils.data.random_split(train_dataset, [train_size, test_size])

        X_t,y_t,s_t = train_split[:]
        X_v,y_v,s_v = valid_split[:] 

        knn = KNN(k=k,whitten=False)
        knn.fit(X_t,y_t)
    
        y_pred = knn.predict(X_v)
        metrics_dict = compute_metrics(y_pred,y_v)
        metrics[k] = metrics_dict

    torch.save(metrics,"knn_metrics_whitten.pt")
    print(metrics)
    

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_dataset)):

        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)

        train_loader = DataLoader(train_subsampler, batch_size=99, sampler=train_subsampler,num_workers=4,pin_memory=True,persistent_workers=True)
        valid_loader = DataLoader(train_subsampler, batch_size=99, sampler=valid_subsampler,num_workers=4,pin_memory=True,persistent_workers=True)
        
        knn = KNN(k=10)
        print(train_loader,valid_loader)

        #knn.fit(X_t,y_t)
        #test_dataloader = DataLoader(test_dataset,shuffle=False,batch_size=len(test_dataset),num_workers=4,pin_memory=True,persistent_workers=True)

    
    X_t,y_t,s_t = train_dataset[:]
    X_v,y_v,s_v = test_dataset[:] 

    pw = ParzenWindow(h=3.75)
    pw.fit(X_t,y_t)
    
    metrics = []
    for x,y in ((X_t,y_t),(X_v,y_v)):
        y_pred = pw.predict(x)
        metrics_dict = compute_metrics(y_pred,y)
        metrics.append(metrics_dict)

    torch.save(metrics,"pzn_metrics.pt")
    """
    k = 6

    X_t,y_t,s_t = train_dataset[:]
    knn = KNN(k=k,whitten=False)
    knn.fit(X_t,y_t)


    test_size = 0.8*len(test_dataset)
    metrics =[]
    for i in tqdm(range(10)):
        test_set_size = int(len(test_dataset) * 0.8)
        leftout_size = len(test_dataset) - test_set_size
        train_set, valid_set = torch.utils.data.random_split(test_dataset, [test_set_size, leftout_size])
        X_v,y_v,s_v = train_set[:]
        y_pred = knn.predict(X_v)
        metrics_dict = compute_metrics(y_pred,y_v)
        metrics.append(metrics_dict)

    torch.save(metrics,"knn_final_metrics.pt")
    print(metrics)