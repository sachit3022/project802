import wget,os,random
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy as scp


import sys
sys.path.append('..')

import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn

import torchmetrics

from Dataloader import WholeDataset,ClassConditionalPermutation
from copy import deepcopy

from PIL import Image

#change the environ here

DATA_DIR = "colored_mnist"
TRAIN_BATCH_SIZE = 30000
TEST_BATCH_SIZE = 1000
LOG  = "disabled" #dryrun #online

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


"""
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
"""


class PCA:

    def __init__(self,num_dim=20):
        self.num_dim=num_dim
        self.projection_space = None
        self.reconstruction_space = None

    def fit(self,X):

        U,S,V = torch.pca_lowrank(X, q=self.num_dim, center=True)
        self.projection_space   = V
        self.reconstruction_space = V.T
        return  X @ self.projection_space  
    
    def project(self,X):
        Z = X @ self.projection_space
        return  Z
    
    def compress(self,X):
        R = X @ self.projection_space @ self.reconstruction_space
        return R

class MDA:
    def __init__(self,num_dim=20):
        self.num_dim=num_dim
        self.projection_space = None
        self.reconstruction_space = None

    def fit(self,X,y,s=None):
        
        d =  X.shape[-1] #number of diamensions
        X = X - X.mean(dim=(-2,), keepdim=True)

        intra_class_variance = torch.zeros(d,d)
        inter_class_variance = torch.zeros(d,d)

        total_mean = X.mean(axis=0,keepdims=True)
        for c in np.unique(y):
            U =  ( X[y==c].mean(axis=0,keepdims=True) - total_mean )
            inter_class_variance += len(X[y==c])*U.T @ U

        
        for c in np.unique(y):
            U = X[y==c] - X[y==c].mean(axis=0)
            intra_class_variance +=  U.T @ U

        bias_class_variance = torch.zeros(d,d)

        if s is not None:
            #here we consider s as a 3 channel one     
            for s_c in range(s.shape[-1]):
                for u_b in np.unique(s[:,s_c]):
                    U = X[s[:,s_c]==u_b] - X[s[:,s_c]==u_b].mean(axis=0,keepdims=True)
                    bias_class_variance +=  U.T @ U

        print((bias_class_variance==0).all())
        print((intra_class_variance==0).all())
        print((inter_class_variance==0).all())

        intra_class_variance +=0.0001*torch.eye(d) #+  0.01*bias_class_variance # improve the condition number of a matrix/
        if s is not None:
            bias_class_variance +=0.0001*torch.eye(d)
        else:
            bias_class_variance = torch.eye(d).float()

        U,S,V = torch.pca_lowrank(torch.linalg.inv(intra_class_variance @ bias_class_variance)@inter_class_variance, q=self.num_dim, center=True)
        self.projection_space   = V
        self.reconstruction_space = V.T
       
        """
        eigenValues,eigVectors = torch.lobpcg(inter_class_variance,self.num_dim,intra_class_variance @ bias_class_variance)
        idx = eigenValues.numpy().argsort()[::-1]
        eigenValues = eigenValues.numpy()[idx]
        self.projection_space = eigVectors.numpy()[:,idx][:,:self.num_dim]
        self.reconstruction_space = self.projection_space.T
        """




        return  X @ self.projection_space  
    """
    def reconstruct(self,Z,X):

        self.reconstruct = self.projection_space.T


        
        W = torch.rand((Z.shape[-1],X.shape[-1]))
        W.requires_grad = True
        
        optimizer = optim.Adam([W], lr=1)
        loss= nn.MSELoss()

        for e in range(1000):
            optimizer.zero_grad()
            l = loss(Z  @ W, X)
            l.backward()
            optimizer.step()

        self.reconstruction_space = W.detach()
        return self.compress(X)
    """
    def project(self,X):
        Z = X @ self.projection_space
        return  Z
    def compress(self,X):
        R = X @ self.projection_space @ self.reconstruction_space
        return R

def plotTopKK(X,k=5,file="placehoder"):
    fig,ax = plt.subplots(k,k)
    for i in range(5):
        for j in range(5):
            ax[i][j].imshow(np.transpose(X[i*5+j],(1,2,0)))
    fig.savefig(f'../results/{file}')



"""
K = {}
for c in np.unique(y_t):
    S= s_t[y_t==c].T
    covar_inv = deepcopy(torch.linalg.inv(torch.cov(S)).numpy())
    mean = deepcopy(S.float().mean(dim=1).numpy())
    K[c]  = lambda x : (x-mean).T @ covar_inv @ (x - mean)





S= s_t[y_t==c].T
covar_inv = torch.linalg.inv(torch.cov(S)).numpy()
mean = S.float().mean(dim=1).numpy()
maha  = lambda x : (x-mean).T @ covar_inv @ (x - mean)
M = np.apply_along_axis(maha,1,s_t)



def class_maha_dist(x):
    return K[x[1].item()](x[0])
M = np.apply_along_axis(class_maha_dist,1,np.array(list(zip(s_t,y_t))))






fig,ax = plt.subplots(5,5)
datapoints = R[M>5].reshape(-1,3,28,28)
for i in range(5):
    for j in range(5):
        ax[i][j].imshow(np.transpose(datapoints[i*5+j],(1,2,0)))
fig.show()
fig,ax = plt.subplots(5,5)
datapoints = X_t[M>5]
for i in range(5):
    for j in range(5):
        ax[i][j].imshow(np.transpose(datapoints[i*5+j],(1,2,0)))
"""

def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    m = X.shape[0]
    X = X - X.mean(axis=0) # cener the data
    C = X.T @ X / m # 

    eig_vals, eig_vecs = np.linalg.eig(C)
    D = np.diag(eig_vals) # eig_vals is a vector, but we want a matrix
    P = eig_vecs

    D_m12 = np.diag(np.diag(D)**(-0.5))
    W_ZCA = P @ D_m12 @ P.T 
    return torch.tensor(W_ZCA)




if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda:5"
    else:
        device = "cpu"

    args_train = Args("train")
    train_dataset = WholeDataset(args_train)
    args_test  = Args("test")
    test_dataset = WholeDataset(args_test)
    
    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=TRAIN_BATCH_SIZE,num_workers=4,pin_memory=True,persistent_workers=True)
    test_dataloader = DataLoader(test_dataset,shuffle=False,batch_size=TEST_BATCH_SIZE,num_workers=4,pin_memory=True,persistent_workers=True)
    
    mda = MDA()
    pca = PCA()

    X_t,l_t,y_t,s_t  = next(iter(train_dataloader))
    X_v,l_v,y_v,s_v  = next(iter(test_dataloader))
    og_dim = X_t.shape
    ov_dim = X_v.shape

    
    plotTopKK(X_t,file="source.png")

    X_t = X_t.flatten(start_dim=1)
    X_v = X_v.flatten(start_dim=1)
    

    Z = mda.fit(X_t,y_t,s_t)     
    X_hat = X_t @ mda.projection_space @ mda.reconstruction_space
    X_hat = X_hat.reshape(og_dim)


    fig,ax = plt.subplots(5,5)
    for i in range(5):
        for j in range(5):
            ax[i][j].imshow(np.transpose(X_hat[i*5+j],(1,2,0)))
    
    plotTopKK(X_hat,file="MDA_target_biased.png")

    
    Z = mda.fit(X_t,y_t)     
    X_hat = X_t @ mda.projection_space @ mda.reconstruction_space
    X_hat = X_hat.reshape(og_dim)

    fig,ax = plt.subplots(5,5)
    for i in range(5):
        for j in range(5):
            ax[i][j].imshow(np.transpose(X_hat[i*5+j],(1,2,0)))
    
    plotTopKK(X_hat,file="MDA_target_biased.png")


    Z = pca.fit(X_t)
    X_hat = X_t @ mda.projection_space @ mda.reconstruction_space
    X_hat = X_hat.reshape(og_dim)
    plotTopKK(X_hat,file="MDA_target_biased.png")
    





