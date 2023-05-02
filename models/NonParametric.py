import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from Dataloader import WholeDataset
from torch.utils.data import  DataLoader
from sklearn.neighbors import KNeighborsClassifier
from Dataloader import BaseModel

class KNN:
    def __init__(self,k=10,center=False):
        self.base_model = BaseModel()
        self.K = k
        self.center = center
        self.data = self.base_model.load_weigts(loc)
        self.knn = KNeighborsClassifier(n_neighbors=k)
    def fit(self,X,y):
        self.knn.fit(X,y)
    def predict(self,X):
        y_test_pred = self.knn.predict(X) 
        return torch.tensor(y_test_pred)


class ParzenWindow:
    def __init__(self,h=10):
        self.base_model = BaseModel()
        self.H = h
        self.data = self.base_model.load_weigts(loc)
    def fit(self,X,y):
        pass
    def predict(self,X,y):
        pass



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda:5"
    else:
        device = "cpu"

    args_train = Args("train")
    train_dataset = WholeDataset(args_train)
    args_test  = Args("test")
    test_dataset = WholeDataset(args_test)

    loc = "ResnetFeatures.pt"

    """
    
    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=TRAIN_BATCH_SIZE,num_workers=4,pin_memory=True,persistent_workers=True)
    test_dataloader = DataLoader(test_dataset,shuffle=False,batch_size=TEST_BATCH_SIZE,num_workers=4,pin_memory=True,persistent_workers=True)


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
    
    """



