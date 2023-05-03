import wandb,os

#change the environ here

DATA_DIR = "colored_mnist"
BATCH_SIZE = 1000
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
        "permutation":False,
        "DATA_DIR":DATA_DIR
    })


class Args(object):
    def __init__(self,data_split):
        self.data_dir = DATA_DIR
        self.data_split = data_split
        self.color_var = 0.040
        self.BATCH_SIZE = BATCH_SIZE
        self.LOG = LOG

config = wandb.config