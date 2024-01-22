from datasets import load_dataset
# import seaborn as sns
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from umap import UMAP
# import pandas as pd




import os
import time
from datetime import datetime
import wandb
import torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score



import os
import time
from datetime import datetime
import wandb
import torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score


# Define GSEA Net
class geneprog_encoder_linear(torch.nn.Module):
    # x is cell by genes, so parameters should thus be genes by gene_programs
    def __init__(self,in_dim,out_dim):
        super(geneprog_encoder_linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lin1 = torch.nn.Linear(in_dim, out_dim, bias = True)

    def forward(self, x):
        # print(x.dtype)
        result = self.lin1(x)
        return result






class geneprog_encoder_MLP(torch.nn.Module):
    # x is cell by genes, so parameters should thus be genes by gene_programs
    def __init__(self,in_dim,out_dim):
        super(geneprog_encoder_MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lin1 = torch.nn.Linear(in_dim, out_dim, bias = True)
        hid1 = hid2 = in_dim
                # First fully connected layer
        self.fc1 = torch.nn.Linear(in_dim, hid1)
        # Second fully connected layer
        self.fc2 = torch.nn.Linear(hid1, hid2)
        # Third fully connected layer that outputs our result
        self.fc3 = torch.nn.Linear(hid2, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Pass through the second layer, then apply ReLU
        x = F.relu(self.fc2(x))
        # Pass through the third layer (no ReLU here)
        x = self.fc3(x)
        return x