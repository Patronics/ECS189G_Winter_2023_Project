'''
Base MethodModule class for all models and frameworks
'''

# Copyright (c) 2023 Patrick leiser
# License: TBD

# -- compatibility layer --
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.insert(0, ROOT_DIR)
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# -----------------------
from codes.stage_5_code.evaluate_RNN import Evaluate_RNN
from codes.base_class.method import method
import torch.nn.functional as F
#from torchtext.data import BucketIterator
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.utils.data as Tdata
from torch import nn
import torch.optim as optim
import torch
import numpy as np

try:
    from tqdm import trange
except (ImportError, ModuleNotFoundError) as e:
    print("tqdm module not detected, for improved progress updates, install tqdm")
    trange = range

#-----------------------------------------------------

class Method_GCN_Class(nn.Module):
    @staticmethod
    def init_params(in_feat, out_feat, bias=True):
        weight = nn.Parameter(torch.FloatTensor(in_feat, out_feat))
        nn.init.uniform_(weight)
        if bias:
            bias = nn.Parameter(torch.FloatTensor(out_feat))
            nn.init.zeros_(bias)
        else:
            bias=None
        return weight, bias
    def graph_conv(self, input, adj, weight, bias):
        adj = adj.to(self.deviceType)
        support=torch.mm(input, weight)
        output=torch.sparse.mm(adj, support)
        if bias is not None:
            output=output+bias
        return output
    def __init__(self,deviceType):
        super(Method_GCN_Class,self).__init__()
        self.deviceType = deviceType
        in_features = 1433
        hidden_dim = 16
        out_features = 7
        self.dropout = 0.1
        
        self.gc1_weight, self.gc1_bias = self.init_params(in_features, hidden_dim)
        self.gc2_weight, self.gc2_bias = self.init_params(hidden_dim, out_features)
        # self.fc1 = nn.Linear(1,1)
    def forward(self,data,adj):
        data = data.to(self.deviceType)
        data = F.relu(self.graph_conv(data, adj, self.gc1_weight, self.gc1_bias))
        data = F.dropout(data, self.dropout, training=self.training)
        data = self.graph_conv(data, adj, self.gc2_weight, self.gc2_bias)
        return F.log_softmax(data, dim=1).cpu() # use softmax perhaps
    
    def train_model(self,dataset):
        epochs = 200
        lr = 0.01
        optimizer = torch.optim.Adam(self.parameters(),lr=lr,weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        self.train()

        train_IDX = dataset['train_test_val']['idx_train']
        x = dataset['graph']['X']
        y = dataset['graph']['y']
        adj = dataset['graph']['utility']['A']
        progress = trange(epochs)
        for epoch in progress:
            outputs = self(x,adj)
            loss = criterion(outputs[train_IDX],y[train_IDX])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_postfix_str(f'Loss: {float(loss.cpu().detach().numpy()):7.6f}', refresh=True)