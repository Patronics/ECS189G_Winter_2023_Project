# -- compatibility layer --
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.insert(0, ROOT_DIR)
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# -----------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from codes.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from sklearn.metrics import accuracy_score,classification_report
DATASET_NAME = 'cora'
dataset_loader = Dataset_Loader(DATASET_NAME, '')
dataset_loader.dataset_name = DATASET_NAME
dataset_loader.dataset_source_file_name = DATASET_NAME #not used
dataset_loader.dataset_source_folder_path = f'../../data/stage_5_data/{DATASET_NAME}'
dataset = dataset_loader.load()

print()

print('The datasets X contains this many features',dataset['graph']['X'].shape[1])
print(dataset['graph']['y'].numpy().max())
print(dataset['graph']['y'].numpy().min())
print('The datasets y contains this amount of categories:',dataset['graph']['y'].numpy().max()+1)

class testNet(nn.Module):
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
        super(testNet,self).__init__()
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

    def test_model(self,dataset):
        test_IDX = dataset['train_test_val']['idx_test']
        x = dataset['graph']['X']
        y = dataset['graph']['y']
        adj = dataset['graph']['utility']['A']
        outputs = self(x,adj)
        _,outputLabels = torch.max(outputs.data,1)
        print(classification_report(y[test_IDX].cpu().detach().numpy(), outputLabels[test_IDX].cpu().detach().numpy()))
        return
print()

#---------------------------------------------------

model = testNet(torch.device('cuda')).to('cuda')

model.train_model(dataset)

model.test_model(dataset)

