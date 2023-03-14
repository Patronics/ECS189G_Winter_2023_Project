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

class Method_GCN_Class(method, nn.Module):
    #-- Hyper Variables --
    max_epoch = 100
    learning_rate = 1e-3
    batch_size = 128
    
    hidden_dim = 16
    dropout = 0.2
    
    def __init__(self, mName=None, mDescription=None, in_features=0,out_features=0, gcn_model=None, mDevice=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.deviceType = mDevice
        self.gc1_weight, self.gc1_bias = self.init_params(in_features, hidden_dim)
        self.gc2_weight, self.gc2_bias = self.init_params(hidden_dim, out_features)
        
        #-- Embedding Layer Definition --
        #self.embedding = nn.Embedding(vocab_size, embedding_dim=self.embedding_dim)
        
        #-- RNN Architecture Definition --
        # if rnn_model == "GCN":
        #     self.rnn = nn.LSTM(self.embedding_dim, 
        #                        hidden_size=self.max_words, 
        #                        num_layers=self.hidden_layers, 
        #                        bidirectional=self.bidirectional, 
        #                        dropout=self.dropout,
        #                        batch_first=True).to(self.deviceType)
        # elif rnn_model == "GRU":
        #     self.rnn = nn.GRU(self.embedding_dim, 
        #                       hidden_size=self.max_words, 
        #                       num_layers=self.hidden_layers, 
        #                       bidirectional=self.bidirectional, 
        #                       dropout=self.dropout,
        #                       batch_first=True).to(self.deviceType)
        # else:
        #     self.rnn = nn.RNN(self.embedding_dim, 
        #                       hidden_size=self.max_words, 
        #                       num_layers=self.hidden_layers, 
        #                       bidirectional=self.bidirectional, 
        #                       dropout=self.dropout,
        #                       batch_first=True).to(self.deviceType)
        # 
        #-- FC Output Layer Definition --
        #self.fc = nn.Linear(self.max_words * (2 if self.bidirectional else 1), 2).to(self.deviceType)
      
    def init_params(in_feat, out_feat, bias=True):
        weight = nn.Parameter(torch.FloatTensor(in_feat, out_feat))
        nn.init.uniform_(weight)
        if bias:
            bias = nn.Parameter(torch.FloatTensor(out_features))
            nn.init.zeros_(bias)
        else:
            bias=None
        return weight, bias
    
    def graph_conv(self, input. adj, weight, bias):
        support=torch.mm(input, weight)
        output=torch.sparse.mm(adj, support)
        if bias is not None:
            output=output+bias
        return output
    
    def forward(self, data, adj):
        # converts indexes into unique embedding vectors
        # data = self.embedding(data)
        
        # packing/padding to more efficeiently ignore empty inputs
        data = data.to(self.deviceType)
        data = F.relu(self.graph_conv(data, adj, self.gc1_weight, self.gc1_bias))
        data = F.dropout(data, self.dropout, training=self.training)
        data = self.graph_conv(data, adj, self.gc2_weight, self.gc2_bias)
        #data = pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
        #data, hn = self.rnn(data)
        #data, _ = pad_packed_sequence(data, batch_first=True)
        
        # connect the last hidden layer in both directions to the fc layer 
        #if self.bidirectional:
        #    data = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        
        #data = torch.sum(data,1)
    
        #data = self.fc(data)
        return F.log_softmax(data, dim=1).cpu()#torch.sigmoid(data).cpu()
    
    def train(self, dataLoader):
        #-- Tool Init --
        
        #model = Method_GCN_Class(self.dataset_name, '')
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)
        loss_function = nn.crossEntropyLoss()#nn.BCELoss()
        
        #-- Mini-Batch GD loop --
        super(Method_GCN_Class, self).train()
        progress = trange(self.max_epoch)
        for epoch in progress:
            for data, lengths, labels in dataLoader(self.batch_size):
                pred = self.forward(data, lengths).squeeze()

                optimizer.zero_grad()
                loss = loss_function(pred, labels)
                
                # Early Stop
                if loss <= 0.001:
                    return
                
                loss.backward()
                optimizer.step()

                progress.set_postfix_str(f'Loss: {float(loss.cpu().detach().numpy()):7.6f}, lrate: {scheduler.get_last_lr()[0]:2.6f}', refresh=True)
                #print(f'{float(loss.cpu().detach().numpy()):7.6f}')
            
            scheduler.step()
            
    def test(self, X):
        prompts,length = X.getData()
        y_pred = self.forward(prompts,length)
        return y_pred.max(1)[1]
            
    def run(self, trainData, testData):
        print('method running...')
        print('--start training...')
        
        self.train(trainData)
        
        print('--start testing...')
        return (self.test(testData),testData.yLabels)