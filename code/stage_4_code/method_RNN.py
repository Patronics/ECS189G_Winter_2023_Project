'''
Base MethodModule class for all models and frameworks
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

# -- compatibility layer --
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.insert(0, ROOT_DIR)
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# -----------------------
from codes.stage_4_code.evaluate_RNN import Evaluate_RNN
from codes.base_class.method import method
import torch.nn.functional as F
#from torchtext.data import BucketIterator
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.utils.data as Tdata
from torch import nn
import torch.optim as optim
import torch

try:
    from tqdm import trange
except (ImportError, ModuleNotFoundError) as e:
    print("tqdm module not detected, for improved progress updates, install tqdm")
    trange = range

#-----------------------------------------------------

class Method_RNN_Class(method, nn.Module):
    #-- Hyper Variables --
    max_epoch = 100
    learning_rate = 1e-3
    batch_size = 128
    
    embedding_dim = 50
    max_words = 32
    hidden_layers = 2
    bidirectional = False
    dropout = 0.2
    
    def __init__(self, mName=None, mDescription=None, vocab_size=0, rnn_model=None, mDevice=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.deviceType = mDevice
        
        #-- Embedding Layer Definition --
        #self.embedding = nn.Embedding(vocab_size, embedding_dim=self.embedding_dim)
        
        #-- RNN Architecture Definition --
        if rnn_model == "lstm":
            self.rnn = nn.LSTM(self.embedding_dim, 
                               hidden_size=self.max_words, 
                               num_layers=self.hidden_layers, 
                               bidirectional=self.bidirectional, 
                               dropout=self.dropout,
                               batch_first=True).to(self.deviceType)
        elif rnn_model == "GRU":
            self.rnn = nn.GRU(self.embedding_dim, 
                              hidden_size=self.max_words, 
                              num_layers=self.hidden_layers, 
                              bidirectional=self.bidirectional, 
                              dropout=self.dropout,
                              batch_first=True).to(self.deviceType)
        else:
            self.rnn = nn.RNN(self.embedding_dim, 
                              hidden_size=self.max_words, 
                              num_layers=self.hidden_layers, 
                              bidirectional=self.bidirectional, 
                              dropout=self.dropout,
                              batch_first=True).to(self.deviceType)
        
        #-- FC Output Layer Definition --
        self.fc = nn.Linear(self.max_words * (2 if self.bidirectional else 1), 2).to(self.deviceType)
        

    def forward(self, data, lengths):
        # converts indexes into unique embedding vectors
        # data = self.embedding(data)
        
        # packing/padding to more efficeiently ignore empty inputs
        data = data.to(self.deviceType)
        data = pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
        data, hn = self.rnn(data)
        data, _ = pad_packed_sequence(data, batch_first=True)
        
        # connect the last hidden layer in both directions to the fc layer 
        if self.bidirectional:
            data = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        
        data = torch.sum(data,1)
    
        data = self.fc(data)
        return F.sigmoid(data).cpu()
    
    def train(self, dataLoader):
        #-- Tool Init --
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)
        loss_function = nn.BCELoss()

        #-- Mini-Batch GD loop --
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
    
class Method_RNN_Gen(method, nn.Module):
     #-- Hyper Variables --
    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 1
    
    # embedding_dim = 300
    max_words = 3
    hidden_layers = 2
    bidirectional = False
    dropout = 0.2
    
    def __init__(self, mName=None, mDescription=None, rnn_model=None, mDevice=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.deviceType = mDevice
        
        #-- Embedding Layer Definition --
        # self.embedding = nn.Embedding(vocab_size, embedding_dim=self.embedding_dim)
        
        #-- RNN Architecture Definition --
        if rnn_model == "lstm":
            self.rnn = nn.LSTM(3, 
                               hidden_size=self.max_words, 
                               num_layers=self.hidden_layers, 
                               bidirectional=self.bidirectional, 
                               dropout=self.dropout,
                               batch_first=True)
        elif rnn_model == "GRU":
            self.rnn = nn.GRU(3, 
                              hidden_size=self.max_words, 
                              num_layers=self.hidden_layers, 
                              bidirectional=self.bidirectional, 
                              dropout=self.dropout,
                              batch_first=True)
        else:
            self.rnn = nn.RNN(3, 
                              hidden_size=self.max_words, 
                              num_layers=self.hidden_layers, 
                              bidirectional=self.bidirectional, 
                              dropout=self.dropout,
                              batch_first=True)
        
        #-- FC Output Layer Definition --
        self.fc = nn.Linear(self.max_words * (2 if self.bidirectional else 1), 2)
        

    def forward(self, data, hn):

        data, hn = self.rnn(data)

        # connect the last hidden layer in both directions to the fc layer 
        if self.bidirectional:
            data = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        
        # data = torch.sum(data,1)
    
        data = self.fc(data)
        return F.relu(data), hn
    
    def train(self, dataLoader):
        #-- Tool Init --
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)
        loss_function = nn.CrossEntropyLoss()

        #-- Mini-Batch GD loop --
        progress = trange(self.max_epoch)
        for epoch in progress:
            loss = 0
            for data, lengths in dataLoader(self.batch_size):
                for i in range(0, data.shape[0]-3):
                    pred = self.forward(data[i:i+2])
                    loss += loss_function(pred, data[i:i+2])
                    
                optimizer.zero_grad()
                
                # Early Stop
                if loss <= 0.001:
                    return
                
                loss.backward()
                optimizer.step()

                progress.set_postfix_str(f'Loss: {float(loss.cpu().detach().numpy()):7.6f}, lrate: {scheduler.get_last_lr()[0]:2.6f}', refresh=True)
                #print(f'{float(loss.cpu().detach().numpy()):7.6f}')
            
            scheduler.step()
            
    def test(self, data):
        prompts,length = data.getData()
        y_pred = self.forward(prompts,length)
        return y_pred.max(1)[1]
            
    def run(self, trainData, testData):
        print('method running...')
        print('--start training...')
        
        self.train(trainData)
        
        print('--start testing...')
        return (self.test(testData),testData.yLabels)
        