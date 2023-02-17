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
from code.stage_3_code.evaluate_CNN import Evaluate_CNN
from code.base_class.method import method
import torch.nn.functional as F
import torch.utils.data as Tdata
from torch import nn
import torch

try:
    from tqdm import trange
except (ImportError, ModuleNotFoundError) as e:
    print("tqdm module not detected, for improved progress updates, install tqdm")
    trange = range

#-----------------------------------------------------


class Method_CNN(method, nn.Module):

    max_epoch = 50
    learning_rate = 1e-3
    batch_size = 120
    
    deviceType = None
    
    def __init__(self, mName=None, mDescription=None, sInput=1 ,mDevice=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.deviceType = mDevice
        
        #-- Layer Definition --
        self.conv1 = nn.Conv2d(in_channels=sInput, out_channels=16, kernel_size=3, stride=1).to(self.deviceType)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1).to(self.deviceType)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1).to(self.deviceType)
        
        self.fc1 = nn.Linear(6912, 96).to(self.deviceType)
        self.fc2 = nn.Linear(96, 10).to(self.deviceType)

    def forward(self, x, train=True):
        #-- Conv Section --
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=0.25, training=train)
        x = F.max_pool2d(x, 2)
        
        #-- FC Section --
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=0.25, training=train)
        x = self.fc2(x)
        
        return F.relu(x)
    
    def train(self, X, y):
        #-- Tool Init --
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss().to(self.deviceType)
        evaluator = Evaluate_CNN('training evaluator', '')
        
        #Pointer to a batch
        data_iter = Tdata.DataLoader(Tdata.TensorDataset(X, y), batch_size=self.batch_size)
        
        #-- Mini-Batch GD loop --
        for epoch in trange(self.max_epoch + 1):
            for batch_i, (images, labels) in enumerate(data_iter):
                print(images.shape)
                y_pred = self.forward(images.to(self.deviceType))
                y_true = torch.LongTensor(labels).to(self.deviceType)
                
                optimizer.zero_grad()
                loss = loss_function(y_pred, y_true)
                loss.backward()
                optimizer.step()
            
    def test(self, X):
        y_pred = self.forward(X.to(self.deviceType), train=False)
        return y_pred.max(1)[1]
            
            

    def run(self, trainData, trainLabel, testData):
        print('method running...')
        print('--start training...')
        
        #print(torch.stack(trainData).unsqueeze(1))
        self.train(torch.stack(trainData).unsqueeze(1), torch.stack(trainLabel))
        print('--start testing...')
        return self.test(testData)