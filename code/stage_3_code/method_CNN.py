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
import torch.optim as optim
import torch

try:
    from tqdm import trange
except (ImportError, ModuleNotFoundError) as e:
    print("tqdm module not detected, for improved progress updates, install tqdm")
    trange = range

#-----------------------------------------------------


class Method_CNN(method, nn.Module):

    max_epoch = 200
    learning_rate = 1e-3
    batch_size = 220
    
    deviceType = None
    
    PRINT_LOSS_PER_EPOCH = True
    
    def __init__(self, mName=None, mDescription=None, sInput=0, fc_input=0, fc_output=0 ,mDevice=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.deviceType = mDevice
        
        #-- Layer Definition --
        self.conv1 = nn.Conv2d(in_channels=sInput, out_channels=16, kernel_size=3, stride=1, padding=1).to(self.deviceType)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=48, kernel_size=3, stride=1, padding=1).to(self.deviceType)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1).to(self.deviceType)
        
        self.fc1 = nn.Linear(fc_input, 128).to(self.deviceType)
        self.fc2 = nn.Linear(128, fc_output).to(self.deviceType)

    def forward(self, x, train=True):
        #-- Conv Section --
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=train)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=train)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=train)
        x = F.max_pool2d(x, 2)
        
        #-- FC Section --
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=train)
        x = self.fc2(x)
        
        return F.relu(x)
    
    def train(self, X, y):
        #-- Tool Init --
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)
        loss_function = nn.CrossEntropyLoss()
        
        #Pointer to a batch
        data_iter = Tdata.DataLoader(Tdata.TensorDataset(X, y), batch_size=self.batch_size)
        
        #-- Mini-Batch GD loop --
        if self.PRINT_LOSS_PER_EPOCH:
            progress = range(self.max_epoch)
        else:
            progress = trange(self.max_epoch, desc='Epoch', leave=True)
        
        for epoch in progress:
            if self.PRINT_LOSS_PER_EPOCH:
                print(epoch+1, end='\t')

            for batch_i, (images, labels) in enumerate(data_iter):
                y_pred = self.forward(images.to(self.deviceType))
                y_true = torch.LongTensor(labels).to(self.deviceType)
                
                optimizer.zero_grad()
                loss = loss_function(y_pred, y_true)
                
                # Early Stop
                # if loss <= 0.001:
                #     return
                
                loss.backward()
                optimizer.step()

                if not self.PRINT_LOSS_PER_EPOCH:
                    progress.set_postfix_str(f'Loss: {float(loss.cpu().detach().numpy()):7.6f}, lrate: {scheduler.get_last_lr()[0]:2.6f}', refresh=True)
            
            if self.PRINT_LOSS_PER_EPOCH:
                print(f'{float(loss.cpu().detach().numpy()):7.6f}')
            
            scheduler.step()
            
                
            
    def test(self, X):
        y_pred = self.forward(X.to(self.deviceType), train=False)
        return y_pred.max(1)[1]
            
    def inputDataPrep(self, data):
        with torch.no_grad():
            if len(data.shape) == 4: # 3ch Color image
                dataSlices = []
                for i in range(data.shape[3]):
                    dataSlices.append(data[:,:,:,i])
                newData = torch.stack(dataSlices,dim=1)
                return newData
            else:                   # 1ch Grey image
                newData = data.unsqueeze(1)
                return newData

    def run(self, trainData, trainLabel, testData, testLabel):
        trainData = self.inputDataPrep(torch.stack(trainData))
        testData = self.inputDataPrep(torch.stack(testData))
        print('method running...')
        print('--start training...')
        
        #Offset for label classification
        if min(testLabel) or min(trainLabel):
            offset = 1
        else:
            offset = 0
        
        self.train(trainData, torch.stack(trainLabel)-offset)
        print('--start testing...')
        return (self.test(testData), torch.stack(testLabel)-offset)