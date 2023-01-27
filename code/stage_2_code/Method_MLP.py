'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

# -- compatibility layer --
import sys
import os
# print(sys.path)
# from the file, add the root directory to python path.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# print(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
# # Then, we make sure the OS's cwd is at the local level to make it work
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# print(sys.path,os.getcwd())

useGPU = True


# -----------------------

from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
try:
    from tqdm import trange
except (ImportError, ModuleNotFoundError) as e:
    trange = range

class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 150 #500
    # it defines the learning rate for gradient descent based optimizer for model learning

    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, mDevice):
        method.__init__(self, mName, mDescription)
        self.deviceType = mDevice
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 500).to(self.deviceType)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU().to(self.deviceType)
        self.fc_layer_2 = nn.Linear(500, 10).to(self.deviceType)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_2 = nn.Softmax(dim=1).to(self.deviceType)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x)).to(self.deviceType)
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_2(self.fc_layer_2(h)).to(self.deviceType)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss().to(self.deviceType)

         #valid loss functions found so far: nn.CrossEntropyLoss, nn.MultiMarginLoss, nn.NLLLoss
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in trange(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)).to(self.deviceType))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y)).to(self.deviceType)
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)#.to(self.deviceType)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()
            if epoch%50 == 0:   # %100 by default
                with torch.no_grad():
                    y_true_numpy = y_true.cpu().detach().numpy()
                    y_pred_numpy = y_pred.cpu().detach().numpy().argmax(1)
                    accuracy_evaluator.data = {'true_y': y_true_numpy, 'pred_y': y_pred_numpy}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
    
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)).to(self.deviceType))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            
