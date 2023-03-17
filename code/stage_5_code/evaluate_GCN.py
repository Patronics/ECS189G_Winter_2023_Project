'''
Base evaluate class for all evaluation metrics and methods
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

#-- compatibility layer --
import sys
import os
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.insert(0, ROOT_DIR)
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# -----------------------

from codes.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score,classification_report


class Evaluate_GCN(evaluate):
    
    def __init__(self, eName=None, eDescription=None):
        self.evaluate_name = eName
        self.evaluate_description = eDescription


    def evaluate(self, dataset):
        print('evaluating performance...\n')
        dataPred = outputLabels[test_IDX].cpu().detach().numpy()
        dataTrue = y[test_IDX].cpu().detach().numpy()
        return accuracy_score(dataTrue.cpu(), dataPred.cpu())

    def classificationReport(self, dataset, outputs):
        test_IDX = dataset['train_test_val']['idx_test']
        x = dataset['graph']['X']
        y = dataset['graph']['y']
        adj = dataset['graph']['utility']['A']
        #outputs = self(x,adj)
        print(outputs)
        _,outputLabels = torch.max(outputs[0],1)
        dataPred = outputLabels[test_IDX].cpu().detach().numpy()
        dataTrue = y[test_IDX].cpu().detach().numpy()
        print(classification_report(dataTrue, dataPred))