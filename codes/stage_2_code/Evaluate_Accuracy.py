'''
Concrete Evaluate class for a specific evaluation metrics
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

from codes.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
import torch
import numpy as np


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.cpu().numpy()
        print('evaluating performance...')
        dataTrue = np.asarray(self.data['true_y'])
        dataPred = self.data['pred_y'].cpu().detach().numpy()
        print()
        return accuracy_score(dataTrue, dataPred)
        