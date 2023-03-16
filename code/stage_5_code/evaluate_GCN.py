'''
Base evaluate class for all evaluation metrics and methods
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

#-- compatibility layer --
import sys
import os

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


    def evaluate(self):
        print('evaluating performance...\n')
        dataPred = self.data[0]
        dataTrue = self.data[1]
        return accuracy_score(dataTrue.cpu(), dataPred.cpu())

    def classificationReport(self):
        dataPred = self.data[0]
        dataTrue = self.data[1]
        print(classification_report(dataTrue.cpu().numpy(), dataPred.cpu().numpy()))