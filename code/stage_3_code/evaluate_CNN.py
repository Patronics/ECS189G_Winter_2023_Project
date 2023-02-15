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

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score,classification_report


class Evaluate_CNN(evaluate):
    
    def __init__(self, eName=None, eDescription=None):
        self.evaluate_name = eName
        self.evaluate_description = eDescription


    def evaluate(self):
        print('evaluating performance...')
        dataTrue = self.data['true_y']
        dataPred = self.data['pred_y']
        print()
        return accuracy_score(dataTrue, dataPred)

    def classificationReport(self):
        dataTrue = self.data['true_y']
        dataPred = self.data['pred_y']
        print(classification_report(dataTrue, dataPred))