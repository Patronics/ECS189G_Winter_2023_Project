'''
Concrete SettingModule class for a specific experimental SettingModule
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

# -----------------------

from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np
import torch

class Setting_Train_Test_Split(setting):
    fold = 3
    
    def load_run_save_evaluate(self, deviceType):
        
        # load dataset
        loaded_data = self.dataset.load()

        X_train, X_test, y_train, y_test = loaded_data['X'], loaded_data['XTest'], loaded_data['y'],loaded_data['yTest'] 

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        #print(self.result.data)
        self.result.data['pred_y'] = torch.asarray(self.result.data['pred_y']).cpu().detach().numpy()
        self.result.save()
        self.evaluate.data = learned_result
        #self.evaluate.data['true_y'] = torch.asarray(self.evaluate.data['true_y']).cpu().detach().numpy()
        self.evaluate.data['pred_y'] = torch.asarray(self.evaluate.data['pred_y']).cpu().detach().numpy()
        return self.evaluate.evaluate(), None

        
