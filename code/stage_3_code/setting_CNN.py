'''
Base SettingModule class for all experiment settings
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

from code.base_class.setting import setting
import torch

#-----------------------------------------------------
class CNN_Trainer(setting):
    
    def selectList_image(item1,item2):
        return torch.tensor(item2['image'], dtype=torch.float)
    
    def selectList_label(item1,item2):
        return torch.tensor(item2['label'], dtype=torch.long)

    def load_run_save_evaluate(self):
        self.dataset.load()
        self.method.data = self.dataset.data
        result = self.method.run(list(map(self.selectList_image,self.dataset.data['train'])), 
                                 list(map(self.selectList_label,self.dataset.data['train'])),
                                 list(map(self.selectList_image,self.dataset.data['test'])),
                                 list(map(self.selectList_label,self.dataset.data['test'])))
        
        self.result.data = result[0]
        self.result.save()
        
        self.evaluate.data = result
        self.evaluate.classificationReport()
        return self.evaluate.evaluate()