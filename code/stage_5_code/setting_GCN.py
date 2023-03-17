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

from codes.base_class.setting import setting

#-----------------------------------------------------
class GCN_Trainer(setting):
    def load_run_save_evaluate(self,datasetName):
        self.dataset.load()
        self.method.data = self.dataset.data
        result = self.method.run(self.dataset.data)
        print(result[0])
        self.result.data = result[0]
        self.result.save()
        self.evaluate.data = result[0]
        self.evaluate.classificationReport(self.method.data,result)
        return self.evaluate.evaluate(result)