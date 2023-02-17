'''
Base evaluate class for all evaluation metrics and methods
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

from code.base_class.result import result
import pickle

class Results_CNN(result):

    def __init__(self, rName=None, rType=None):
        self.result_name = rName
        self.result_description = rType

    def save(self):
        print('saving results...')
        f = open(self.result_destination_folder_path + self.result_destination_file_name, 'wb')
        pickle.dump(self.data, f)
        f.close()
 
    def load(self):
        print('loading results...')
        f = open(self.result_destination_folder_path + self.result_destination_file_name, 'rb')
        self.data = pickle.load(f)
        f.close()
