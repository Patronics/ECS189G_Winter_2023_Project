'''
Simple Dataset loader for CNN
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

# -- compatibility layer --
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.insert(0, ROOT_DIR)
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# -----------------------

from code.base_class.dataset import dataset
import numpy as np
import pickle

class Image_Dataset(dataset):
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading ' + self.dataset_name + '...')
        
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        self.data = pickle.load(f)
        f.close()
        
        
        
        
        
        
        
        
        
        
        
        
    