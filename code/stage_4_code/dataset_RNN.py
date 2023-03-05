'''
Simple Dataset loader for RNN
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

class Text_Dataset(dataset):
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load_gen(self):
        print('loading ' + self.dataset_name + '...')

        #TODO: load the training and testing data into @self.data as one data structure
        """
        Convert all cleaned words in the test/training data to their corresponding indexes.
        
        start indexes from 1
        
        self.data = ["train":[...word_id],
                     "test": [...word_id]]
        """
        pass

        
    def load_class(self):
        print('loading ' + self.dataset_name + '...')
        
        #TODO: load the training and testing data into @self.data as one data structure
        """
        Convert all cleaned words in the test/training data to their corresponding indexes
        Label the neg data with 0 and pos data with 1 before passing in
        
        start indexes from 1
        
        self.data = ["train":[[...word_id], label],
                     "test": [[...word_id], label]
        """
        pass
        
        
        
        
        
        
        
        
        
        
        
    