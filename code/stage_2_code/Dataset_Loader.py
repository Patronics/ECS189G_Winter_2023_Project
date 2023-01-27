'''
Concrete IO class for a specific dataset
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

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X = []
        y = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X.append(elements[1:])
            y.append(elements[0])
        f.close()
        XTest = []
        yTest = []
        f2 = open(self.dataset_source_folder_path + self.dataset_test_file_name, 'r')
        for line in f2:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            XTest.append(elements[1:])
            yTest.append(elements[0])
        f2.close()
        return {'X': X, 'y': y, 'XTest': XTest, 'yTest': yTest}
