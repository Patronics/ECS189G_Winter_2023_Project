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

from codes.base_class.dataset import dataset
import torch
import torch.nn as nn
import numpy as np
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
import pandas as pd
from tqdm import trange


class Text_Dataset(dataset):
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.datasetName = dName

    def load(self):
        if self.datasetName == "Class":
            self.load_class()
        else:
            self.load_gen()

    def load_gen(self):
        print('loading ' + self.dataset_name + '...')
        #TODO: load the training and testing data into @self.data as one data structure
        trainDir = self.dataset_source_folder_path + 'generated_stage_4_data/joke_data_clean'
        trainLoader = generationWordLoader(trainDir)
        self.data = {train: trainLoader}


        
    def load_class(self):
        print('loading ' + self.dataset_name + '...')
        
        #TODO: load the training and testing data into @self.data as one data structure
        trainDir = self.dataset_source_folder_path + 'generated_stage_4_data/review_data_clean_train'
        testDir = self.dataset_source_folder_path + 'generated_stage_4_data/review_data_clean_test'

        trainLoader = classificationWordLoader(trainDir)
        testLoader = classificationWordLoader(testDir)

        # set self.dataset to the tuple of loader objects.
        self.data = {'train':trainLoader,'test':testLoader}



class classificationWordLoader(nn.Module):
    def __init__(self,filedir):
        super(classificationWordLoader,self).__init__()
        # load the file and put in the format in the sample code above
        print(os.getcwd())
        df = pd.read_csv(filedir,delimiter=',')
        #df = df.iloc[10000:15000]
        reviews = df['review'].to_list()

        # tokenizer and embedding setup
        tokenizer = get_tokenizer('basic_english')
        self.global_vectors = GloVe(name='6B',dim=50)
        self.vocab = self.global_vectors.stoi
        self.reverseVocab = self.global_vectors.itos
        self.tokens = [tokenizer(x) for x in reviews]
        self.tokens = np.array(self.tokens)
        rawLabels = df['label'].to_list()
        self.yLabels = torch.tensor(rawLabels)
        labelsOneHot = [[1,0] if i == 0 else [0,1] for i in rawLabels]
        self.yTrue = torch.tensor(labelsOneHot,dtype=torch.float)

        self.lengths = []
        for i in self.tokens:
            self.lengths.append(len(i))
        
        max_words = max(self.lengths)
        tokens = [token+[""] * (max_words-len(token))  if len(token)<max_words else token[:max_words] for token in self.tokens]
        
        self.x = torch.zeros(len(tokens),max_words,50)
        for i in trange(len(tokens)):
            self.x[i] = self.global_vectors.get_vecs_by_tokens(tokens[i])
        self.lengths = torch.tensor(self.lengths)

    def getData(self):
        return self.x,self.lengths

    def forward(self,batchSize):
        promptList = torch.randperm(self.x.shape[0])
        xShuffle = self.x[promptList]
        yShuffle = self.yTrue[promptList]
        lengthShuffle = self.lengths[promptList]
        # split into batches
        xList = xShuffle.split(batchSize)
        yList = yShuffle.split(batchSize)
        lengthList = lengthShuffle.split(batchSize)
        return zip(xList,lengthList,yList)


class generationWordLoader(nn.Module):
    def __init__(self,filename):
        super(generationWordLoader,self).__init__()
        # load the file and put in the format in the sample code above
        print(os.getcwd())
        with open(filename) as df:
            jokes = df.readlines()

        # tokenizer and embedding setup
        tokenizer = get_tokenizer('basic_english')
        self.global_vectors = GloVe(name='6B',dim=50)
        self.vocab = self.global_vectors.stoi
        self.reverseVocab = self.global_vectors.itos
        self.tokens = [tokenizer(x) for x in jokes]
        self.tokens = np.array(self.tokens)
        
        self.lengths = []
        for i in self.tokens:
            self.lengths.append(len(i))
        
        max_words = max(self.lengths)
        tokens = [token+[""] * (max_words-len(token))  if len(token)<max_words else token[:max_words] for token in self.tokens]
        
        self.x = torch.zeros(len(tokens),max_words,50)
        for i in trange(len(tokens)):
            self.x[i] = self.global_vectors.get_vecs_by_tokens(tokens[i])
        self.lengths = torch.tensor(self.lengths)

    def getData(self):
        return self.x,self.lengths

    def forward(self,batchSize):
        promptList = torch.randperm(self.x.shape[0])
        xShuffle = self.x[promptList]
        lengthShuffle = self.lengths[promptList]
        # split into batches
        xList = xShuffle.split(batchSize)
        lengthList = lengthShuffle.split(batchSize)
        return zip(xList,lengthList,yList)


