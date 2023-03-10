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
import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
import pandas as pd
from tqdm import trange
from collections import Counter


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
        windowSize = 3
        trainLoader = generationDataset(trainDir,windowSize)
        self.data = {'train': trainLoader, 'test':trainLoader} #TODO REMOVE OR CHANGE TEST FUNCTION


        
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
        self.tokens = np.array(self.tokens, dtype=object)
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

class generationDataset(torch.utils.data.Dataset):
    def __init__(self,fileDir,windowSize):
        self.windowSize = windowSize
        # load:
        print(os.getcwd())
        with open(fileDir) as df:
            jokes = df.readlines()
        # tokenize and combine into one long list of tokens
        tokenizer = get_tokenizer('basic_english')
        self.tokens = [np.array(tokenizer(x)) for x in jokes]
        #self.tokens = self.tokens[:100]
        self.tokens = np.concatenate(self.tokens)
        tokenCounter = Counter(self.tokens)
        self.uniqueTokenList = sorted(tokenCounter, key=tokenCounter.get,reverse=True)
        self.vocab = {index:word for index, word in enumerate(self.uniqueTokenList)}
        self.reverseVocab = {word:index for index, word in enumerate(self.uniqueTokenList)}
        self.indicies = [self.reverseVocab[token] for token in self.tokens]

    def __len__(self):
        return len(self.indicies) - self.windowSize
    def __getitem__(self, index):
        return (
            torch.tensor(self.indicies[index:index+self.windowSize]),
            torch.tensor(self.indicies[index+self.windowSize]),
        )

# class generationWordLoader(nn.Module):
#     def __init__(self,filename):
#         super(generationWordLoader,self).__init__()
#         # load the file and put in the format in the sample code above
#         print(os.getcwd())
#         with open(filename) as df:
#             jokes = df.readlines()

#         # tokenizer and embedding setup
#         tokenizer = get_tokenizer('basic_english')
#         self.tokens = [np.array(tokenizer(x)) for x in jokes]
#         self.tokens = self.tokens[:100]
#         wordBag = np.concatenate(self.tokens)
#         self.tokens = np.array(self.tokens, dtype=object)
        
#         wordCounter = Counter(wordBag)
#         self.uniqueWordList = sorted(wordCounter, key=wordCounter.get,reverse=True)

#         self.vocab = {index:word for index, word in enumerate(self.uniqueWordList)}
#         self.reverseVocab = {word:index for index, word in enumerate(self.uniqueWordList)}

#         self.indicies = []
#         for i in range(self.tokens.shape[0]):
#             self.indicies.append([self.reverseVocab[token] for token in self.tokens[i]])
        
#         self.indicies = [torch.tensor(data) for data in self.indicies]

#         print()

#     def getData(self):
#         return self.indicies

#     def forward(self):
#         return self.indicies


