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


        
    def load_class(self):
        print('loading ' + self.dataset_name + '...')
        
        #TODO: load the training and testing data into @self.data as one data structure
        trainDir = 'path/to/training/data'
        testDir = 'path/to/test/data'

        trainLoader = classificationWordLoader(trainDir)
        testLoader = classificationWordLoader(testDir)

        # set self.dataset to the tuple of loader objects.
        self.data = {'train':trainLoader,'test':testLoader}



class classificationWordLoader(nn.Module):
    def __init__(self,filedir):
        super(classificationWordLoader,self).__init__()
        # load the file and put in the format in the sample code above
        # temp code
        testData = [
            "Hello, How are you?",
            "This is a bucket, dear god, no",
            "What the fuck did you just fucking say about me, you little bitch?",
            "I’ll have you know I graduated top of my class in the Navy Seals, and I’ve been involved in numerous secret raids on Al-Quaeda, and I have over 300 confirmed kills.",
            "I am trained in gorilla warfare and I’m the top sniper in the entire US armed forces.",
            "You are nothing to me but just another target. I will wipe you the fuck out with precision the likes of which has never been seen before on this Earth, mark my fucking words."
        ]
        # a list of strings, please.
        # tokenizer and embedding setup
        tokenizer = get_tokenizer('basic_english')
        global_vectors = GloVe()
        self.x = []
        for i in testData:
            self.x.append(global_vectors.get_vecs_by_tokens(tokenizer(i), lower_case_backup=True))
        # label loading:
        # load labels here 
        # format: [0,1] or [1,0], as floats for BCE
        # temp code:
        self.yTrue = torch.tensor([[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]],dtype=torch.float)
        self.x = np.array(self.x)

    def forward(self,batchSize):
        # bucketIterator
        # sort
        lengths = []
        for i in self.x:
            lengths.append(i.shape[0])
        lengths = np.array(lengths)
        sortOrder = np.argsort(lengths)
        sortedLengths = lengths[sortOrder]
        sortedX = self.x[sortOrder]
        sortedY = self.yTrue[sortOrder]
        # split
        xList = np.array(np.array_split(sortedX,[batchSize]))
        yList = np.array(sortedY.split(batchSize))
        # shuffle
        shuffleIndex = np.random.permutation(xList.shape[0])
        xListShuffled = xList[shuffleIndex]
        yListShuffled = yList[shuffleIndex]
        # little shuffle
        for i in range(xListShuffled.shape[0]):
            shuffleIndex = np.random.permutation(xList[i].shape[0])
            xListShuffled[i] = xList[i][shuffleIndex]
            yListShuffled[i] = yList[i][shuffleIndex]
        return zip(xListShuffled,yListShuffled)
        # standard randomization
        # # randomize:
        # shuffleIndex = np.random.permutation(self.x.shape[0])
        # xShuffled = self.x[shuffleIndex]
        # yShuffled = self.yTrue[shuffleIndex]
        # # batch:
        # xList = np.array(np.array_split(xShuffled,[batchSize]))
        # yList = np.array(yShuffled.split(batchSize))
        # return zip(xList,yList)

        
        
        
        
        
        
        
        
        
        
        
    