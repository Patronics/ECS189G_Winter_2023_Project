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
        self.global_vectors = GloVe()
        self.vocab = self.global_vectors.stoi
        self.reverseVocab = self.global_vectors.itos
        self.tokens = [tokenizer(x) for x in testData]
        self.tokens = np.array(self.tokens)
        self.yTrue = torch.tensor([[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]],dtype=torch.float)
        self.yLabels = torch.tensor([1,0,1,0,1,0])

    def getData(self):
        tokens = self.tokens
        lengths = []
        for i in tokens:
            lengths.append(len(i))
        max_words = max(lengths)
        tokens = [token+[""] * (max_words-len(token))  if len(token)<max_words else token[:max_words] for token in tokens]
        tensorList = [self.global_vectors.get_vecs_by_tokens(token) for token in tokens]
        tensor = torch.stack(tensorList)
        return tensor,lengths

        

    def forward(self,batchSize):
        #sort
        lengths = []
        for i in self.tokens:
            lengths.append(len(i))
        lengths = np.array(lengths)
        sortOrder = np.argsort(lengths)
        sortedLengths = lengths[sortOrder]
        sortedTokens = self.tokens[sortOrder]
        sortedY = self.yTrue[sortOrder]
        #split
        tokenList = np.array(np.array_split(sortedTokens,[batchSize]))
        yList = np.array(sortedY.split(batchSize))
        lengthList = np.array(np.array_split(sortedLengths,[batchSize]))
        #pad 
        tensorList = []
        for i in range(tokenList.shape[0]):
            max_words = max(lengthList[i])
            tokenList[i] = [tokens+[""] * (max_words-len(tokens))  if len(tokens)<max_words else tokens[:max_words] for tokens in tokenList[i]]
            for j in range(len(tokenList[i])):
                tokenList[i,j] = self.global_vectors.get_vecs_by_tokens(tokenList[i,j])
            tensorList.append(torch.stack(tokenList[i].tolist()))
        # shuffle
        tensorList = np.array(tensorList)
        shuffleIndex = np.random.permutation(tensorList.shape[0])
        xListShuffled = tensorList[shuffleIndex]
        yListShuffled = yList[shuffleIndex]
        lengthList = lengthList[shuffleIndex]
        # little shuffle
        for i in range(xListShuffled.shape[0]):
            shuffleIndex = np.random.permutation(xListShuffled[i].shape[0])
            xListShuffled[i] = xListShuffled[i][shuffleIndex]
            yListShuffled[i] = yList[i][shuffleIndex]
            lengthList[i] = lengthList[i][shuffleIndex]
        return zip(xListShuffled,lengthList,yListShuffled)

        
        
        
        
        
        
        
        
        
