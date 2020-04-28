"""

This is a file for meme_classifier model

"""

import torch 
import torch.nn as nn 
import json

import torch.nn.functional as F
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import ImageFile
from torch import optim


class Baseline_1(nn.Module):
    """
    This model only contains Image feature
    """
    def __init__(self,backbone,weight_matrix,hidden_size2, hidden_size ,batch_size):
        """
        Input:
            backbone: This the image feature extractor, we used pretrained model for this.
            weight_matrix: (Tensor) This is matrix for initilize the word embedding layer
            hidden_size2: size for first image fc output
            hidden_size: this is size for LSTM hidden layer and this is the feature size for superimposed text on the image
            batch_size: batch_size
        """
        super(Baseline_1, self).__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.hidden_size2 = hidden_size2
#        for param in self.backbone:
#            param.requires_grad = False
        self.fc1 = nn.Linear(2048,self.hidden_size2,bias = True)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(self.hidden_size2)
        ## final extractor
        self.fc2 = nn.Linear(self.hidden_size2,400,bias=True)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(400)
        self.fc3 = nn.Linear(400,200,bias=True)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(200)
        self.fc4 = nn.Linear(200,1,bias =True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x,y,h0,c0,status,z):
        """
        Input: 
            x: image input. type: tensor after transform 
            y: text superimposed over image. type:tensor (batch,seq_len) integer. 
        """
        ## branch of image feature
        out = self.backbone(x)
        out = self.fc1(out.view([-1,2048]))
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.bn3(out)
        out = self.fc4(out)
        out = self.sigmoid(out.view([-1]))
        return(out)

class Baseline_2(nn.Module):

    """
    This model only contains textual feature
    """
    def __init__(self,backbone,weight_matrix,hidden_size2, hidden_size ,batch_size):
        """
        Input:
            backbone: This the image feature extractor, we used pretrained model for this.
            weight_matrix: (Tensor) This is matrix for initilize the word embedding layer
            hidden_size2: size for first image fc output
            hidden_size: this is size for LSTM hidden layer and this is the feature size for superimposed text on the image
            batch_size: batch_size
        """
        super(Baseline_2, self).__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.hidden_size2 = hidden_size2
#        for param in self.backbone:
#            param.requires_grad = False
        self.fc1 = nn.Linear(50,self.hidden_size2,bias = True)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(self.hidden_size2)
        ## extract text feature:
        self.input_size = weight_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(weight_matrix)
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size ,batch_first = True)
        ## final extractor
        self.fc2 = nn.Linear(self.hidden_size2,400,bias=True)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(400)
        self.fc3 = nn.Linear(400,200,bias=True)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(200)
        self.fc4 = nn.Linear(200,1,bias =True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x,y,h0,c0,status,z):
        """
        Input: 
            x: image input. type: tensor after transform 
            y: text superimposed over image. type:tensor (batch,seq_len) integer. 
        """
        out = self.embedding(y) # size (batch,seq_len, embedding_size)
        out = torch.mean(out,1)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.bn3(out)
        out = self.fc4(out)
        out = self.sigmoid(out.view([-1]))
        return(out)


class Meme_classifier(nn.Module):
    def __init__(self,backbone,weight_matrix,hidden_size2, hidden_size ,batch_size):
        """
        Input:
            backbone: This the image feature extractor, we used pretrained model for this.
            weight_matrix: (Tensor) This is matrix for initilize the word embedding layer
            hidden_size2: size for first image fc output
            hidden_size: this is size for LSTM hidden layer and this is the feature size for superimposed text on the image
            batch_size: batch_size
        """
        super(Meme_classifier, self).__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.hidden_size2 = hidden_size2
#        for param in self.backbone:
#            param.requires_grad = False
        self.fc1 = nn.Linear(2048,self.hidden_size2,bias = True)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(self.hidden_size2 + 50 )
        ## extract text feature:
        self.input_size = weight_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(weight_matrix)
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size ,batch_first = True)
        ## final extractor
        self.fc2 = nn.Linear(50+self.hidden_size2,400,bias=True)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(400)
        self.fc3 = nn.Linear(400,200,bias=True)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(200)
        self.fc4 = nn.Linear(200,1,bias =True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x,y,h0,c0,status,z):
        """
        Input: 
            x: image input. type: tensor after transform 
            y: text superimposed over image. type:tensor (batch,seq_len) integer. 
        """
        ## branch of image feature
        out = self.backbone(x)
        out = self.fc1(out.view([-1,2048]))
        ## brance of text feature
        out1 = self.embedding(y) # size (batch,seq_len, embedding_size)
        out1 = torch.mean(out1,1)
#        out2 = self.embedding(z)
#        out2 = torch.mean(out2,1)
#        out1,(hn,cn) = self.lstm(out1,(h0,c0))
#        out1 = hn[0,:,:] # output of superimposed text feature (batch_size, hidden_size)
#        out2,(hn,cn) = self.lstm(out2,(h0,c0))
#        out2 = hn[0,:,:]
        out = torch.cat([out,out1],1) # concatenate two tensor according to second dimension
        out = self.relu1(out)
        out = self.bn1(out)
#        out = self.dp(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.bn2(out)
#        out = self.dp(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.bn3(out)
#        out = self.dp(out)
        #retweet = retweet.view([-1,1])
        #fc = fc.view([-1,1])
#        out = torch.cat([out,status],1)
        #out = torch.cat([retweet,out,fc],1)
        out = self.fc4(out)
        #out = F.log_softmax(input=out,dim = 1)
        out = self.sigmoid(out.view([-1]))
        return(out)
