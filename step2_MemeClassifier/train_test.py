"""
This is a function for train_test function
"""
import torch
import torch.nn as nn
import json
import parameters as p
import torch.nn.functional as F
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from torch.optim.lr_scheduler import MultiStepLR
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import ImageFile
from torch import optim
import time
from sklearn.metrics import recall_score, precision_score, accuracy_score,roc_curve,roc_auc_score,precision_recall_curve,average_precision_score
import pickle


def train(model, loss, gpu, optimizer, dataset):
    """
    This is the function for training the model
    Input: model: training model
           loader: dataloader
           criterion: critertion  loss
           gpu: using gpu or not (binary)
           dataset: dataset  training
 
    """
    dataloader = DataLoader(dataset,batch_size=p.batch_size, shuffle=True,num_workers=4)
    model.train()
    current_loss = 0
    index = 1
    for (train_,text_,name_,label_) in dataloader:
        index += 1
        optimizer.zero_grad()
        if gpu:
            train_, text_,label_= train_.cuda(),text_.type(torch.LongTensor).cuda(),label_.cuda()
            model = model.cuda() # put data into GPU
            output_ = model.forward(train_,text_)
            label_ = label_.view([-1,1]).type(torch.FloatTensor)
            output_ = output_.view([-1,1]).type(torch.FloatTensor)
            loss_ = loss(output,label)
            loss_.cuda() # send loss to GPU
        loss_.backward()
        optimizer.step()
        current_loss += loss_.item() * train_.shape[0]
    epoch_loss = current_loss / len(dataset)
    return model,epoch_loss,optimizer

def test(model,gpu,dataset,threshold):
    """
    This is the function for evaluating the model
    Input: model: evaluation model
           loader: dataloader
           criterion: critertion for loss
           gpu: using gpu or not (binary)
           dataset: dataset for evaluation
    Output: Directory_IWTmeme: 
            Directory_nonIWTmeme:
    """
    dataloader = DataLoader(dataset, batch_size = p.batch_size, shuffle = True, num_workers = 4)

    model.eval()
    current_loss = 0
    target = []
    output = []
    preds = []
    name = []
    for (image_,text_ ,name_) in dataloader:
        image_,text_ = image_.cuda(),text_.type(torch.LongTensor).cuda()
        model = model.cuda() # put data into GPU
        output_ = model.forward(image_,text_).type(torch.FloatTensor)
        preds.extend(1-output_.detach().cpu().numpy())
        name.extend(list(name_))
    preds = [True if i > threshold else False for i in preds ]
    IWTmeme_name = [i for index,i in enumerate(name) if preds[index]]
    nonIWTmeme_name = [i for index,i in enumerate(name) if not preds[index]]
    return IWTmeme_name, nonIWTmeme_name