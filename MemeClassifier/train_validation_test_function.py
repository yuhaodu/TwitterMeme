"""
This is a function for train_validation_test function
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
from collections import Counter
import regex as re
import time
from sklearn.metrics import recall_score, precision_score, accuracy_score,roc_curve
from sklearn.metrics import roc_auc_score
import bcolz
import pickle
from pytesseract import image_to_string
import TwitterDataset as td
import MemeModel as mm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def train(model, loss, gpu, optimizer, dataset):
    """
    This is the function for training the model
    Input: model: training model
           loader: dataloader
           criterion: critertion  loss
           gpu: using gpu or not (binary)
           dataset: dataset  training
 
    """
    dataloader = DataLoader(dataset,batch_size=cu.batch_size, shuffle=True,num_workers=4)
    model.train()
    current_loss = 0
    index = 1
    for (train_,label,name,text_,leng, status,text_2,leng_2) in dataloader:
        index += 1
        optimizer.zero_grad()
        if gpu:
            train_, label, text_, status = train_.cuda(), label.cuda(), text_.type(torch.LongTensor).cuda(), status.view([-1,1]).type(torch.FloatTensor).cuda() # put data into GPU
            text_2 = text_.type(torch.LongTensor).cuda()
            #retweet = retweet.type(torch.FloatTensor).cuda()
            #fc = fc.type(torch.FloatTensor).cuda()
            model = model.cuda() # put data into GPU
            h0 = torch.zeros(1,train_.shape[0],cu.hidden_size).cuda()
            c0 = torch.zeros(1,train_.shape[0],cu.hidden_size).cuda()
            output = model.forward(train_,text_,h0,c0,status,text_2)
            #_, preds = torch.max(output,1)
            label = label.view([-1,1]).type(torch.FloatTensor)
            output = output.view([-1,1]).type(torch.FloatTensor)
            loss_ = loss(output,label)
            loss_.cuda() # send loss to GPU
        loss_.backward()
        optimizer.step()
        current_loss += loss_.item() * train_.shape[0]
    epoch_loss = current_loss / len(dataset)
    return model,epoch_loss,optimizer

def validation(model, loss, gpu, optimizer, dataset):
    """
    This is the function for evaluating the model
    Input: model: evaluation model
           loader: dataloader
           criterion: critertion for loss
           gpu: using gpu or not (binary)
           dataset: dataset for evaluation
    Output: precision, accuracy, recall, name array of false negative, name array of false positive
    """
    dataloader = DataLoader(dataset, batch_size = cu.batch_size, shuffle = True, num_workers = 4)

    model.eval()
    current_loss = 0
    target = []
    output = []
    preds_l = []
    name_list = []
    for (validation_, label, name, text_ , leng, status,text_2,leng_2) in dataloader:

        validation_, label, text_, status = validation_.cuda(), label.cuda(), text_.type(torch.LongTensor).cuda(),status.view([-1,1]).type(torch.FloatTensor).cuda() # put data into GPU
        text_2 = text_2.type(torch.LongTensor).cuda()
        #retweet = retweet.type(torch.FloatTensor).cuda()
        #fc = fc.type(torch.FloatTensor).cuda()
        model = model.cuda() # put data into GPU
        h0 = torch.zeros(1,validation_.shape[0],cu.hidden_size).cuda()
        c0 = torch.zeros(1,validation_.shape[0],cu.hidden_size).cuda()
        output_ = model.forward(validation_,text_,h0,c0,status,text_2)
        output_ = output_.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        preds = output_ > 0.5
        preds_l.extend(1-output_.detach().cpu().numpy())
        target.extend(label.cpu().numpy()) # add label to vector
        output.extend(preds.cpu().numpy()) #_, preds = torch.max(output_,1)
        output_ = output_.view([-1,1])
        label = label.view([-1,1])
        loss_ = loss(output_,label)
        name_list.extend(name)
        current_loss += loss_.item() * validation_.shape[0]

    target = np.array([i-1 for i in target]) * (-1) # change meme's label to 1 and non meme to 0
    output = np.array([i-1 for i in output]) * (-1)
#    output = (np.array(output) -1) * (-1) # change meme's label to 1 and non meme to 0 
    index_fn = [i for i in range(len(target)) if target[i] == 1 and output[i]==0]
    index_fp = [i for i in range(len(target)) if target[i] == 0 and output[i]==1]
    index_tn = [i for i in range(len(target)) if target[i] == 0 and output[i]==0]
    index_tp = [i for i in range(len(target)) if target[i]==1 and output[i]==1]

    name_list = np.array(name_list)
    recall =  recall_score(target,output)
    precision = precision_score(target,output)
    accuracy = accuracy_score(target,output)
    epoch_loss = current_loss / len(dataset)
    p,r,thresholds = precision_recall_curve(target,preds_l)
    auc = average_precision_score(target,preds_l)
    p_r = (p,r,thresholds)
    return epoch_loss,recall,precision,accuracy,name_list[index_fn],name_list[index_fp],p_r,name_list[index_tp],auc




def test(model, loss, gpu,dataset,threshold):
    """
    This is the function for evaluating the model
    Input: model: evaluation model
           loader: dataloader
           criterion: critertion for loss
           gpu: using gpu or not (binary)
           dataset: dataset for evaluation
    Output: precision, accuracy, recall, name array of false negative, name array of false positive
    """
    dataloader = DataLoader(dataset, batch_size = cu.batch_size, shuffle = True, num_workers = 4)

    model.eval()
    current_loss = 0
    target = []
    output = []
    preds_l = []
    name_list = []
    for (validation_, label, name, text_ , leng, status,text_2,leng_2) in dataloader:

        validation_, label, text_, status = validation_.cuda(), label.cuda(), text_.type(torch.LongTensor).cuda(),status.view([-1,1]).type(torch.FloatTensor).cuda() # put data into GPU
        text_2 = text_2.type(torch.LongTensor).cuda()
        #retweet = retweet.type(torch.FloatTensor).cuda()
        #fc = fc.type(torch.FloatTensor).cuda()
        model = model.cuda() # put data into GPU
        h0 = torch.zeros(1,validation_.shape[0],cu.hidden_size).cuda()
        c0 = torch.zeros(1,validation_.shape[0],cu.hidden_size).cuda()
        output_ = model.forward(validation_,text_,h0,c0,status,text_2)
        output_ = output_.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        preds = output_ > 0.5
        preds_l.extend(1-output_.detach().cpu().numpy())
        target.extend(label.cpu().numpy()) # add label to vector
        output.extend(preds.cpu().numpy()) #_, preds = torch.max(output_,1)
        output_ = output_.view([-1,1])
        label = label.view([-1,1])
        loss_ = loss(output_,label)
        name_list.extend(name)
        current_loss += loss_.item() * validation_.shape[0]

    target = np.array([i-1 for i in target]) * (-1) # change meme's label to 1 and non meme to 0
    output = np.array([i-1 for i in output]) * (-1)
#    output = (np.array(output) -1) * (-1) # change meme's label to 1 and non meme to 0 
    index_fn = [i for i in range(len(target)) if target[i] == 1 and output[i]==0]
    index_fp = [i for i in range(len(target)) if target[i] == 0 and output[i]==1]
    index_tn = [i for i in range(len(target)) if target[i] == 0 and output[i]==0]
    index_tp = [i for i in range(len(target)) if target[i]==1 and output[i]==1]
    predict_label = [1 if i > threshold else 0 for i in preds_l ]
    name_list = np.array(name_list)
    recall =  recall_score(target,predict_label)
    precision = precision_score(target,predict_label)
    accuracy = accuracy_score(target,predict_label)
    epoch_loss = current_loss / len(dataset)
    p,r,thresholds = precision_recall_curve(target,preds_l)
    auc = average_precision_score(target,preds_l)
    print('auc score is {}'.format(auc))
    p_r = (p,r,thresholds)
    return epoch_loss,recall,precision,accuracy,name_list[index_fn],name_list[index_fp],p_r,name_list[index_tp]
