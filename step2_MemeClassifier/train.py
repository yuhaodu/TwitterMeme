"""
Training
"""
import torch
import torch.nn as nn
import json
import parameters as p
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import numpy as np
import pickle
import MemeDataset as md
import MemeModel as mm
import train_test as tt
import csv
import matplotlib.pyplot as plt
import help as hp
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import OrderedDict
from shutil import copyfile
from zipfile import ZipFile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',help='meme directory')
parser.add_argument('--dict_dir',help='{meme:text} dictionary')
parser.add_argument('--dict_label',help='{meme:label} dictionary')
parser.add_argument('--output_dir',help='output_model directory')
threshold = p.threshold
glove_dir = p.glove_dir
input_dir = parser.input_dir
dict_dir = parser.dict_dir
dict_dir2 = parser.dict_label
output_dir = parser.output_dir

with ZipFile(glove_dir,'r') as zip:
    zip.extract('glove.6B.50d.txt',path = '../data')

vectors = []
words = []
word2idx = {}
with open("../data/glove.6B.50d.txt", 'r') as f:
    for index,line in enumerate(f):
        values = line.split()
        words.append(values[0])
        vector = np.asarray(values[1:], "float32")
        vectors.append(vector)
        word2idx[values[0]] = index

matrix = torch.FloatTensor(vectors)
resnet = models.resnet50(pretrained = True) # load pretrained the resnet50 model
backbone = nn.Sequential(*list(resnet.children())[:-1]) 
trans = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=p.mean,std=p.std)])
dict_=pickle.load(open(dict_dir,'rb'))
dict_2=pikcle.load(open(dict_dir2,'rb'))
train_dataset = md.MemeDataset_train(input_dir,word2idx,dict_,dict_2,p.max_len,transform = trans)
model = mm.MemeClassifier(backbone,matrix)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
loss = nn.BCELoss() 

start_loss = 100000000
for i in range(cu.num_epoch):
    model,epoch_loss,optimizer = tt.train(model,loss,gpu,optimizer,train_dataset)
    log = "{}th epoch------------------- loss is: {}".format(i+1,epoch_loss)
    if epoch_loss < start_loss:
        checkpoint = {'model_state': model.state_dict(),'criterion_state': loss.state_dict(), 'optimizer_state': optimizer.state_dict(),'epochs': i+1}
        torch.save(checkpoint, '{}/model.pth'.format(output_dir))
        start_loss = epoch_loss
    else:
        break

