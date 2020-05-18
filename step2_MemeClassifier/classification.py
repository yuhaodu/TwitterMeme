"""
This is a file to make classification for IWT memes
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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import OrderedDict
from shutil import copyfile
from zipfile import ZipFile

input_dir = p.input_dir
threshold = p.threshold
glove_dir = p.glove_dir
dict_dir = p.dict_dir
model_dir = p.model_dir

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

trans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=p.mean,std=p.std)])

# load test_dataset
dict_=pickle.load(open(dict_dir,'rb'))
test_dataset = md.MemeDataset(input_dir,word2idx,dict_,p.max_len,transform = trans)
model = mm.MemeClassifier(backbone,matrix)

gpu = True # use GPU or not
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['model_state'])
IWTmeme_name, nonIWTmeme_name = tt.test(model,gpu,test_dataset,threshold)

try:
    os.mkdir('../data/IWTmeme')
except:
    os.rmdir('../data/IWTmeme')
    os.mkdir('../data/IWTmeme')
try:
    os.mkdir('../data/nonIWTmeme')
except:
    os.rmdir('../data/nonIWTmeme')
    os.mkdir('../data/nonIWTmeme')
for i in IWTmeme_name:
    copyfile(os.path.join(test_root,i),os.path.join('./IWTmeme',i))
for i in nonIWTmeme_name:      
    copyfile(os.path.join(test_root,i),os.path.join('./nonIWTmeme',i))

