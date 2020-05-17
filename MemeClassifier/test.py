"""

This is a file for training the meme_classifier which combines the image feature and text feature.


"""

import torch 
import torch.nn as nn 
import json
import classifier_utils as cu
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
from sklearn.metrics import recall_score, precision_score, accuracy_score
import bcolz
import pickle
from pytesseract import image_to_string
import TwitterDataset as td
import MemeModel as mm
import train_validation_test_function as tvt

load_model = True
ImageFile.LOAD_TRUNCATED_IMAGES = True

threshold = 
    
test_root = '/data/yuhao/web_sci/meme_classifier/'


model_dir = '/data/yuhao/web_sci/meme_classifier/checkpoint/model_text_image.pth'

## Loading the pretrained word embedding matrix and backbone for model initiation

glove_dir = '/projects/academic/kjoseph/download_result/glove.6B/'
vectors = bcolz.open(glove_dir + '6B.50.dat')[:]
words = pickle.load(open(glove_dir + '6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(glove_dir + '6B.50_idx.pkl', 'rb'))
matrix = torch.FloatTensor(vectors)
resnet = models.resnet50(pretrained = True) # load pretrained the resnet50 model
backbone = nn.Sequential(*list(resnet.children())[:-1]) 


mean_1 = 0.579662
mean_2 = 0.5555058
mean_3 = 0.5413896
std_1 = 0.3494197
std_2 = 0.3469673
std_3 = 0.35115704

composed_vali = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[mean_1, mean_2, mean_3],std=[std_1, std_2, std_3])])


dic_3
test_dataset = td.TwitterDataset(test_root,word2idx,dic_3,cu.max_len,transform = composed_vali)

model = mm.Meme_classifier(backbone,matrix,cu.hidden_size2,cu.hidden_size,cu.batch_size)


optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
loss = nn.NLLLoss()
scheduler = MultiStepLR(optimizer,milestones = cu.milestones,gamma = 0.1)
cu.write_log(log_dir,'-------------------------')
cu.write_log(log_dir,'-------------------------')

print('Load model:-------------')
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
index = checkpoint['epochs']
for state in optimizer.state.values():
    for k,v in state.items():
       if torch.is_tensor(v):
            state[k] = v.cuda() 

gpu = True # use GPU or not
test_loss,recall,precision,accuracy = tvt.test(model,loss,gpu,threshold,test_dataset)
log_3 = 'Test ----------------Recall is : {}'.format(recall)
log_4 = 'Test ----------------Precision is : {}'.format(precision)
log_5 = 'Test ----------------Accuracy is {}'.format(accuracy)
log_6 = 'Test ----------------Test loss is {}'.format(test_loss)
cu.write_log(log_dir, log_3)
cu.write_log(log_dir, log_4)
cu.write_log(log_dir, log_5)
cu.write_log(log_dir, log_6)
print('correct!')
