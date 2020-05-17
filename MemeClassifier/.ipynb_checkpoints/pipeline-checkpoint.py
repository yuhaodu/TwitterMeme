"""

This is a file for training the meme_classifier which combines the image feature and text feature.

# skip 114

"""
import pandas as pd 
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
from skimage import  transform
import io
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
import csv
import matplotlib.pyplot as plt
import help as hp
import zipfile
import sys,traceback

ImageFile.LOAD_TRUNCATED_IMAGES = True

model_dir = '/data/yuhao/web_sci/meme_classifier/checkpoint/model_text_image.pth'

threshold = 0.31037354 #0.2543277

## Loading the pretrained word embedding matrix and backbone for model initiation

glove_dir = '/data/yuhao/web_sci/meme_classifier/glove.6B/'
vectors = bcolz.open(glove_dir + '6B.50.dat')[:]
words = pickle.load(open(glove_dir + '6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(glove_dir + '6B.50_idx.pkl', 'rb'))
matrix = torch.FloatTensor(vectors)
resnet = models.resnet50(pretrained = True) # load pretrained the resnet50 model
backbone = nn.Sequential(*list(resnet.children())[:-1]) 
dict_stat =json.load(open('/data/yuhao/web_sci/meme_classifier/data/name_statistic.json','rb'))
## 

mean_1 = 0.579662
mean_2 = 0.5555058
mean_3 = 0.5413896
std_1 = 0.3494197
std_2 = 0.3469673
std_3 = 0.35115704

composed_vali = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[mean_1, mean_2, mean_3],std=[std_1, std_2, std_3])])
model = mm.Meme_classifier(backbone,matrix,cu.hidden_size2,cu.hidden_size,cu.batch_size)
checkpoint = torch.load(model_dir)   # load the model
model.load_state_dict(checkpoint['model_state'])




def meme_selection(file_num,model,threshold):
    """
    Extract memes from zip file
    write the name of the image into file
    """
    model.cuda() # send model to gpu
    file_str = str(file_num)
    o_num = 5 - len(file_str)
    for i in range(o_num):
        file_str = '0' + file_str
    print(file_str)

    image_zip = '/data/yuhao/download/image_data/caption_image/{}cap.zip'.format(file_str)
    caption_file = '/data/yuhao/download/image_data/caption/{}.json'.format(file_str)
    dict_ = {}
    with open(caption_file,'r') as cap_file:
        lines = cap_file.readlines()
        for index, line in enumerate(lines):
            line_dict = json.loads(line)
            dict_.update(line_dict)
        keys = [*line_dict.keys()][0]
    len_keys = len(keys.split('_'))
    if len_keys == 0:
        choose = True           # check is key is l_index~m_index
    else:
        choose = False
    print('choose is: {}'.format(choose))
    dataset = td.TwitterDataset_final(image_zip,word2idx,dict_,cu.max_len,composed_vali)
    image_loader = DataLoader(dataset, batch_size = cu.batch_size)
    preds_l = []
    name_list = []
    save_dict = {}
    for (train_,name,text_,leng,status,text_2,leng_2) in image_loader:
        if train_.size()[0] == 1:
            continue
        train_ = train_.cuda()
        text_ = text_.type(torch.LongTensor).cuda()
        status = status.view([-1,1]).type(torch.FloatTensor).cuda()
        text_2 = text_.type(torch.LongTensor).cuda()
        h0 = torch.zeros(1,train_.shape[0],cu.hidden_size).cuda()
        c0 = torch.zeros(1,train_.shape[0],cu.hidden_size).cuda()
        output = model.forward(train_,text_,h0,c0,status,text_2)
        output = output.type(torch.FloatTensor)
        preds_l.extend(1-output.detach().cpu().numpy())
        name_list.extend(name)
    predict_label =[1 if i > threshold else 0 for i in preds_l ]
    
    name_list = np.array(name_list)
    index = [i for i in range(len(predict_label)) if predict_label[i] == 1]

    for idx,name in enumerate(name_list):
        user_id = name.split('_')[2]
        if user_id not in [*save_dict.keys()] and idx in index:
            save_dict[user_id] = [[],[]]
            save_dict[user_id][1].append(name)
        elif user_id not in [*save_dict.keys()] and idx not in index:
            save_dict[user_id] = [[],[]]
            save_dict[user_id][0].append(name)
        elif user_id in [*save_dict.keys()] and idx not in index:
            save_dict[user_id][0].append(name)
        else:
            save_dict[user_id][1].append(name)
    image_num = len(name_list)
    meme_num = len(index)
    return name_list[index],file_str,image_num, meme_num, save_dict

file_1 = open('/data/yuhao/download/image_data/missed_file_2.txt')
lines = file_1.readlines()
missed = []
for i in lines:
    missed.append(int(i))
print(missed)

for i in missed:
    start_time = time.time()
    try:
        name_list,file_str,image_num,meme_num,save_dict = meme_selection(i,model,threshold)
        end_time = time.time()
        json.dump(save_dict,open('/data/yuhao/download/image_data/addmeme_stat/{}_stat.json'.format(file_str),'w'))
        print('Using time: {}'.format(end_time - start_time))
        print('Image number is {}, meme number is {}, the proportion of meme in image is {}'.format(image_num,meme_num,meme_num/image_num))
    except:
        traceback.print_exc(file=sys.stdout)
