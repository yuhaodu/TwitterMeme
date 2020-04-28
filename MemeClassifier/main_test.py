"""

This is a file for training the meme_classifier which combines the image feature and text feature.


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
import csv
import matplotlib.pyplot as plt
import help as hp
ImageFile.LOAD_TRUNCATED_IMAGES = True

test_root = '/data/yuhao/web_sci/meme_classifier/test'


model_dir = '/data/yuhao/web_sci/meme_classifier/checkpoint/Baseline_2.pth'

threshold =0.31037354

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

composed = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[mean_1, mean_2, mean_3],std=[std_1, std_2, std_3])])

composed_vali = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[mean_1, mean_2, mean_3],std=[std_1, std_2, std_3])])



# load test_dataset
dic_3 =json.load(open('/data/yuhao/web_sci/meme_classifier/test/dic1.json','rb'))
test_dataset = td.TwitterDataset(test_root,word2idx,dic_3,dict_stat,cu.max_len,transform = composed_vali)
model = mm.Baseline_2(backbone,matrix,cu.hidden_size2,cu.hidden_size,cu.batch_size)


#loss = nn.NLLLoss()
loss = nn.BCELoss()

test_loss_p = 1000000
index = 0

name_url_dict =pickle.load(open('/data/yuhao/web_sci/meme_classifier/data/all_twitter.pkl','rb'))
def write_mis_csv(name_list,output_dir):
    """
    write misclassified image into csv file
    name_list: misclassification image name list
    """
    dic_l = []
    for i in name_list:
        if len(i.split('_')) == 4:
            index = int(i.split('_')[-2])
            dic = {}
            dic['media_url'] = df['media_url'][index]
            dic['name'] = i
            dic['id_str'] = df['id_str'][index]
            dic.update(dict_stat[i])
            dic_l.append(dic)

        else:
            index = int(i.split('_')[-2])
            dic_pkl = name_url_dict[index]
            dic = {}
            dic['media_url'] = dic_pkl['media_url']
            dic['name'] = i
            dic['id_str'] = dic_pkl['id_str']
            dic.update(dict_stat[i])
            dic_l.append(dic)
    keys = dic_l[1].keys()
    with open(output_dir,'w') as csv_file:
        writer = csv.DictWriter(csv_file,keys)
        writer.writeheader()
        writer.writerows(dic_l)

def extract_dataframe(df,name_list,dir_):
    """
    df: dataframe 
    name_list: name_list of misclassified image
    dir_: output dir
    """ 
    df = df.set_index(['name'])
    out_df = df.loc[name_list]
    out_df.to_csv(dir_)


gpu = True # use GPU or not
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['model_state'])
test_loss2,recall,precision,accuracy,fn,fp,roc,tp = tvt.test(model,loss,gpu,test_dataset,threshold)
log_3 = 'Test ----------------Recall is : {}'.format(recall)
log_4 = 'Test ----------------Precision is : {}'.format(precision)
log_5 = 'Test ----------------Accuracy is {}'.format(accuracy)
log_6 = 'Test ----------------Test loss is {}'.format(test_loss2)
(p,r,t) = roc
#out = [(p[i],r[i],t[i]) for i in range(len(r)) if r[i] > 0.7 and p[i] > 0.7  ]
#print('feasible cut off is: {}'.format(out))
#cu.write_log(log_dir, log_3)
#cu.write_log(log_dir, log_4)
#cu.write_log(log_dir, log_5)
#cu.write_log(log_dir, log_6)
print(log_3)
print(log_4)
print(log_5)
print(log_6)
fn = [i.split('.')[-2][:-2]+'.png' for i in fn]
fp = [i.split('.')[-2][:-2] + '.png' for i in fp]
tp = [i.split('.')[-2][:-2] + '.png' for i in tp]
#df = pd.read_csv('/data/yuhao/web_sci/meme_classifier/data/vali.csv')
#extract_dataframe(df,fn,'/data/yuhao/web_sci/meme_classifier/false_negative.csv')
#extract_dataframe(df,fp,'/data/yuhao/web_sci/meme_classifier/false_positive.csv')  
#extract_dataframe(df,tp,'/data/yuhao/web_sci/meme_classifier/true_positive.csv')
print('correct!')
