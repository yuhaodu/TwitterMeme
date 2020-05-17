"""

This is a file to match the text validaiton name with the (url,original_text)
"""
import pandas as pd
import TwitterDataset as td 
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
#  part one 

test_root = '/projects/academic/kjoseph/meme_classifier/test_2'
dic_3 = pickle.load(open('/projects/academic/kjoseph/download_result/newdict_tmeme.pkl','rb'))
glove_dir = '/projects/academic/kjoseph/download_result/glove.6B/'
vectors = bcolz.open(glove_dir + '6B.50.dat')[:]
words = pickle.load(open(glove_dir + '6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(glove_dir + '6B.50_idx.pkl', 'rb'))
composed = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
test_dataset = td.TwitterDataset(test_root,word2idx,dic_3,cu.max_len,transform = composed)


dir_text = '/projects/academic/kjoseph/download_result/test_sample.csv'

df = pd.read_csv(dir_text)
dic_ = {}
for (data_,label_,name,text_,leng) in test_dataset:
     url = df.loc[df['user_id_str'] == int(name.split('_')[0]),'url'].iloc[0]
     full_text = df.loc[df['user_id_str'] == int(name.split('_')[0]),'full_text'].iloc[0]
     dic_[name] = (full_text,url)
    
pickle.dump(dic_,open('/projects/academic/kjoseph/meme_classifier/testname_text_url.pkl','wb')) 

dic_1 = pickle.load(open('/projects/academic/kjoseph/meme_classifier/testname_text_url.pkl','rb'))



