"""
This is a file for twitter dataset


"""

from collections import Counter
import regex as re
from pytesseract import image_to_string
import torch 
import torch.nn as nn 
import json
import zipfile
import torch.nn.functional as F
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from skimage import transform
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import ImageFile
from torch import optim
import caption_preprocessing as cp
import io
class TwitterDataset(Dataset):
    """
    twitter image dataset
    Load the image and its corresponding caption and original text.
    And this is for training validation and testing.
    """
    def __init__(self, dir_, dict_, dict_2 , dict_3,len_,transform = None):
        """
        Args:
            dir_ : the directory for data
            transform(callable, optinal) : Optional transform to be applied on a sample
            dict_ : the dictionary for word_to_idx
            dict_2 : dictionary (name : [superimposedtext,original text])
            dict_3 ; dictionary  (name: statistics)
            len_: maximum len for superimposed text
        """
        meme_dir = dir_ + '/meme'
        nmeme_dir = dir_ + '/not_meme'
        not_caption = dir_ + '/not_caption'
        meme_newdir = dir_ + '/meme_new'
        reddit_meme = dir_ + '/reddit_memes'
        meme1 = dir_ + '/meme1'
        quote = dir_ +'/quote'
        update_not_meme = dir_ + '/not_meme1'
#        self.image_dir = []
        self.image_dir = glob.glob(meme_dir + '/' + '*.png')
        self.image_dir.extend(glob.glob(nmeme_dir + '/' + '*.png'))
        self.image_dir.extend(glob.glob(not_caption + '/*.png'))
        self.image_dir.extend(glob.glob(meme_newdir + '/*.png'))
#        self.image_dir.extend(glob.glob(reddit_meme + '/*.png'))
        self.image_dir.extend(glob.glob(meme1 + '/*.png'))
        self.image_dir.extend(glob.glob(quote + '/*.png'))
        self.image_dir.extend(glob.glob(update_not_meme + '/*.png'))
        self.transform = transform
        self.dict_ = dict_
        ## preprocess the dict_2:
        caption = [value[0] for key,value in dict_2.items()]
        output = cp.preprocess_data_spacy(caption) # preprocess superimposed text
        original = [value[1] for key,value in dict_2.items()]
        output2 = cp.preprocess_data_spacy(original) # preprocess original text 
        self.dict_2 = {key:output[index] for index,key in enumerate([*dict_2.keys()])}  # dictionary name: superimposed text
        self.dict_3 = {key:output2[index] for index,key in enumerate([*dict_2.keys()])} # dictionary name: original tweet if reddit meme then is ' '
        self.len_ = len_
        self.transform = transform
        keys = [*dict_3.keys()]
        rt_c = [dict_3[key]['rt_c'] for key in keys ]
        self.max_rt = max(rt_c)
        f_c = [dict_3[key]['f_c'] for key in keys ]
        self.f_c = max(f_c)
        self.dict_4 = dict_3
    def __len__(self):
        return len(self.image_dir) 

    def __getitem__(self, idx):
        """
        Input the index of image and return its label and there is more stuff then add it here
        """
        img_name = self.image_dir[idx]
        label = int(img_name.split('.')[-2].split('_')[-1])
        name = img_name.split('/')[-1]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        image = image[:3,:,:]
        if image.shape[0] != 3:
           add = 3-image.shape[0]
           if add == 1:
              image = torch.cat([image,image[:1,:,:]],0)
           else:
              image = image.repeat(3,1,1)
        text_1,leng = self.preprocess_text(name)
        status = self.preprocess_text2(name)
        text_2,leng_2 = self.preprocess_text_3(name)
#        retweet = self.dict_4[name]['rt_c']/self.max_rt
#        fc = self.dict_4[name]['f_c']/self.f_c
        return (image,label,name,text_1,leng,status,text_2,leng_2)#,retweet,fc)

    def preprocess_text(self,name):
            """
            Aim:  preprocess the data and trim them and change them into index that match the word embedding network
            Input:
                 name: name of the file
            """
            text_1 = self.dict_2[name]
            #print(text_1)
            #print('________')
            text_1 = text_1.split(' ')
            idx = np.zeros(self.len_)
            start = 0
            for i in text_1:
                try:
                    index = self.dict_[i]
                    idx[start] = index
                    start += 1
                    if start == self.len_:
                       break
                except:
                    continue
            return idx,start
     
    def preprocess_text_3(self,name):
            """
            Aim:  preprocess the data and trim them and change them into index that match the word embedding network
            Input:
                 name: name of the file
            """
            text_1 = self.dict_3[name]
            #print(text_1)
            #print('________')
            text_1 = text_1.split(' ')
            idx = np.zeros(self.len_)
            start = 0
            for i in text_1:
                try:
                    index = self.dict_[i]
                    idx[start] = index
                    start += 1
                    if start == self.len_:
                       break
                except:
                    continue
            return idx,start
    def preprocess_text2(self,name):
            """
            Aim:  preprocess the data and trim them and change them into index that match the word embedding network
            Input:
                 name: name of the file
            """
            tweet = self.dict_3[name]
            status = tweet.split(' ')[0]
            if status == 'rt':
               status = 1
            else:
                status = 0
            return status


class TwitterDataset_final(Dataset):
    """
    twitter image dataset
    Load the image and its corresponding caption and original text.
    And this is for training validation and testing.
    """
    def __init__(self,dir_,dict_,dict_2,len_,transform = None):
        """
        Args:
            dir_ : the directory for data
            transform(callable, optinal) : Optional transform to be applied on a sample
            dict_ : the dictionary for word_to_idx
            dict_2 : dictionary (name_to_superimposedtext,original text)
            len_: maximum len for superimposed text
        """
        self.image_dir = zipfile.ZipFile(dir_).namelist()
        self.file = zipfile.ZipFile(dir_)
        self.transform = transform
        self.dict_ = dict_
        ## preprocess the dict_2:
        caption = [value[0] for key,value in dict_2.items()]
        output = cp.preprocess_data_spacy(caption) # preprocess superimposed text
        original = [value[1] for key,value in dict_2.items()]
        output2 = cp.preprocess_data_spacy(original) # preprocess original text 
        self.dict_2 = {key:output[index] for index,key in enumerate([*dict_2.keys()])}  # dictionary name: superimposed text
        self.dict_3 = {key:output2[index] for index,key in enumerate([*dict_2.keys()])} # dictionary name: original tweet if reddit meme then is ' '
        self.len_ = len_
        self.transform = transform
    def __len__(self):
        return len(self.image_dir) 

    def __getitem__(self, idx):
        """
        Input the index of image and return its label and there is more stuff then add it here
        """
        img_name = self.image_dir[idx]
        file_ = self.file
        image  = Image.open(io.BytesIO(file_.read(img_name))).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image = image[:3,:,:]
        if image.shape[0] != 3:
           add = 3-image.shape[0]
           if add == 1:
              image = torch.cat([image,image[:1,:,:]],0)
           else:
              image = image.repeat(3,1,1)
        text_1,leng = self.preprocess_text(img_name)
        status = self.preprocess_text2(img_name)
        text_2,leng_2 = self.preprocess_text_3(img_name)
        return (image,img_name,text_1,leng,status,text_2,leng_2)#,retweet,fc)

    def preprocess_text(self,name):
            """
            Aim:  preprocess the data and trim them and change them into index that match the word embedding network
            Input:
                 name: name of the file
            """
            text_1 = self.dict_2[name]
            text_1 = text_1.split(' ')
            idx = np.zeros(self.len_)
            start = 0
            for i in text_1:
                try:
                    index = self.dict_[i]
                    idx[start] = index
                    start += 1
                    if start == self.len_:
                       break
                except:
                    continue
            return idx,start
     
    def preprocess_text_3(self,name):
            """
            Aim:  preprocess the data and trim them and change them into index that match the word embedding network
            Input:
                 name: name of the file
            """
            text_1 = self.dict_3[name]
            #print(text_1)
            #print('________')
            text_1 = text_1.split(' ')
            idx = np.zeros(self.len_)
            start = 0
            for i in text_1:
                try:
                    index = self.dict_[i]
                    idx[start] = index
                    start += 1
                    if start == self.len_:
                       break
                except:
                    continue
            return idx,start
    def preprocess_text2(self,name):
            """
            Aim:  preprocess the data and trim them and change them into index that match the word embedding network
            Input:
                 name: name of the file
            """
            tweet = self.dict_3[name]
            status = tweet.split(' ')[0]
            if status == 'rt':
               status = 1
            else:
                status = 0
            return status

class TwitterDataset3(Dataset):
    """
    twitter image dataset
    Load the image and its corresponding caption and original text.
    And this is for training validation and testing.
    """
    def __init__(self, dir_, dict_,dict_2,len_,transform = None):
        """
        Args:
            dir_ : the directory for data
            transform(callable, optinal) : Optional transform to be applied on a sample
            dict_ : the dictionary for word_to_idx
            len_: maximum len for superimposed text
        """
        self.image_dir = glob.glob(dir_ + '/' + '*.png')
        self.transform = transform
        self.dict_ = dict_
        ## preprocess the dict_2:
        caption = [value[0] for key,value in dict_2.items()]
        output = cp.preprocess_data_spacy(caption) # preprocess superimposed text
        self.dict_2 = {key:output[index] for index,key in enumerate([*dict_2.keys()])}  # dictionary name: superimposed text
        self.len_ = len_
    def __len__(self):
        return len(self.image_dir) 

    def __getitem__(self, idx):
        """
        Input: the index of image
        Return: its label
        """
        img_name = self.image_dir[idx]
        label = int(img_name.split('.')[-2].split('_')[-1])
        name = img_name.split('/')[-1]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        image = image[:3,:,:]
        if image.shape[0] != 3:
           add = 3-image.shape[0]
           if add == 1:
              image = torch.cat([image,image[:1,:,:]],0)
           else:
              image = image.repeat(3,1,1)
        text,leng = self.preprocess_text(name)
        return (image,text,name)

    def preprocess_text(self,name):
            """
            Aim:  preprocess the data and trim them and change them into index that match the word embedding network
            Input:
                 name: name of the file
            """
            text_1 = self.dict_2[name]
            text_1 = text_1.split(' ')
            idx = np.zeros(self.len_)
            start = 0
            for i in text_1:
                try:
                    index = self.dict_[i]
                    idx[start] = index
                    start += 1
                    if start == self.len_:
                       break
                except:
                    continue
            return idx,start