"""
This is a file for twitter dataset


"""

import torch 
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from PIL import Image


class MemeDataset(Dataset):
    """
    MemeImage dataset
    Load the image and its corresponding superimposed texts.
    """
    def __init__(self, dir_, dict_,dict_2,len_,transform = None):
        """
        Args:
            dir_ : the directory for data
            transform(callable, optinal) : Optional transform to be applied on a sample
            dict_ : Type: dictionary. Key: word. Value: index of word in word embedding
            dict_2 : Type: dictionary. Key:image_name. Value: preprocessed superimposed texts
            len_: maximum length of superimposed texts
        """
        self.dir_ = dir_ 
        self.image_list = os.listdir(dir_)
        self.transform = transform
        self.dict_ = dict_
        ## preprocess the dict_2:
        self.dict_2 = dict_2
        self.len_ = len_
    def __len__(self):
        return len(self.image_list) 

    def __getitem__(self, idx):
        """
        Input: the index of image
        Return: its label
        """
        img_name = self.image_list[idx]
        image = Image.open(os.path.join(self.dir_,img_name))
        if self.transform:
            image = self.transform(image)
        image = image[:3,:,:]
        if image.shape[0] != 3:
           add = 3-image.shape[0]
           if add == 1:
              image = torch.cat([image,image[:1,:,:]],0)
           else:
              image = image.repeat(3,1,1)
        text,leng = self.preprocess_text(img_name)
        return (image,text,img_name)

    def preprocess_text(self,name):
            """
            Aim: 
            Input:
                 name: name of the image
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
        
        
class MemeDataset_train(Dataset):
    """
    MemeImage dataset
    Load the image and its corresponding superimposed texts.
    """
    def __init__(self, dir_, dict_,dict_2,dict_3,len_,transform = None):
        """
        Args:
            dir_ : the directory for data
            transform(callable, optinal) : Optional transform to be applied on a sample
            dict_ : Type: dictionary. Key: word. Value: index of word in word embedding
            dict_2 : Type: dictionary. Key:image_name. Value: preprocessed superimposed texts
            len_: maximum length of superimposed texts
        """
        self.dir_ = dir_ 
        self.image_list = os.listdir(dir_)
        self.transform = transform
        self.dict_ = dict_
        ## preprocess the dict_2:
        self.dict_2 = dict_2
        self.dict_3 = dict_3
        self.len_ = len_
    def __len__(self):
        return len(self.image_list) 

    def __getitem__(self, idx):
        """
        Input: the index of image
        Return: its label
        """
        img_name = self.image_list[idx]
        image = Image.open(os.path.join(self.dir_,img_name))
        if self.transform:
            image = self.transform(image)
        image = image[:3,:,:]
        if image.shape[0] != 3:
           add = 3-image.shape[0]
           if add == 1:
              image = torch.cat([image,image[:1,:,:]],0)
           else:
              image = image.repeat(3,1,1)
        text,leng = self.preprocess_text(img_name)
        label = self.dict_3[img_name]
        return (image,text,img_name,label)

    def preprocess_text(self,name):
            """
            Aim: 
            Input:
                 name: name of the image
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
