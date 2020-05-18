#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 17:27:59 2018

@author: du

This is neural network configuration file
"""
num_epoch = 1000
batch_size = 32
milestones = [2]
max_len = 40
input_dir='../data/Image_with_Text' # input images directory
glove_dir='../data/glove.6B.zip'
dict_dir='../data/name_text.pkl'

threshold =0.31037354
mean = [0.579662,0.5555058,0.5413896]
std =[0.3494197,0.3469673,0.35115704]
model_dir = '../data/model_text_image.pth'
