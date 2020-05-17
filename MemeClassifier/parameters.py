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

hidden_size2 = 50 # hidden size for image feature
hidden_size = 50 # hidden size for superimpoed text feature
t_d='/data/yuhao/web_sci/meme_classifier/test/meme' # test_directory

threshold =0.31037354
mean = [0.579662,0.5555058,0.5413896]
std =[0.3494197,0.3469673,0.35115704]
model_dir = '/data/yuhao/checkpoint/model.pth'

def write_log(dir_,log):
    """
    write log file
    Input: dir_: the directory for writing the log file
           log: the string log
    """
    with open(dir_,'a') as file:
        file.write(log)
        file.write('\n')
    file.close()
