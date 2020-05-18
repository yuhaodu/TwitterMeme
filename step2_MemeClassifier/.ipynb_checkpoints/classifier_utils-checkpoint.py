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
