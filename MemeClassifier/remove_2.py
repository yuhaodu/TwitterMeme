"""
This is file to remove the image len(name.split('_')) == 2
"""
import glob
import os

dir_ = '/data/yuhao/web_sci/meme_classifier/validation_3/meme/*png'

list_1 = glob.glob(dir_)
list_1 = [i for i in list_1 if len(i.split('/')[-1].split('_')) == 2]
for i in list_1:
    os.remove(i)
    print(i)
