"""
This is a file to generate caption for image
"""

import pytesseract
import glob
from PIL import Image
import pickle
import re
import sys,traceback
import os
print('start')

dir2 = '/data/yuhao/web_sci/meme_classifier/test_3/*png'

dr = '/data/yuhao/web_sci/meme_classifier/test_3/'

list_2 = glob.glob(dir2)
dict_  = {}
for index,i in enumerate(list_2):
    print(index)
    name = i.split('/')[-1].split('.')[0] + '_0.png'
    try:
        image = Image.open(i)
        caption = pytesseract.image_to_string(image)
        caption = re.sub('\n' ,' ',caption)
        print(caption)
        if len(caption.split(' ')) >= 2:
            print('add')
            dict_[name] = (caption,' ')
            os.rename(i, dr + name)
        else:
            print('remove')
            os.remove(i)
    except:
        traceback.print_exc(file = sys.stdout)
        continue

pickle.dump(dict_,open('/data/yuhao/web_sci/meme_classifier/test_3.pkl','wb'))


