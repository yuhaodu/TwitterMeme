from PIL import Image
from pytesseract import image_to_string
import os
import argparse
import pickle
import text_preprocessing as tp
import regex as re
import shutil
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='')
args = parser.parse_args()
input_d = args.input_dir
os.mkdir('../data/Image_with_Text')
image_list = os.listdir(input_d)
text_list = []
name_list = []
for i in tqdm(image_list):
    
    img = Image.open(os.path.join(input_d,i))
    text = image_to_string(img)
    text = re.sub('\n',' ', text)
    if len(text.split(' ')) >= 2:
        text_list.append(text)
        name_list.append(i)
        shutil.move(os.path.join(input_d,i),os.path.join('../data/Image_with_Text',i))

text_list = tp.preprocess(text_list)
dict_name_text = dict(zip(name_list,text_list))
pickle.dump(dict_name_text,open('../data/name_text.pkl','wb'))

        
    
