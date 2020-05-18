from PIL import Image
from pytesseract import image_to_string
import os
import arg
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='')
args = parser.parse_args()
input_d = args.input_dir
os.mkdir('../data/Image_with_Text')
image_list = os.listdir(input_d)
for i in image_list:
    
    img = Image.open(os.path.join(input_d,i))
    text = image_to_string(img)
    text = re.sub('\n',' ', text)
    if len(text.split(' ')) >= 2:
        shutil.move(os.path.join(input_d,i),os.path.join('../data/Image_with_Text',i))
        
        
    
