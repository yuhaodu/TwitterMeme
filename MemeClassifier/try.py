from pytesseract import  image_to_string
import glob
from PIL import Image 
list_ = glob.glob('/data/yuhao/web_sci/meme_classifier/train_3/meme1/google*')

for i in list_:
    try:
        img = Image.open(i)
        print('OK : {}'.format(i))
    except:
        print(i)
        continue 
