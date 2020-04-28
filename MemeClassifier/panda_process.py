"""
This is a file to manipulate panda and csv

"""
import pandas as pd

dir_ = '/data/yuhao/web_sci/meme_classifier/test_sample.csv' # this is the dir of csv file


df = pd.read_csv(dir_)
print(df.columns)

def extract_row()
