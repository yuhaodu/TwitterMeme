# Understanding Visual Memes: an Empirical Analysis of Text Superimposed on Memes Shared on Twitter

This repository contains code and instruction to reproduce the IWT meme classifier pipeline presented in the paper **Understanding Visual Memes: an Empirical Analysis of Text Superimposed on Memes Shared on Twitter (ICWSM 2020)**. For those who want analyze IWT memes and use our code, please cite the paper! Thank you.  

## Content

1. Introduction to two-step meme classification pipeline
2. Step 1
3. Step 2


## Introduction to two-step meme classification pipeline
Given a dataset of images shared on social media, we develop a two-step pipeline to identify IWT memes. <br />


- Step 1 is  to use  the  Optical Character Recognition (OCR) engine Tesseract [1] to filter out all images that do not contain text.<br />
- Step 2 is to use a supervised classifier to distinguish, amongst the remaining images, those that are IWT memes from those that are not.<br />

The  overall  structure  of  our  multimodal  neural network is shown as below . First, input images are fed into pretrained neural networks to extract visual feature and textual features. We then concatenate these two feature vectors into a single mutlimodal feature representation and use a final neural network to perform classification. 
![neural_network_image](image/neural_network.png)

## Step 1
After cloning the repo, run following command from this directory to conduct the step1 of the pipeline.
```
sh install.sh
cd step1_filter
python filter.py --input_dir=[image_directory]
```
The scripts will first pull down the GloVe and to install a few packages <br />
After that, running `filter.py` will forward input images to Tesseract to filter out images without any superimposed texts. The remaining images are moved into '../data/Image_with_Text' directory. Besides that, texts extracted by Tesseract are preprocessed by [SpaCy](https://spacy.io/) and stored at '../data/name_text.pkl' using dictionary form. The keys of the dictionary are the names of images and values are preprocessed texts that are extracted from them.  

## Step 2
After finishing the step1, download the Pretrained Model at [Link](https://drive.google.com/open?id=1uNju_ZNTYvqOxFbfJyHwxcMMgdsawEGG) and put the pretrained model at '../data' directory. Then run following commands from the current directory to conduct the step2 of the pipeline.
```
cd ../step2_MemeClassifier
python classification.py
```
The scipts will forward extracted images to the multimodal neural IWT meme classification network. The identified IWT memes will be put in the '../data/IWTmeme' directory and the resutls of images will be put in the '../data/nonIWTmeme' directory. 
### Train your own model
In order to train your IWT meme classifier, you have to first prepare the dataset which contains IWT memes and nonIWTimages. Then prepare the labels for the dataset, store the label information in a dictionary whose key is image name and value is label, and use pickle to store it. After these steps, run following command.
```
python train.py --input_dir=[meme_directory] --dict_dir=[meme_text dictionary pickle file directory] --dict_label=[meme_label dictionary pickle file directory] --output_dir=[model_directory]
```
Trained model will be placed in the output directory that you specified. 

## Contact
Feel free to contact me if you encounter any problem.



## Reference
[1] Smith, R., Antonova, D., & Lee, D. (2009). Adapting the Tesseract open source OCR engine for multilingual OCR. MOCR '09.
