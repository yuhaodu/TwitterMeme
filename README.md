# Understanding Visual Memes: an Empirical Analysis of Text Superimposed on Memes Shared on Twitter

This repository contains code and instruction to reproduce the IWT meme classifier pipeline presented in the paper **Understanding Visual Memes: an Empirical Analysis of Text Superimposed on Memes Shared on Twitter (ICWSM 2020)**. For those who want analyze IWT memes and use our code, please cite the paper! Thank you.  

Given a dataset of images shared on social media, we develop a two-step pipeline to identify IWT memes. <br />


Step 1 is  to use  the  Optical Character Recognition (OCR) engine Tesseract [1] to filter out all images that do not contain text.<br />
<li>Step 2 is to use a supervised classifier to distinguish, amongst the remaining images, those that are IWT memes from those that are not.<br />

The  overall  structure  of  our  multimodal  neural network is shown as below . First, input images are fed into pretrained neural networks to extract visual feature and textual features. We then concatenate these two feature vectors into a single mutlimodal feature representation and use a final neural network to perform classification.

<br />
![neural_network_image ](image/neural_network.png)
<br />






# Reference
[1] Smith, R., Antonova, D., & Lee, D. (2009). Adapting the Tesseract open source OCR engine for multilingual OCR. MOCR '09.
