
## Project Overview

The goal of the project is to classify images of dogs according to their breed and to develop an algorithm that could be used as part of a mobile or web app.

There are following steps in the project

    Step 0: Import Datasets
    Step 1: Detect Humans
    Step 2: Detect Dogs
    Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
    Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
    Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
    Step 6: Algorithm for dog/human detection
    Step 7: Test algorithm
 For the results and discussions please see my blog article.  Link: https://medium.com/@zhanglijuan1016/dog-breed-classification-project-be281af5c2cc
 
 ## Project files
 ./images - contains the images to test the algorithm 
 ./requirments - requirements to run this project for each systems
 ./saved_models - trained best performance models 
 ./dog_app.ipynb - the jupyter notebook implements the project
 ./html - contain the pdf version of the blog, html of the jupyter notebook, a test file with the link to the blog
 ./extract_bottleneck_features.py - code to extract bottleneck features
 ./haarcascades - the folder contains trained classifiers for detecting objects of a particular type
 
 Please notice, i did not include the bottleneck feature files, since it is too big, you can put the relavant feature files into ./bottleneck_features.
 
 
 ## requirement:
 
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline 
import random
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from extract_bottleneck_features import *
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint 
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.applications.resnet50 import preprocess_input, decode_predictions