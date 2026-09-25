r'''
    testing the connection to the database

    everyone's path to kaggle dataset: (add yours below)
    Ali: "C:\Users\AlexisBragg\loaf-cache\set_1\images\loaf"
    Connor:
    Conner:
    Nick: 
    Angelina: 
    
'''

# run these 2 lines once to install the required packages and then comment them out
# import sys, subprocess
# subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "pandas", "torch", "scikit-learn"]) # install required packages

# imports
from pathlib import Path
import pathlib # for path manipulation
import cv2 # open cv2 for image processing
import numpy as np 
import pandas as pd # dataframe manipulation
import torch # pytorch for deep learning
import sklearn # sklearn for machine learning

# dataset path
dataset_path_loafing = pathlib.Path(r"C:\Users\AlexisBragg\loaf-cache\set_1\images\loaf") # change this to your local path
dataset_path_not_loafing = pathlib.Path(r"C:\Users\AlexisBragg\loaf-cache\set_1\images\cat") # change this to your local path

# define the image extensions
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# Count files recursively matching the extensions
loafed_image_count = sum(
    # rglob('*') recursively searches for all files in the directory and its subdirectories
    # check if the file has an image extension (case-insensitive)
    # if it does, count it

    1 for file in dataset_path_loafing.rglob('*')
    if file.suffix.lower() in image_extensions
)

not_loafed_image_count = sum(
    1 for file in dataset_path_not_loafing.rglob('*')
    if file.suffix.lower() in image_extensions
)

# print image count for testing of connection/path
print(f"Total images found in loaf dataset: {loafed_image_count}")
print(f"Total images found in not-loaf dataset: {not_loafed_image_count}")