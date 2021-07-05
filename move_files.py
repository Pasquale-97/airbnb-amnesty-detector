import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import fileinput
import shutil



# step into dataset directory
os.chdir("Small_Dataset")
DIRS = os.listdir(os.getcwd())

# for all train, validation and test folders.
for DIR in DIRS:
    if os.path.isdir(DIR):
        os.chdir(DIR)
        print("Currently in subdirectory:", DIR)

        if(DIR == "test"):
            imagesTargetFolder = r'/Users/pasqualeiuliano/Google Drive/airbnb/new_small/test/images'
            labelsTargetFolder = r'/Users/pasqualeiuliano/Google Drive/airbnb/new_small/test/labels'
        if(DIR == "train"):
            imagesTargetFolder = r'/Users/pasqualeiuliano/Google Drive/airbnb/new_small/train/images'
            labelsTargetFolder = r'/Users/pasqualeiuliano/Google Drive/airbnb/new_small/train/labels'
        if(DIR == "validation"):
            imagesTargetFolder = r'/Users/pasqualeiuliano/Google Drive/airbnb/new_small/validation/images'
            labelsTargetFolder = r'/Users/pasqualeiuliano/Google Drive/airbnb/new_small/validation/labels'
        
        CLASS_DIRS = os.listdir(os.getcwd())
        # for all class folders step into directory to change annotations
        for CLASS_DIR in CLASS_DIRS:
            if os.path.isdir(CLASS_DIR) and CLASS_DIR:
                
                os.chdir(CLASS_DIR)
                print("Converting annotations for class: ", CLASS_DIR)

                for file in os.listdir():
                    if file.endswith('.jpg'):
                        shutil.copy2(file, imagesTargetFolder) #copies images to new folder
                    elif file.endswith('.txt'):
                        shutil.copy2(file, labelsTargetFolder) #copies labels to new folder
                    else:
                        continue
                
            os.chdir("..")
        os.chdir("..")