import zipfile
import glob
from collections import OrderedDict
import os
import json

# create train.zip, train_map.txt, val.zip, val_map.txt
OUTPUT_PATH = "../RESISC45/zipped_archives/"

# Number of samples per classes
NUM_SAMPLES = 700

# 80%: TRAIN, 20%: VAL
TRAIN_VAL_SPLIT = 0.8

SAMPLES_TRAIN = TRAIN_VAL_SPLIT * NUM_SAMPLES

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

# Creating output zipped and map text file objects
train_zip_obj = zipfile.ZipFile(OUTPUT_PATH + 'train.zip', 'w')
val_zip_obj = zipfile.ZipFile(OUTPUT_PATH + 'val.zip', 'w')

train_map_path = OUTPUT_PATH + "train_map.txt"
val_map_path = OUTPUT_PATH + "val_map.txt"

train_map_obj = open(train_map_path, "w")
val_map_obj = open(val_map_path, "w")

# Walk through all images and perform train, val split
img_files = glob.glob('./raw/*.jpg')

#  Building an OrderedSet from OrderedDict, with keys = None
RESISC45_CLASSES = OrderedDict()

SAMPLE_COUNT = 0
CURRENT_CLASS = ""
CURRENT_CLASS_ID = -1

for img_file in img_files:
    # image filepath format: <LABEL>_<ITERATIVE_ID>.jpg
    rel_path = img_file.replace("./raw/","")
    label = rel_path.split(".")[0][:-4]
    
    # if unqiue label doesn't exist, add to the list
    if not label in RESISC45_CLASSES.keys():
        # print("Registering " + label + " in RESISC45_CLASSES\n")
        RESISC45_CLASSES[label] = None
        CURRENT_CLASS = label
        SAMPLE_COUNT = 0
        CURRENT_CLASS_ID = CURRENT_CLASS_ID + 1
    
    if SAMPLE_COUNT < SAMPLES_TRAIN:
        # print("train_map: " + img_file + " " + str(CURRENT_CLASS_ID) + "\n")
        # append to train.zip
        train_zip_obj.write(img_file, rel_path)
        
        # append to train_map.txt
        train_map_obj.write(rel_path + " " + str(CURRENT_CLASS_ID) + "\n")
        
    else:
        # print("val_map: " + img_file + " " + str(CURRENT_CLASS_ID) + "\n")
        # append to val.zip
        val_zip_obj.write(img_file, rel_path)
        
        # append to val_map.txt
        val_map_obj.write(rel_path + " " + str(CURRENT_CLASS_ID) + "\n")

    SAMPLE_COUNT = SAMPLE_COUNT + 1
    RESISC45_CLASSES[label] = SAMPLE_COUNT

with open(OUTPUT_PATH + 'RESISC45_CLASSES.json', 'w') as class_file:
    class_file.write(json.dumps(RESISC45_CLASSES))
class_file.close()

# Closing all files
val_map_obj.close()
train_map_obj.close()

val_zip_obj.close()
train_zip_obj.close()