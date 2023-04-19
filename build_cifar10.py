import os
import cv2
#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
import json
import tensorflow_datasets as tfds
from tqdm import tqdm

def write_json(label_dict, json_path):
    json_object = json.dumps(label_dict, indent=4)
    with open(json_path, 'w') as f:
        f.write(json_object)

label_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"}

base_dir = "data"
dataset_name = "cifar10"

train_class = {}
test_class = {}

dataset_dir = os.path.join(base_dir, dataset_name)
os.makedirs(dataset_dir, exist_ok=True)
write_json(label_map, os.path.join(dataset_dir,"label2name.json"))
ds = tfds.load(dataset_name)
for traintest in list(ds.keys()):
    cur_ds = ds[traintest]
    # train/test folder dir
    traintest_dir = os.path.join(dataset_dir, traintest)
    os.makedirs(traintest_dir, exist_ok=True)
    if traintest == "train":
        class_dict = train_class
    elif traintest == "test":
        class_dict = test_class
    else:
        raise ValueError("unknown traintest" + traintest)
    for data in tqdm(cur_ds):
        image_id = data['id'].numpy().decode()
        image_label = data['label'].numpy()
        label_name = label_map[image_label]
        image_rgb = data['image'].numpy()
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        # save_dir
        save_dir = os.path.join(traintest_dir, label_name)
        # class stat
        if label_name in class_dict.keys():
            class_dict[label_name]+=1
        else:
            class_dict[label_name] = 1
            os.makedirs(save_dir, exist_ok=True)
        image_name = f"{image_id}_{image_label}.png"
        cv2.imwrite(os.path.join(save_dir, image_name), image_bgr)
    
    write_json(class_dict, os.path.join(base_dir, dataset_name, f"{traintest}_stat.json"))