#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:46:41 2020

@author: user
"""

import pandas as pd
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import numpy as np
import csv
import glob

all_label = []
with open('./cs-t0828-2020-hw1/training_labels.csv') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        all_label.append(row[1])
final_label= sorted(list(set(all_label)))
type_label=[]
for j in range(len(final_label)):
    type_label.append(final_label[j].split(" ")[-2])
type_label= sorted(list(set(type_label)))
final_label_dic = {}
type_label_dic = {}
make_label=[]
for f in range(len(final_label)):
    make_label.append(final_label[f].split(" ")[0])
make_label= sorted(list(set(make_label)))
make_label_dic = {}
for i in range(len(make_label)):
    make_label_dic.update({make_label[i]:i})
for i in range(len(final_label)):
    final_label_dic.update({final_label[i]:i})
    # type_label_dic.update({final_label[i].split(' ')[-2]:i})
for k in range(len(type_label)):
    type_label_dic.update({type_label[k]:k})
    
def getData(mode):
    all_data = []
    with open('./cs-t0828-2020-hw1/training_labels.csv') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            all_data.append(row)
    train_data = all_data[0:7485]
    test_data = all_data[7485:]
        # glob_img = './cs-t0828-2020-hw1/training_data/training_data/*.jpg'
        # glob_train = []
        # for k in range(len(train_data)):
        #     glob_img.append('./cs-t0828-2020-hw1/training_data/training_data/'+str(train_data[k][0])+'.jpg')
        # paths_img = sorted(glob_img)
    if mode=='train':
        return train_data
    if mode=='eval':
        glob_eval = './cs-t0828-2020-hw1/testing_data/testing_data/*.jpg'
        paths_eval = sorted(glob.glob(glob_eval))
        return paths_eval
    else:
        return test_data

class CarclassifyLoader(data.Dataset):
    def __init__(self, mode, transforms = None):
        self.data = getData(mode)
        self.mode = mode
        if self.mode == "train":
            self.transforms =  T.Compose([
                T.Resize((400,400)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ToTensor(),
                T.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
                ])
        else:
            self.transforms =  T.Compose([
                T.Resize((400,400)),
                T.ToTensor(),
                T.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
                ])

        print("> Found %d images..." % (len(self.data)))
        
    def __len__(self):
        """'return the size of dataset"""
        return len(self.data)
    
    def __getitem__(self, index):
        # Step 1.
        if self.mode == 'eval':
            img_path = self.data[index]
            img_name = int(str((self.data[index].split('/')[-1]).split('.')[-2]))
            print(img_name)
            label_final=0
            label_make=0
            label_type=0
        else:
            img_path = './cs-t0828-2020-hw1/training_data/training_data/'+(self.data[index][0]).zfill(6)+'.jpg'
            img_name = (self.data[index][0]).zfill(6)
            # Step 2.
            label_final  = final_label_dic[self.data[index][1]]
            label_make = make_label_dic[self.data[index][1].split(' ')[0]]
            # print(label_make)
            label_type = type_label_dic[self.data[index][1].split(' ')[-2]]
            # print(label_type)

        # Step 3.
        img = Image.open(img_path).convert("RGB")
        # print(img.size)
        img = self.transforms(img)

        
        # Step 4.
        return img_name , img, label_final, label_make, label_type

# if __name__ == "__main__":
#     test = CarclassifyLoader('test')
#     img_name , image1,label1,lb2,lb3 = test[5]
#     for name , _ , _ , make , _ in test:
#         if make > 21:
#             print(name , make)
