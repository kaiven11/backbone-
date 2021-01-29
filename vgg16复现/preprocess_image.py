#coding=utf-8

import shutil
import glob
import os
import numpy as np
base_path = "/Users/huyonglei/Downloads/dogs-vs-cats/train/train"

# for img in glob.glob(os.path.join(base_path,"*.jpg")):
#     print(img)
#     basename = os.path.basename(img)
#     print(basename)
#     if  basename.startswith('cat'):
       
        
#         shutil.copy(os.path.join(base_path,img),"./data/traindata/cat")
#     else:
#         shutil.copy(os.path.join(base_path,img),"./data/traindata/dog")


img_list = []

for i in glob.glob("./data/traindata/cat/*.jpg"):
    img_list.append(i)

for j in glob.glob('./data/traindata/dog/*.jpg'):
    img_list.append(j)

# print(img_list)
import random

def split(full_list,shuffle=False,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2


if __name__ == "__main__":
    
    sublist_1,sublist_2 = split(img_list,shuffle=True,ratio=0.2)

    print (len(sublist_1))
    for i in sublist_1:
        basename = os.path.basename(i)
        print(basename)
        if basename.startswith("cat"):
            shutil.move(i,"./data/valdata/cat/")
        else:
            shutil.move(i,"./data/valdata/dog/")






















