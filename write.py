# !/usr/bin/env python
# -*- encoding: utf-8 -*-
 
import os
 
img_path1 = '/home/dgyu/public/workspace/airs/data/vaihingen/train/images_512Jpg'
img_path2 = '/home/dgyu/public/workspace/airs/data/vaihingen/train/masks_512'

img_list1=os.listdir(img_path1)
img_list2=os.listdir(img_path2)

img_list1.sort()
img_list2.sort()

# print('img_list1: ',img_list1)

# print('img_list2: ',img_list2)

 
with open('/home/dgyu/public/workspace/UniMatch-main/splits/vaihingen/train.txt','w') as f:
    n = len(img_list1)
    print(n)
    for i in range(n):
        image =img_list1[i]
        mask_rgb = img_list2[i]
        f.write('train/images_512Jpg/'+image+' train/masks_512/'+mask_rgb+'\n')
        
