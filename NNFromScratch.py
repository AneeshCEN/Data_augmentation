#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:22:15 2017

@author: aneesh
"""

import os
from scipy import misc

if __name__ == "__main__" :
    file_path = '/home/aneesh/Data_augmentation/train'
    images = []
    y = []
    for path, subdirs, files in os.walk(file_path):
        for name in files:
            if name.endswith('.png'):
                img_path = os.path.join(path,name)
                correct_cat = name
                images.append(misc.imread(img_path))
                y.append(correct_cat)