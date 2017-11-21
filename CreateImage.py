#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:14:12 2017

@author: aneesh
"""

import os

img_size_cropped, img_size_cropped = 148, 148

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


num_channels = 3

def image_transformation(input_image, folder_name, file_name):
    datagen = ImageDataGenerator(
                                 rotation_range=180,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest'
                                 )



# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
    i = 0
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for batch in datagen.flow(input_image, batch_size=1,
                              save_to_dir=folder_name, save_prefix=file_name, save_format='png'):
        i += 1
        if i > 1000:
            break  # otherwise the generator would loop indefinitely
    
    
if __name__ == "__main__":
    batch_size = 16
    file_path = '/home/aneesh/Data_augmentation/'
    datagen = ImageDataGenerator(
                                 rotation_range=180,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest'
                                 )
#    train_generator = datagen.flow_from_directory(
#    '/home/aneesh/Data_augmentation/train',  # this is the target directory
#        target_size=(150, 150),  # all images will be resized to 150x150
#        batch_size=batch_size,
#        class_mode='binary')
    for files in os.listdir(file_path):
        if files.endswith('.png'):
            img = load_img(file_path+files)  # this is a PIL image
            print img.size
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
            image_transformation(x, file_path+files.strip('.png'), files.strip('.png'))
    
    