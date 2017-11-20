#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:14:12 2017

@author: aneesh
"""



img_size_cropped, img_size_cropped = 148, 148

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


num_channels = 3

def image_transformation(input_image):
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
    for batch in datagen.flow(input_image, batch_size=1,
                              save_to_dir='preview', save_prefix='AIRBAG_INDICATOR', save_format='png'):
        i += 1
        if i > 100:
            break  # otherwise the generator would loop indefinitely
    
    
if __name__ == "__main__":
    img = load_img('/home/aneesh/Data_augmentation/AIRBAG_INDICATOR.png')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    image_transformation(x)
    
    