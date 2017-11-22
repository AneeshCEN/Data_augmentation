#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:22:15 2017

@author: aneesh
"""

import os
import cv2
from scipy import misc
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras import backend as K
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt

if __name__ == "__main__" :
    file_path = '/home/aneesh/Data_augmentation/train'
    num_classes = 3
    batch_size = 128
    epochs = 12
    img_rows, img_cols = 28, 28
    images = []
    y = []
    for path, subdirs, files in os.walk(file_path):
        for name in files:
            if name.endswith('.png'):
                img_path = os.path.join(path,name)
                correct_cat = name
                images.append(cv2.resize(misc.imread(img_path),(28,28)))
                y.append(os.path.basename(os.path.normpath(path)))
    train_data = np.array(images)
    labels = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels, 
                                                        test_size=0.5, 
                                                        random_state=4
                                                        )
    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 3)
        
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    
    model = Sequential()
#convolutional layer with rectified linear unit activation
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    #again
    model.add(Conv2D(64, (3, 3), activation='relu'))
    #choose the best features via pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #randomly turn neurons on and off to improve convergence
    model.add(Dropout(0.25))
    #flatten since too many dimensions, we only want a classification output
    model.add(Flatten())
    #fully connected to get all relevant data
    model.add(Dense(128, activation='relu'))
    #one more dropout for convergence' sake :) 
    model.add(Dropout(0.5))
    #output a softmax to squash the matrix into output probabilities
    model.add(Dense(num_classes, activation='softmax'))
    #Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
    #categorical ce since we have multiple classes (10) 
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    #train that ish!
    hist = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test))
    
    
    training_acc = hist.history['acc']
    validation_acc = hist.history['val_acc']

    plt.figure(1,figsize=(7,5))
    plt.plot(range(epochs),training_acc)
    plt.plot(range(epochs),validation_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    
    
    
    training_loss = hist.history['loss']
    validation_loss = hist.history['val_loss']

    plt.figure(2,figsize=(7,5))
    plt.plot(range(epochs),training_loss)
    plt.plot(range(epochs),validation_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    



