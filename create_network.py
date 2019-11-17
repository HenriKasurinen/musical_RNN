# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:09:06 2019

@author: Henri_2
"""

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation


def create_network(network_input, n_unique):
    print('creating network')
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        1024,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))#LSTM layer takes the input and returns a sequence
    model.add(Dropout(0.3))#Changes 0.3 of the input to 0 to avoid overfitting
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(1024))
    model.add(Dense(512))
    model.add(Dropout(0.3))
    model.add(Dense(n_unique))
    model.add(Activation('softmax'))#Determines what function is used to calculate the weights
    model.compile(loss='categorical_crossentropy', optimizer='adam')#used to be 'rsmprop'
    
    #The training can be continude by loading existing weights to the network
    #model.load_weights('weights.hdf5')

    return model