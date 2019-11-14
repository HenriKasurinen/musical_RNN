# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:11:01 2019

@author: Henri_2
"""
from __future__ import print_function
from keras.callbacks import ModelCheckpoint

"""This module trains the network"""
def train(model, network_input, network_output):
    print('Training network')
    filepath = "After-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(#by using checkpoints the weigths are saved after each epoch
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min')
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=60, batch_size=64, callbacks=callbacks_list)