# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:04:47 2019

@author: Henri_2
"""

from __future__ import print_function
from keras.utils import np_utils
import numpy


""" This function cuts the notes list into "sequence_length" long sequences
    and their respective outputs: X notes from the list and the X+1:th note as 
    the output of the sequence. These sequences are used to train the network"""
def make_sequence(note_tubles, n_unique, notenames, tubles_to_int):       
    print('total length of training data:', len(note_tubles), ' notes')
    print('number of individual notes-duration combinations: ', n_unique)
           
    # cut the notes list into sequences of length sequence_length
    sequence_length = 50
    net_input = []
    output = []
    for i in range(0, len(note_tubles) - sequence_length, 1):
        sequence_in = note_tubles[i: i + sequence_length]
        sequence_out = note_tubles[i + sequence_length]
        for key in sequence_in:
            net_input.append(tubles_to_int[key])
        output.append(tubles_to_int[sequence_out])
                
    patterns = int(len(net_input)/(sequence_length))
    print(patterns)
    
    input_array = numpy.asarray(net_input)
    input_array = input_array.reshape(patterns, sequence_length, 1)

    #normalize the input
    input_array = input_array/float(n_unique)
    
    #convert the output to binary class matrix
    output = np_utils.to_categorical(output)
    
    return (input_array, output)