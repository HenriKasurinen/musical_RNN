# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:33:26 2019

@author: Henri_2
"""
import numpy


"""This function is basically the same as in the Musical_RNN and
    it cuts the notes list into sequences of inputs and their outputs"""
def make_sequence(note_tubles, n_unique, notenames, tubles_to_int):
    print('total length of training data:', len(note_tubles), ' notes')
    print('number of individual notes-duration combinations: ', n_unique)
           
    # cut the notes list into sequences of length sequence_length
    sequence_length = 50
    net_input = []
    for i in range(0, len(note_tubles) - sequence_length, 1):
        sequence_in = note_tubles[i: i + sequence_length]
        for key in sequence_in:
            net_input.append(tubles_to_int[key])
                
    patterns = int(len(net_input)/(sequence_length))
    print(patterns)
    
    input_array = numpy.asarray(net_input)
    input_array = input_array.reshape(patterns, sequence_length, 1)

    input_array = input_array/float(n_unique)
        
    return (net_input, input_array)