# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:36:46 2018

@author: Henri_2

This script trains a recursive neural network with midi data
"""
from __future__ import print_function
import pickle

from make_sequence import make_sequence
from get_notes import get_notes
from create_network import create_network
from train import train

def main():
    #The notes can be extracted from midi files or loaded directly from as a list    
    #notes = get_notes()    
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    pitches = []
    durations = []
    #Getting aouont of individual notes and durations
    for i in range(0, len(notes)):
        if (i % 2 == 0): 
            pitches.append(notes[i])
        else: 
            durations.append(notes[i])
    
    #Makes the notes into tubles, where each tuble is like (pitch, duration)
    note_tubles = []    
    note_tubles = list(zip(durations, pitches))
    
    #All unique notenames
    notenames = sorted(set(item for item in note_tubles))
    
    #Making a dict for cahngin notes to integers
    tubles_to_int = dict((tuble, number) for number,
                         tuble in enumerate(notenames))
        
    
    # get amount of unique notes
    n_unique = len(set(note_tubles))
    
    #create the input and corresponding output for the network for training
    network_input, network_output = make_sequence(note_tubles, n_unique, 
                                                  notenames, tubles_to_int)
    #create the network in a way that fits the data
    model = create_network(network_input, n_unique)
    
    #train the model 
    train(model, network_input, network_output)
    
if __name__ == '__main__':
    main()