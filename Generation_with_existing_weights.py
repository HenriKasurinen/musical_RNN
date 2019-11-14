# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:28:19 2018

@author: Henri_2

This function generates notes by using a neural network, which uses existing
weights for each node of the network"""

import pickle

from make_generating_sequence import make_sequence
from create_network import create_network 
from generate_notes import generate_notes
from create_midi import create_midi

def main():
    #use same notes as in training
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
    
    tubles_to_int = dict((tuble, number) for number,
                         tuble in enumerate(notenames))       
  
        # get amount of unique notes
    n_unique = len(set(note_tubles))
    
    network_input, normalized_input = make_sequence(note_tubles, n_unique, notenames, 
                                                  tubles_to_int)
    model = create_network(normalized_input, n_unique)
    prediction_output = generate_notes(model, network_input, note_tubles, n_unique)
                                      
    create_midi(prediction_output)

if __name__ == '__main__':
    main()