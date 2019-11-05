# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:36:46 2018

@author: Henri_2

This script trains a recursive neural network with midi data
"""
from __future__ import print_function
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy
import os
import glob
import pickle


def main():
    """ Train a Neural Network to generate music """
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
    
    #notes_to_int = dict((note, number) for number, 
    #                    note in enumerate(pitches))
    #durations_to_int = dict((duration, number) for number,
    #                   duration in enumerate(durations))
    
    tubles_to_int = dict((tuble, number) for number,
                         tuble in enumerate(note_tubles))
        
    #All unique notenames
    notenames = sorted(set(item for item in note_tubles))
    
        # get amount of unique notes
    n_unique = len(set(note_tubles))
    
    network_input, network_output = make_sequence(note_tubles, n_unique, notenames, 
                                                  tubles_to_int)
    model = create_network(network_input, len(pitches))
    train(model, network_input, network_output)
    

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
    
    #net_input = numpy.reshape(net_input, (patterns, sequence_length, 1))

    #input_array = numpy.array(net_input)
    input_array = numpy.asarray(net_input)
    input_array = input_array.reshape(patterns, sequence_length, 1)

    input_array = input_array/float(n_unique)
    
    output = np_utils.to_categorical(output)
    
    return (input_array, output)

""" This function gets all the notes and chords from the midi files in the 
    ./Data directory and creates a list of each note in the midi songs"""
def get_notes():
    print('Loading midi-files from \Data')
    notes = []
             
    os.chdir("./Data")        
    for file in glob.glob("*.mid"):
        midi = converter.parse(file)    
        print("Parsing %s" % file)    
        notes_to_parse = None
    
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has only notes from a single instrument
            notes_to_parse = midi.flat.notes
    
        #chords are appended as individual notes separated with dots
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                notes.append(str(element.duration.type))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                notes.append(str(element.duration.type))
    
    with open('notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

"""This function creates the network"""
def create_network(network_input, n_vocab):
    print('creating network')
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))#LSTM layer takes the input and returns a sequence
    model.add(Dropout(0.3))#Changes 0.3 of the input to 0 to avoid overfitting
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))#Determines what function is used to calculate the weights
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    #The training can be continude by loading existing weights to the network
    #model.load_weights('weights.hdf5')

    return model

"""This function does the actual training of the network"""
def train(model, network_input, network_output):
    print('Training network')
    filepath = "After_53_epochs_weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(#by using checkpoints the weigths are saved after each epoch
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=60, batch_size=64, callbacks=callbacks_list)
    
if __name__ == '__main__':
    main()