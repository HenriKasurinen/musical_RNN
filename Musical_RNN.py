# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:36:46 2018

@author: Henri_2
"""
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy
import random
import sys
import os
import io
import music21
import glob
import pickle


def main():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))
    
    notenames = sorted(set(item for item in notes))
    
    network_input, network_output = make_sequence(notes, n_vocab, notenames)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)
    
    ##GENERATION OF MUSIC##
    # Get all pitch names

    network_input, normalized_input = make_sequence(notes, notenames, n_vocab)
    #model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, notenames, n_vocab)
    create_midi(prediction_output)

def make_sequence(notes, n_vocab, notenames):
    print('total length of training data:', len(notes), ' notes')
    print('number of individual notes: ', n_vocab)
    
    
    notes_to_int = dict((note, number) for number, note in enumerate(notenames))
    #int_to_notes = dict((number, note) for number, note in enumerate(notenames))
    
    # cut the text in semi-redundant sequences of maxlen characters
    sequence_length = 100
    net_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]
        net_input.append([notes_to_int[char] for char in sequence_in])
        output.append(notes_to_int[sequence_out])
        
    patterns = len(net_input)

    net_input = numpy.reshape(net_input, (patterns, sequence_length, 1))

    net_input = net_input/float(n_vocab)
    
    output = np_utils.to_categorical(output)
    
    return (net_input, output)


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    print('Loading midi-files from \Data')
    notes = []
    
    files = 'C:\\Users\\Henri_2\\Desktop\\Musical_RNN\\Data\\*.mid'
         
    for file in glob.glob(files):
        midi = converter.parse(file)
    
        print("Parsing %s" % file)
    
        notes_to_parse = None
    
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
    
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def create_network(network_input, n_vocab):
    print('creating network')
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    print('Training network')
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=2, batch_size=64, callbacks=callbacks_list)
    
    
def generate_notes(model, network_input, pitchnames, n_vocab):
    print('generating notes')
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    print('creating midi-file')
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    main()