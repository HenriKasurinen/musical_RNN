# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:28:19 2018

@author: Henri_2

This function generates notes by using a neural network, which uses existing
weights for each node of the network"""

import numpy
import pickle
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from music21 import converter, instrument, note, chord, stream
from keras.utils import np_utils


def main():

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

"""The network has to be the same as in the training
    for the generation to work"""
def create_network(network_input, n_unique):
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
    model.add(Dense(n_unique))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    model.load_weights('weights.hdf5')

    return model
    
"""This function generates notes into an array"""    
def generate_notes(model, network_input, note_tubles, n_vocab):
    print('generating notes')
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_tubles = dict((number, tuble) for number, 
                         tuble in enumerate(note_tubles))

    pattern = network_input[start: start + 50]
    prediction_output = []
    
    previous_notes = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        
        index = numpy.argmax(prediction)

        #to avoid repeating a single note over and over again
        #the output is monitored and if there is not enough variance 
        #(7 individual notes whithin 20 notes)
        #a random note is sampled as the next note using the sample function
        previous_notes.append(index)
        if len(previous_notes) > 20:
            previous_notes.pop(0)
            
        if len(set(previous_notes)) < 7:
            index = sample(prediction[0], 1)
            
        result = int_to_tubles[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)

"""This function transforms the output array into a midi file"""
def create_midi(prediction_output):
    print('creating midi-file')
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        try:
            # pattern is a chord
            if ('.' in pattern[1]) or pattern[1].isdigit():
                
                notes_in_chord = pattern[1].split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.duration.type = pattern[0]
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern[1])
                new_note.offset = offset
                new_note.duration.type = pattern[0]
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
        except KeyError:    
            continue
        except:
            continue

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    

    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    main()