# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:40:14 2019

@author: Henri_2
"""
import numpy

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)


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