# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:06:59 2019

@author: Henri_2
"""
from __future__ import print_function
from music21 import converter, instrument, note, chord
import os
import glob
import pickle

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