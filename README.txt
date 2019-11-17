Musical_RNN.py takes a set of midi files and trains a recurrent neural network with the data. The weights for each node of the network
are saved after each epoch of training and can be used to create new music using the file Generation_with_existing_weights.py.

All the songs in the /Data directory are transposed to the same key (C major or A minor) in order to teach the netwrok only notes
which fit together. New midifiles can be transposed to the same key with the file transpose_midis_to_same_key.py

WHEN TRAINING:
Comment out line 36 from file "create_network.py" (model.load_weights('weights.hdf5')) if you do not have existing weights for the network.
WHEN PREDICTING:
Make sure that line 36 in file "create_network.py" is not commented out, since it loads the weights created in training.