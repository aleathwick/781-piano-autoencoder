import pretty_midi
import numpy as np
import pandas as pd
import os
import pickle
import src.midi_utils as midi_utils
import src.ml_classes as ml_classes
from collections import namedtuple
from scipy.sparse import csc_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


def stretch(pm, speed):
    '''stretches a pm midi file'''
    for note in pm.instruments[0].notes:
        note.start = note.start * speed
        note.end = note.end * speed


def folder2examples(folder, return_ModelData_object=True, sparse=True, beats_per_ex=16, sub_beats=4, use_base_key=False):
    """Turn folder of midi files into examples for piano autoencoder

    Arguments:

    Returns:

    Todo:
    Inputs can also be outputs
    may not want to store data with input property set... think about that.

    """
    
    examples = {key: [] for key in ['H', 'O', 'V', 'tempo', 'key']}
    example_length = 64
    piano_range = 88
    files = [file for file in os.scandir(folder)]
    for i, file in enumerate(files):
        if i % 10 == 0:
            print(f'processing file {i} of {len(files)}')
        pm = pretty_midi.PrettyMIDI(file.path)
        # get the key from the filename, assuming it is the last thing before the extension
        key = file.path.split('_')[-1].split('.')[0]
        file_examples = midi_utils.pm2example(pm, key, sparse=sparse, beats_per_ex=beats_per_ex, sub_beats=sub_beats, use_base_key=use_base_key)
        for key, data in file_examples.items():
            examples[key].extend(data)
    
    if return_ModelData_object:
        examples['H'] = ml_classes.ModelData(examples['H'], 'H', transposable=True, activation='sigmoid')
        examples['O'] = ml_classes.ModelData(examples['O'], 'O', transposable=True, activation='tanh')
        examples['V'] = ml_classes.ModelData(examples['V'], 'V', transposable=True, activation='sigmoid')
        examples['key'] = ml_classes.ModelData(examples['key'], 'key', transposable=True)
        examples['tempo'] = ml_classes.ModelData(examples['tempo'], 'tempo', transposable=False)
    return examples


def HOV2pm(md, sub_beats=4):
    """go from HOV and tempo to pretty midi
    
    Arguments:
    md -- dictionary containing data for HOV and tempo
    sub_beats - number of sub beats used for quantizing
    
    """
    
    H = md['H']
    O = md['O']
    V = md['V']
    tempo = (md['tempo']  + 1) * 100
    beat_length = 60 / tempo[0]
    sub_beat_length = beat_length / sub_beats
    max_offset = sub_beat_length / 2
    pm = pretty_midi.PrettyMIDI(resolution=960)
    pm.instruments.append(pretty_midi.Instrument(0, name='piano'))
    beats = [i * beat_length for i in range(len(H))]
    sub_beat_times = [i + j * sub_beat_length for i in beats for j in range(sub_beats)]
    for timestep in range(len(H)):
        for pitch in np.where(H[timestep] == 1)[0]:
            note_on = sub_beat_times[timestep] + O[timestep, pitch] * max_offset
            # for now, everything will just have to be staccato
            note_off = note_on + sub_beat_length
            noteM = pretty_midi.Note(velocity=int(V[timestep, pitch] * 127), pitch=pitch+21, start=note_on, end=note_off)
            pm.instruments[0].notes.append(noteM)
    return pm

# generate_midi

def examples2pm(md, sub_beats=4):
    """Turn a random training example into a pretty midi file
    
    Arguments:
    md -- a dictionary of model datas or np matrices. Shouldn't be in sparse format.
    
    """

    i = np.random.randint(0, len(md['H']))
    print(f'example {i} chosen')
    md = {md.name: md.data[i] for md in md.values()}
    for name, data in md.items():
        if isinstance(data, csc_matrix):
            md[name] = data.toarray()
    md = {name: data for name, data in md.items()}
    pm = HOV2pm(md)
    return pm


def nb_data2chroma(examples, mode='normal'):
    chroma = np.empty((examples.shape[0], examples.shape[1], 12))
    for i, e in enumerate(examples):
        if i % 100 == 0:
            print(f'processing example {i} of {len(chroma)}')
        chroma[i,:,:] = nb2chroma(e, mode=mode)
    return(chroma)


def dump_pickle_data(item, filename):
    with open(filename, 'wb') as f:
        pickle.dump(item, f, protocol=2)

def get_pickle_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def make_one_hot(examples, n_values=333):
    """ takes in a list of training examples, returns them in one hot format
    
    Inputs
    ----------
    examples : list
        should contain training examples, which themselves are lists of events expressed in integers
    
    """
    arr = np.empty((len(examples), len(examples[0]), n_values), dtype=object)
    for i in range(len(examples)):
        one_hots = to_categorical(examples[i], num_classes=n_values, dtype='float32')
        arr[i] = one_hots
    return arr

def get_max_pred(l):
    array = np.zeros((1, 1, 333))
    array[0][0] = to_categorical(np.argmax(l), num_classes=333)
    return tf.convert_to_tensor(array, dtype=tf.float32)

