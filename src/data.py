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

######## converting format ########

def folder2examples(folder, return_ModelData_object=True, sparse=True, beats_per_ex=16, sub_beats=4, use_base_key=False):
    """Turn folder of midi files into examples for piano autoencoder

    Arguments:

    Returns:

    Todo:
    Inputs can also be outputs
    may not want to store data with input property set... think about that.

    """
    
    examples = {key: [] for key in ['H', 'O', 'V', 'R', 'S', 'tempo', 'key']}
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
        examples['R'] = ml_classes.ModelData(examples['R'], 'R', transposable=True, activation='sigmoid')
        # could change pedal to three indicator variables instead of two
        examples['S'] = ml_classes.ModelData(examples['S'], 'S', transposable=False, activation='sigmoid')
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
    # add a column of zeros to the end of the training example, so that notes end sensibly
    R = np.concatenate((md['R'], np.zeros((1,md['R'].shape[-1]))))
    S = md['S']
    # to use note offs, we keep a record of notes for each pitch that are still sounding
    notes_sounding = [[] for _ in range(88)]
    # reverse transform tempo. If handling of tempo when generating examples is changed, then this will need to change
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
            h = sub_beat_times[timestep]
            note_on = h + O[timestep, pitch] * max_offset
            # calculating note off: add h to the time until the next 0 in the piano roll
            note_off = h + np.where(R[timestep:, pitch] == 0)[0][0] * sub_beat_length
            noteM = pretty_midi.Note(velocity=int(V[timestep, pitch] * 127), pitch=pitch+21, start=note_on, end=note_off)
            pm.instruments[0].notes.append(noteM)
        # sort pedal
        if S[timestep, 0] == 1:
            pm.instruments[0].control_changes.append(pretty_midi.ControlChange(64, 0, sub_beat_times[timestep]))
        if S[timestep, 1] == 1:
            pm.instruments[0].control_changes.append(pretty_midi.ControlChange(64, 127, sub_beat_times[timestep]))
        if S[timestep, 1] == 1 and S[timestep, 0] == 1:
            print('simultaneous pedal events!')

    return pm


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

######## ########

def int_transpose(np1, semitones):
    """Transpose pitches represented as integers (stored as np array), keeping pitches within piano range"""
    for idx,value in np.ndenumerate(np1):
        np1[idx] = min(max(value + semitones, 0), 87)
    # # slightly slower, and not in place
    # int_transpose = np.vectorize(lambda x: min(max(x + semitones, 0), 87))
    # b = np.array([[1,2,3],[3,4,5]])
    # np1 = int_transpose(np1)
    # return np1


def transpose_by_slice(np1, semitones):
    """Transpose by slicing off the top semitones rows of an np array, and stick them on the bottom (for pitches represented as indicator variables)"""
    np1 = np.concatenate((np1[...,-semitones:], np1[...,:-semitones]), axis=-1)
    return np1

def nb_data2chroma(examples, mode='normal'):
    chroma = np.empty((examples.shape[0], examples.shape[1], 12))
    for i, e in enumerate(examples):
        if i % 100 == 0:
            print(f'processing example {i} of {len(chroma)}')
        chroma[i,:,:] = nb2chroma(e, mode=mode)

    return(chroma)


######## pickling ########

def dump_pickle_data(item, filename):
    with open(filename, 'wb') as f:
        pickle.dump(item, f, protocol=2)

def get_pickle_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)



