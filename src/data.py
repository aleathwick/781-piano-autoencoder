import pretty_midi
import numpy as np
import pandas as pd
import os
import time
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

def normalize_tempo(tempo, inverse=False):
    """tempo needs to be normalized in several places, so this fn is for consistency
    
    Arguments:
    tempo -- int or float to be normalized and centered (or np array of values)
    inverse -- if true, perform inverse operation
    """
    if not inverse:
        return tempo / 100 - 1
    else:
        return (tempo + 1) * 100


def folder2examples(folder, return_ModelData_object=True, sparse=True, beats_per_ex=16, sub_beats=4, use_base_key=False):
    """Turn folder of midi files into examples for piano autoencoder

    Arguments:
    folder -- folder of midi files
    return_ModelData_object -- choose to return ModelData objects, or arrays
    sparse -- whether or not data is stored sparsely (scipy csc arrays)
    beats_per_ex -- in whatever measure of beats the midi files provide
    sub_beats -- smallest note value, as no. of notes per beat - fineness of grid to use when factoring midi files
    use_base_key -- if true, transpose all examples to C/Am

    Returns:
    examples -- dictionary of ModelData objects or arrays

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

    # check out how much training data there is
    mean_bpm = np.mean(normalize_tempo(np.array(examples['tempo']), inverse=True))
    seconds = 60 / mean_bpm * beats_per_ex * len(examples['H'])
    time.strftime('%Hh %Mm %Ss', time.gmtime(seconds))
    print(time.strftime('%Hh %Mm %Ss', time.gmtime(seconds)), 'of training data')
    
    if return_ModelData_object:
        examples['H'] = ml_classes.ModelData(examples['H'], 'H', transposable=True, activation='sigmoid', seq=True)
        examples['O'] = ml_classes.ModelData(examples['O'], 'O', transposable=True, activation='tanh', seq=True)
        examples['V'] = ml_classes.ModelData(examples['V'], 'V', transposable=True, activation='sigmoid', seq=True)
        examples['R'] = ml_classes.ModelData(examples['R'], 'R', transposable=True, activation='sigmoid', seq=True)
        # could change pedal to three indicator variables instead of two
        examples['S'] = ml_classes.ModelData(examples['S'], 'S', transposable=False, activation='sigmoid', seq=True)
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

    # invert transform tempo. If handling of tempo when generating examples is changed, then this will need to change
    tempo = normalize_tempo(md['tempo'], inverse=True)
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



