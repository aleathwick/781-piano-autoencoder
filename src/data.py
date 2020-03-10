import pretty_midi
import numpy as np
import pandas as pd
import pickle
import src.midi_utils as midi_utils
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
data_path = 'F:\Google Drive\Study\Machine Learning\Datasets\MaestroV2.00\maestro-v2.0.0/'



def get_processed_oore_data(data_path, filenames, skip = 1, print_cut_events=True, n = 10,
                            n_events=600, speed=1):
    """Reads in midi files, converts to oore, splits into training examples
    
    Parameters:
    ----------
    data_path: str
        path to the data directory, where all the midi files are

    skip : int
        take every nth file, controls number of files in between those taken

    print_cut_events : bool
        If true, then lists of numbers of discarded events will be printed, that didn't make the training examples, because they
        would have made the example too long, or they weren't long enough to form an example. 

    n : int
        sends every nth training example to the validation set

    n_events : int
        no. of events per training example

    Returns
    ----------
    X : list
        list of training examples

    Y : list
        Essentially the same as X, but displaced by a timestep, so it can act as the truth set at each step
    
    X_val : list
        list of validation examples

    Y_val : list
        Essentially the same as X_val, but displaced by a timestep, so it can act as the truth set at each step


    n_events: int
        number of events in a training example
    
    """
    #just want a selection at this stage
    X = []
    Y = []
    X_val = []
    Y_val = []
    val_counter = 1
    # We'll check how many events have to be discarded because they're longer than target sequence length
    leftover = []
    # And we'll check too how many sequences are too short
    too_short = []
    for i in range(0, len(filenames) - 1, skip):
        pm = pretty_midi.PrettyMIDI(data_path + filenames[i])
        sustain_only(pm)
        desus(pm)
        # keep a copy of the notes for later (well, technically a reference...)
        # but we'll reassign to notes with a slice from all_notes, so it will work
        if speed != 1:
            stretch(pm, speed)
        all_notes = pm.instruments[0].notes
        #n_notes is the number of notes to process at once
        # originally 80, now 200
        n_notes = 200
        #n_events is the number of events in a training example
        for i in range(n_notes, len(all_notes) - 1, n_notes):
            pm.instruments[0].notes = all_notes[i - n_notes:i]
            trim_silence(pm)
            events = midi2oore(pm)
            if len(events) >= n_events + 1:
                val_counter += 1
                if val_counter % n == 0:
                    X_val.append(events[0:n_events])
                    leftover.append(len(events) - (n_events+1))
                else:
                    X.append(events[0:n_events])
                    leftover.append(len(events) - (n_events+1))
            else:
                too_short.append(len(events))
    if print_cut_events:
        print('leftover: ', leftover)
        print('too_short: ', too_short)

    return (X, X_val)


def get_processed_oore2_data(data_path, filenames, skip = 1, print_cut_events=True, n = 10,
                            n_events=600, speed=1):
    """Second version. Reads in midi files, converts to oore, splits into training examples
    
    Parameters:
    ----------
    data_path: str
        path to the data directory, where all the midi files are

    skip : int
        take every nth file, controls number of files in between those taken

    print_cut_events : bool
        If true, then lists of numbers of discarded events will be printed, that didn't make the training examples, because they
        would have made the example too long, or they weren't long enough to form an example. 

    n : int
        sends every nth training example to the validation set

    n_events : int
        no. of events per training example. 

    Returns
    ----------
    X : list
        list of training examples. each of length n_events + 1, so it can be broken into X, and y displaced by one timestep

    Notes
    --------
    Normally there are left over events from any given group of notes selected. Because of this, overlapping groups of notes are selected.

    """
    #just want a selection at this stage
    X = []
    # We'll check how many events have to be discarded because they're longer than target sequence length
    leftover = []
    # And we'll check too how many sequences are too short
    too_short = []
    # get lengths of each example
    lengths = []
    for i in range(0, len(filenames), skip):
        print(f'file {i} of {len(filenames)}')
        pm = pretty_midi.PrettyMIDI(data_path + filenames[i])
        sustain_only(pm)
        desus(pm)
        # keep a copy of the notes for later (well, technically a reference...)
        # but we'll reassign to notes with a slice from all_notes, so it will work
        if speed != 1:
            stretch(pm, speed)
        all_notes = pm.instruments[0].notes
        #n_notes is the number of notes to process at once
        # originally 80, now 180
        n_notes = int(n_events/2.72)
        #n_events is the number of events in a training example
        for i in range(n_notes, len(all_notes), int(n_notes//2)):
            pm.instruments[0].notes = all_notes[i - n_notes:i]
            trim_silence(pm)
            # get length, for calculating stats
            lengths.append(pm.instruments[0].notes[-1].end - pm.instruments[0].notes[0].start)
            events = pm2oore2(pm)
            if len(events) >= n_events + 1:
                X.append(events[0:n_events + 1])
                leftover.append(len(events) - (n_events+1))
                print(f'Success: {len(events) - (n_events+1)} leftover')
            else:
                too_short.append(len(events))
                print(f'Failure: {len(events)} long, too short')
    if print_cut_events:
        print(f'{len(leftover)} examples with leftovers, average lenth {np.mean(leftover):.2f}')
        print(f'{len(too_short)} examples too short: {np.mean(too_short):.2f}')
        print(f'average example length: {np.mean(lengths):.2f}')

    return X


def stretch(pm, speed):
    '''stretches a pm midi file'''
    for note in pm.instruments[0].notes:
        note.start = note.start * speed
        note.end = note.end * speed

def files2note_bin_examples(data_path, filenames, skip = 1, print_cut_events=True, n_notes=220, speed=1):
    """Reads in midi files, converts to oore, splits into training examples
    
    Arguments:
    data_path -- str, path to the data directory, where all the midi files are
    filenames -- list of filenames to read in
    skip -- int, 'take every nth file'
    starting_note -- int, 'start reading files at nth note'
    print_cut_events -- bool: If true, then lists of numbers of discarded events will be printed, that didn't make the training examples, because they
        would have made the example too long, or they weren't long enough to form an example. 
    n_events -- int, no. of events per training example

    Returns:
    X -- list of training examples, each of length n_notes + 1. X[:-1] for input, X[1:] for output.
    
    """
    n_velocity=16
    n_files = len(filenames)
    file_n = 1
    #just want a selection at this stage
    X = []
    exceeded = []
    max_shift = 9 # 10 total ticks... one of them is ZERO!
    max_duration = 12
    shifts_exceeded = 0
    durations_exceeded = 0
    notes_lost = 0
    # We'll check how many events have to be discarded because they're longer than target sequence length
    leftover = []
    # And we'll check too how many sequences are too short
    too_short = []
    # iterate over the files, taking every skipth file
    for i in range(0, len(filenames), skip):
        pm = pretty_midi.PrettyMIDI(data_path + filenames[i])
        sustain_only(pm)
        desus(pm)
        if speed != 1:
            stretch(pm, speed)
        note_bin = pm2note_bin(pm, M_shift_ms = 600, m_shift_ms = 25,  M_duration_ms = 800, m_duration_ms = 50, n_velocity=16)

        # iterate over all the notes, in leaps of n_notes
        print('######## Example no.', file_n, 'of', n_files, ', length ' + str(len(note_bin)))
        file_n += skip
        for i in range(0, len(note_bin), int(n_notes//2)):
            # check there are enough notes left for a training example
            if len(note_bin[i:]) >= n_notes + 1:
                # example, initially, has one extra note, so it can be X and Y
                example = note_bin[i:(i+n_notes+1)]
                # check that there are no notes that are too long, or shifted too much
                if max([note[1] for note in example]) <= max_shift:
                    if max([note[3] for note in example]) <= max_duration:
                        X.append(example)
                    else:
                        # print('exceeded: ' + str(max([note[3] for note in example])))
                        durations_exceeded += 1
                        notes_lost += n_notes + 1
                        exceeded.append(example)

                else:
                    # print('exceeded: ' + str(max([note[1] for note in example])))
                    shifts_exceeded +=1
                    notes_lost += n_notes + 1
                    exceeded.append(example)
            else:
                notes_lost += len(note_bin[i:])
        if print_cut_events:
            print('notes: ', (n_notes + 1) * len(X))
            print('total_durations_exceeded: ', durations_exceeded)
            print('total_shifts_exceeded: ', shifts_exceeded)
            print('notes_lost: ', notes_lost)
    
    return X

def nb_data2chroma(examples, mode='normal'):
    chroma = np.empty((examples.shape[0], examples.shape[1], 12))
    for i, e in enumerate(examples):
        if i % 100 == 0:
            print(f'processing example {i} of {len(chroma)}')
        chroma[i,:,:] = nb2chroma(e, mode=mode)
    return(chroma)


def oore_data2chroma(examples, mode='normal'):
    chroma = np.empty((examples.shape[0], examples.shape[1], 12))
    for i, e in enumerate(examples):
        if i % 100 == 0:
            print(f'processing example {i} of {len(chroma)}')
        chroma[i,:,:] = oore2chroma(e, mode=mode)
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

