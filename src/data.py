import pretty_midi
from tqdm import tqdm
import pymongo
import gridfs
import numpy as np
import pandas as pd
import os
import time
import pickle
import src.midi_utils as midi_utils
import src.ml_classes as ml_classes
import src.models as models
from collections import namedtuple
from scipy.sparse import csc_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from IPython.display import clear_output
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt


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

def normalize_velocity(velocity, inverse=False):
    if not inverse:
        return velocity ** 0.6
    if inverse:
        return velocity ** 1/0.6

def filepath2key(filepath):
    """takes a file path and returns key (relying on the filename convention I've used)"""
    return filepath.split('_')[-1].split('.')[0]

def folder2examples(folder, return_ModelData_object=True, sparse=True, beats_per_ex=16, sub_beats=4, use_base_key=False, nth_file=None, vel_cutoff=4, V_no_zeros=False):
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
    seconds -- total amount of data, in seconds

    """
    
    examples = {key: [] for key in ['H', 'O', 'V', 'R', 'S', 'tempo', 'key', 'V_mean']}
    example_length = 64
    piano_range = 88
    files = [file for file in os.scandir(folder) if not file.is_dir()]
    if nth_file != None:
        files = [f for i, f in enumerate(files) if i % nth_file == 0]
    for file in tqdm(files):
        pm = pretty_midi.PrettyMIDI(file.path)
        # get the key from the filename, assuming it is the last thing before the extension
        key = filepath2key(file.path)
        file_examples = midi_utils.pm2example(pm, key, sparse=sparse, beats_per_ex=beats_per_ex, sub_beats=sub_beats, use_base_key=use_base_key, vel_cutoff=vel_cutoff, V_no_zeros=V_no_zeros)
        if file_examples != None:
            for key, data in file_examples.items():
                examples[key].extend(data)

    # check out how much training data there is
    mean_bpm = np.mean(normalize_tempo(np.array(examples['tempo']), inverse=True))
    seconds = 60 / mean_bpm * beats_per_ex * len(examples['H'])
    time.strftime('%Hh %Mm %Ss', time.gmtime(seconds))
    print(time.strftime('%Hh %Mm %Ss', time.gmtime(seconds)), 'of data')
    
    if return_ModelData_object:
        examples['H'] = ml_classes.ModelData(examples['H'], 'H', transposable=True, seq=True)
        examples['O'] = ml_classes.ModelData(examples['O'], 'O', transposable=True, seq=True)
        examples['V'] = ml_classes.ModelData(examples['V'], 'V', transposable=True, seq=True)
        examples['R'] = ml_classes.ModelData(examples['R'], 'R', transposable=True, seq=True)
        # could change pedal to three indicator variables instead of two
        examples['S'] = ml_classes.ModelData(examples['S'], 'S', transposable=False, seq=True)
        examples['key'] = ml_classes.ModelData(examples['key'], 'key', transposable=True)
        examples['tempo'] = ml_classes.ModelData(examples['tempo'], 'tempo', transposable=False)
        examples['V_mean'] = ml_classes.ModelData(examples['V_mean'], 'V_mean', transposable=False)
    return examples, seconds

def folder2nbq(folder, return_ModelData_object=True,seq_length=50, sub_beats=2, example_bars_skip=4, use_base_key=False, nth_file=None, nth_example=None, vel_cutoff=4):
    """Turn folder of midi files into examples for piano autoencoder

    Arguments:
    folder -- folder of midi files
    return_ModelData_object -- choose to return ModelData objects, or arrays
    seq_length -- length of each example in sub beats
    sub_beats -- sub beats per beat
    example_bars_skip -- skip in bars to start of next example
    use_base_key -- if true, transpose all examples to C/Am
    nth_file -- use only every nth file
    vel_cutoff -- set velocity threshold for note inclusion

    Returns:
    examples -- dictionary of ModelData objects or arrays
    seconds -- total amount of data, in seconds

    """
    
    examples = {key: [] for key in ['TSn', 'TEn', 'TBn', 'TMn', 'TSBn', 'Pn', 'PSn', 'PCn', 'Vn', 'LRn', 'tempo', 'key', 'V_mean']}
    files = [file for file in os.scandir(folder) if not file.is_dir()]
    if nth_file != None:
        files = [f for i, f in enumerate(files) if i % nth_file == 0]
    for file in tqdm(files):
        pm = pretty_midi.PrettyMIDI(file.path)
        midi_utils.filter_notes(pm, vel_cutoff)
        if len(pm.instruments[0].notes) >= seq_length:
            # get the key from the filename, assuming it is the last thing before the extension
            key = filepath2key(file.path)
            file_examples = midi_utils.pm2nbq(pm, seq_length=seq_length, sub_beats=sub_beats, example_bars_skip=example_bars_skip, key=key, use_base_key=use_base_key, nth_example=nth_example)
            if file_examples != None:
                for key, data in file_examples.items():
                    examples[key].extend(data)

    # check out how much training data there is
    mean_bpm = np.mean(normalize_tempo(np.array(examples['tempo']), inverse=True))
    unique_beats_per_example = example_bars_skip * 4
    print(mean_bpm, unique_beats_per_example, len(examples['TSn']))
    seconds = 60 / mean_bpm * unique_beats_per_example * len(examples['TSn'])
    time.strftime('%Hh %Mm %Ss', time.gmtime(seconds))
    print(time.strftime('%Hh %Mm %Ss', time.gmtime(seconds)), 'of data')
    
    if return_ModelData_object:
        # info for each input/output as regards transposability and sequentiality can be retrieved from the get_model_reqs function
        model_inputs = models.get_model_reqs('all', 'all')
        for k in examples.keys():
            examples[k] = ml_classes.ModelData(examples[k], k, transposable=model_inputs[k].transposable, seq=model_inputs[k].seq)
    return examples, seconds


def HOV2pm(md, sub_beats=4):
    """go from HOV and tempo to pretty midi
    
    Arguments:
    md -- dictionary containing data for HOV and tempo... NOT a md object, as the name would imply
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


def nbq2pm(md, sub_beats=2):
    """given dicionary containing NBQ data for a single example, returns a pm object"""
    tempo = normalize_tempo(md['tempo'], inverse=True)
    beat_length = 60 / tempo[0]
    sub_beat_length = beat_length / sub_beats
    pm = pretty_midi.PrettyMIDI(resolution=960)
    pm.instruments.append(pretty_midi.Instrument(0, name='piano'))
    pitches = np.where(md['Pn'] == 1)[-1] + 21
    for i in range(len(md['TEn'])):
        velocity = int(md['Vn'][i] * 127)
        start = md['TSn'][i] * sub_beat_length
        end = md['TEn'][i] * sub_beat_length
        pitch = pitches[i]
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
        pm.instruments[0].notes.append(note)
    return pm
    

def examples2pm(md, i=None, sub_beats=4):
    """Turn a random training example into a pretty midi file
    
    Arguments:
    md -- a dictionary of model datas or np matrices. Shouldn't be in sparse format.
    
    """

    if i == None:
        i = np.random.randint(0, len(md['H']))
    print(f'example {i} chosen')
    md = {md.name: md.data[i] for md in md.values()}
    for name, data in md.items():
        if isinstance(data, csc_matrix):
            md[name] = data.toarray()
    pm = HOV2pm(md, sub_beats=sub_beats)
    return pm


def n_rand_examples(model_datas, n=10, idx=[0,45,70,100,125,150,155]):
    """gets n random examples from dictionary of model datas, prepared for prediction (duplicates some as ar inputs, adds dummy input)
    
    Arguments:
    model_datas -- dictionary of model data objects (see ml_classes.py)
    n -- number of random examples to select
    idx -- a list of indices, used if n is 0 or None 
    
    Returns:
    random_examples -- dictionary ready of data for predicting on
    idx -- indices of examples
    """

    random_examples = {}

    # select n random examples
    if n == 'all':
        idx = list(range(len(list(model_datas.values())[0])))
    elif n !=0 and n != None:
        idx = np.random.randint(0, len(list(model_datas.values())[0]), n)
    
    # add dummy variable (for conductor LSTM, if it exists)
    random_examples['dummy']= np.zeros((len(idx),0))

    # extract data for each training example from the model datas
    for md in model_datas.values():
        data = md.data[idx,...]
        # need to check data isn't a sparse matrix
        if isinstance(data[0], csc_matrix):
            data = np.array([d.toarray() for d in data])
        random_examples[md.name + '_in'] = data
        # if it is sequential data, also add it as an ar input (just in case)
        # TSn and TEn lack the extra dimension, but they're never used for prediciton anyway
        if md.seq and md.name not in ['TSn', 'TEn']:
            random_examples[md.name + '_ar'] = np.concatenate([np.zeros((len(idx),1, md.dim)), data[:,:-1]], axis=-2)
    for k, v in random_examples.items():
        print(k, v.shape)
    return random_examples, idx

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


######## convenient ########

def sizeof_fmt(num, suffix='B'):
    """by Fred Cirera"""
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)




######## Plotting sacred runs ##########
def plot_sacred_runs(id_list, x, split=None, parameter_is_list=False, index_of_interest=-1, return_df=False, plot_params={}):
    """retrieves and plots runs from sacred mongodb
    
    Arguments:
    id_list -- list of run ids
    x -- hyperparameter of interest
    split -- config hyperparameter to split plotting by (multiple lines on same plot)
    parameter_is_list -- specific case where a hyperparameter may be a list
    index_of_interest -- if hyperapameter of interest is a list, which index of list do we care about?
    
    """
    # establish connection to database
    client = pymongo.MongoClient()
    fs = gridfs.GridFS(client.sacred)
    runs = client.sacred.runs
    metrics = client.sacred.metrics

    # determine which runs are needed
    run_entries = list(runs.find({'_id': {'$in': id_list}}))
    # metric_ids = {m['name']: ObjectId(m['id']) for m in run_entry['info']['metrics']}
    df = pd.DataFrame()

    # What is the hyperparameter of interst? (x axis)
    df[x] = [run['config'][x] for run in run_entries]
    # change index to reflect run id
    df.index = [run['_id'] for run in run_entries]

    if parameter_is_list:
        df[x] = df[x].apply(lambda x: x[index_of_interest])

    # stats where minimum is best, or maximum is best
    min_stats = [k for k in run_entries[0]['info']['logs'].keys() if any(x in k for x in ['categ', 'loss', 'mse'])]
    max_stats = [k for k in run_entries[0]['info']['logs'].keys() if any(x in k for x in ['acc'])]

    for stat in min_stats:
    #     df[stat] = [min([step['value'] for step in run['info']['logs'][stat]]) for run in run_entries]
        # for some inexplicable reason, sometimes the entry is a list of values, rather than a list of dictionaries with dtype recorded
        df[stat] = [min([step['value'] for step in run['info']['logs'][stat]]) if isinstance(run['info']['logs'][stat][-1], dict) else min(run['info']['logs'][stat]) for run in run_entries]
    for stat in max_stats:
        df[stat] =[max([step['value'] for step in run['info']['logs'][stat]]) if isinstance(run['info']['logs'][stat][-1], dict) else max(run['info']['logs'][stat]) for run in run_entries]
    if split != None:
        df[split] = [str(run['config'].get(split)) for run in run_entries]
    # how many epochs did training run for? Early stopping might have kicked in.
    df['epochs'] = [len(run['info']['logs'][max_stats[0]]) for run in run_entries]
    
    ### simple checkbox gui
    # one checkbox per metric
    checkboxes = [widgets.Checkbox(description=col,) for col in df.columns if col != x]
    # plot button
    plot_button = widgets.Button(description='Plot',button_style='success')
    # button_style one of 'success', 'info', 'warning', 'danger' or ''
    log_buttons = [widgets.ToggleButton(description=f'log {i} axis', button_style='info') for i in ['x', 'y']]
    save_button = widgets.Button(description=f'save', button_style='warning')
    
    # output of plotting
    out = widgets.Output()
    def on_button_click(_, save=False):
        """function for plotting ticked metrics on button click"""
        with out:
            # don't neglect to clear the output!
            if not save:
                clear_output()
            plt.figure(figsize=plot_params.get('figsize', None))
            # get metrics that have been ticked
            plot_metrics = [c.description for c in checkboxes if c.value]
            if split != None:
                split_values = df[split].unique()
                plot_dfs = [df[df[split] == s] for s in split_values]
            else:
                plot_dfs = [df]
            for pdf in plot_dfs:
                for m in plot_metrics:
                    plt.plot(pdf.sort_values(x)[x], pdf.sort_values(x)[m], marker='o')
                if log_buttons[0].value:
                    plt.xscale('log')
                if log_buttons[1].value:
                    plt.yscale('log')
                # sort and print run numbers according to order they appear in plot
                print([run for _, run in sorted(zip(pdf[x], pdf.index))])
            if split != None:
                plt.legend([s.replace('_', ' ') for s in split_values], loc= 'best')
            else:
                plt.legend(plot_metrics, loc='best')
            plt.title(plot_params.get('title',plot_metrics))
            plt.xlabel(plot_params.get('xlabel', x.replace('_', ' ')))
            plt.ylabel(plot_params.get('ylabel', 'value'))
            labels = df.index
            if save:
                plt.savefig('plots/' + plot_params.get('title',str(plot_metrics)), bbox_inches = "tight", dpi=200)
            else:
                plt.show()
    # linking button and function together using a button's method
    plot_button.on_click(on_button_click)
    # displaying button and its output together
    display(widgets.VBox(checkboxes + log_buttons + [save_button, plot_button,out]))
    save_button.on_click(lambda x: on_button_click(x, save=True))
    if return_df:
        return df


def plot_sacred_training(id_list, x=None, epoch_lim = 10000, parameter_is_list=False, index_of_interest=-1, return_df=False, plot_params={}):
    """retrieves and plots runs from sacred mongodb
    
    Arguments:
    id_list -- list of run ids
    x -- hyperparameter of interest
    split -- config hyperparameter to split plotting by (multiple lines on same plot)
    parameter_is_list -- specific case where a hyperparameter may be a list
    index_of_interest -- if hyperapameter of interest is a list, which index of list do we care about?
    
    """
    # establish connection to database
    client = pymongo.MongoClient()
    fs = gridfs.GridFS(client.sacred)
    runs = client.sacred.runs
    metrics = client.sacred.metrics

    # determine which runs are needed
    run_entries = list(runs.find({'_id': {'$in': id_list}}))
    # metric_ids = {m['name']: ObjectId(m['id']) for m in run_entry['info']['metrics']}
    df = pd.DataFrame()
    
    if x != None:
        # What is the hyperparameter of interst? (labels)
        df[x] = [run['config'][x] for run in run_entries]
    
    stats = [k for k in run_entries[0]['info']['logs'].keys() if any(x in k for x in ['categ', 'loss', 'mse', 'acc'])]
    for stat in stats:
        df[stat] = [[step['value'] for step in run['info']['logs'][stat]] if isinstance(run['info']['logs'][stat][-1], dict) else run['info']['logs'][stat] for run in run_entries]
    
    df.index = [run['_id'] for run in run_entries]
    ### simple checkbox gui
    # one checkbox per metric
    checkboxes = [widgets.Checkbox(description=col,) for col in df.columns if col != x]
    # plot button
    plot_button = widgets.Button(description='Plot',button_style='success')
    # button_style one of 'success', 'info', 'warning', 'danger' or ''
    log_buttons = [widgets.ToggleButton(description=f'log {i} axis', button_style='info') for i in ['x', 'y']]
    save_button = widgets.Button(description=f'save', button_style='warning')
    # output of plotting
    out = widgets.Output()
    def on_button_click(_, save=False):
        """function for plotting ticked metrics on button click"""
        with out:
            # don't neglect to clear the output!
            if not save:
                clear_output()
            plt.figure(figsize=plot_params.get('figsize', None))
            # get metrics that have been ticked
            plot_metrics = [c.description for c in checkboxes if c.value]
            legend_values = []
            for index, row in df.iterrows():
                for m in plot_metrics:
                    plt.plot([i for i in range(len(row[m][:epoch_lim]))], row[m][:epoch_lim])
                    metric = 'train ' + m if m[:3] != 'val' else m.replace('_', ' ')
                    if x != None:
                        x_string = x.replace('_', ' ')
                        legend_values.append(f'run {index}, {metric}, {x_string} {row[x]}')
                    else:
                        legend_values.append(f'run {index}, {metric}')
                if log_buttons[0].value:
                    plt.xscale('log')
                if log_buttons[1].value:
                    plt.yscale('log')
                # sort and print run numbers according to order they appear in plot
#                 print([run for _, run in sorted(zip(pdf[x], pdf.index))])
            plt.legend(legend_values, loc='best')
            plt.title(plot_params.get('title',plot_metrics))
            plt.xlim(plot_params.get('xlim', None))
            plt.ylim(plot_params.get('ylim', None))
            plt.xlabel(plot_params.get('xlabel', 'epoch'))
            plt.ylabel(plot_params.get('ylabel', 'value'))
            labels = df.index
            if save:
                plt.savefig('plots/' + plot_params.get('title',str(plot_metrics)), bbox_inches = "tight", dpi=200)
            else:
                plt.show()
                
    # linking button and function together using a button's method
    plot_button.on_click(on_button_click)
    save_button.on_click(lambda x: on_button_click(x, save=True))
    # displaying button and its output together
    display(widgets.VBox(checkboxes + log_buttons + [save_button, plot_button,out]))
    if return_df:
        return df

