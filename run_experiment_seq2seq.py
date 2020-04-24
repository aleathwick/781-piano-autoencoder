from importlib import reload
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, csr_matrix
import pickle
import json
import pretty_midi
import sys
from collections import namedtuple
import timeit
import tensorflow as tf
from tensorflow.keras import layers
# import my modules
import src.midi_utils as midi_utils
import src.data as data
import src.models as models
import src.ml_classes as ml_classes
import src.exp_utils as exp_utils

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('autoencoder')
ex.observers.append(MongoObserver(db_name='sacred'))

@ex.config
def my_config():
    # data params
    model_inputs = ['H', 'V_mean']
    model_outputs = ['H', 'V']
    seq_length = 32
    use_base_key = True
    transpose = False
    st = 0
    nth_file = None

    # network params
    hidden_state = 512
    lstm_layers = 2
    dense_layers = 1
    dense_size = 512
    latent_size = 256
    batch_size = 128

    # training params
    lr = 0.0001
    epochs = 60
    monitor = 'loss'
    clipvalue = 1
    loss = 'categorical_crossentropy' 


@ex.automain
def train_model(_run,
                # data params
                model_inputs,
                model_outputs,
                seq_length,
                use_base_key,
                transpose,
                st,
                nth_file,
                
                # network params
                hidden_state,
                lstm_layers,
                dense_layers,
                dense_size,
                latent_size,
                batch_size,
                
                # training params
                lr,
                epochs,
                monitor,
                clipvalue,
                loss):
    
    no, path = exp_utils.set_up_path(_run)
    
    # save text file with the parameters used
    with open(f'{path}description.txt', 'w') as f:
        for key, value in locals().items():
            f.write(f'{key} = {value}\n')

    # get training data
    assert seq_length % 4 == 0, 'Sequence length must be divisible by 4'
    model_datas_train, seconds = data.folder2examples('training_data/midi_train', sparse=True, use_base_key=use_base_key, beats_per_ex=int(seq_length / 4), nth_file=nth_file)
    _run.info['seconds_train_data'] = seconds
    model_datas_val, seconds = data.folder2examples('training_data/midi_val', sparse=True, use_base_key=use_base_key, beats_per_ex=int(seq_length / 4))
    _run.info['seconds_val_data'] = seconds

    model_input_reqs, model_output_reqs = models.get_model_reqs(model_inputs, model_outputs)

    checkpoint_train = tf.keras.callbacks.ModelCheckpoint(path + f'{no}_best_train_weights.hdf5',
                                monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
    checkpoint_val = tf.keras.callbacks.ModelCheckpoint(path + f'{no}_best_val_weights.hdf5',
                                monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    # early stopping, if needed
    # stop = tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=5)
    callbacks = [checkpoint_train, checkpoint_val, exp_utils.KerasInfoUpdater(_run)]


    # create model
    seq_model = models.create_simple_LSTM_RNN(model_input_reqs, model_output_reqs, seq_length=seq_length, dense_layers=dense_layers, dense_size=dense_size)
    seq_model.summary()

    z, model_inputs = models.create_LSTMencoder_graph(model_input_reqs, hidden_state_size = hidden_state, dense_size=dense_size, latent_size=latent_size, seq_length=seq_length)
    pred, ar_inputs = models.create_LSTMdecoder_graph2(z, model_output_reqs, seq_length=seq_length, hidden_state_size = hidden_state, dense_size=dense_size)

    autoencoder = tf.keras.Model(inputs=model_inputs + ar_inputs, outputs=pred, name=f'autoencoder')
    autoencoder.summary()


    # save a plot of the model
    # tf.keras.utils.plot_model(seq_model, to_file=f'{path}model_plot.png')

    dg = ml_classes.ModelDataGenerator([md for md in model_datas_train.values()],
                                        [model_in.name for model_in in model_input_reqs],
                                        [model_out.name for model_out in model_output_reqs],
                                        t_force=True, batch_size = batch_size, seq_length=seq_length)

    dg_val = ml_classes.ModelDataGenerator([md for md in model_datas_val.values()],
                                        [model_in.name for model_in in model_input_reqs],
                                        [model_out.name for model_out in model_output_reqs],
                                        t_force=True, batch_size = batch_size, seq_length=seq_length)

    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=clipvalue)
    autoencoder.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    history = autoencoder.fit(dg, validation_data=dg_val, epochs=epochs, callbacks=callbacks, verbose=1)

    # save the model history
    with open(f'{path}history-{epochs}epochs.json', 'w') as f:
        json.dump(str(history.history), f)

    # add weights to sacred
    exp_utils.capture_weights(_run)

    # save a graph of the training vs validation progress
    models.plt_metric(history.history)
    plt.savefig(f'{path}model_training')
    # clear the output
    plt.clf()
