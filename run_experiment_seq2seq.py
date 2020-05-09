from importlib import reload
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
import src.losses as losses

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('more_capacity_test')
ex.observers.append(MongoObserver(db_name='sacred'))

# seem to need this to use my custom loss function, see here: https://github.com/tensorflow/tensorflow/issues/34944
# last answer might fix it: https://stackoverflow.com/questions/57704771/inputs-to-eager-execution-function-cannot-be-keras-symbolic-tensors
# the trick is for the step that defines the loss fnc to return a symbolic tensor, rather than returning another function which uses a symbolic tensor
tf.compat.v1.disable_eager_execution()
# alternatively, could do something like this?
# https://github.com/Douboo/tf_env_debug/blob/master/custom_layers_and_model_subclassing_API.ipynb

@ex.config
def train_config():
    # data params
    model_inputs = ['H', 'V_mean']
    model_outputs = ['H', 'V']
    seq_length = 32
    use_base_key = True
    transpose = False
    st = 0
    nth_file = None
    vel_cutoff = 4

    ##### Model Config ####
    ### general network params
    hierarchical = True
    variational = False
    latent_size = 400
    hidden_state = 800
    dense_size = 800
    dense_layers = 2
    recurrent_dropout=0.0

    ### encoder params
    encoder_lstms = 2
    z_activation = None

    ### sampling params... if applicable.
    epsilon_std=1

    ### decoder params
    decoder_lstms=2
    # ar_inputs only works as parameter for non hierarchical graph, currently
    ar_inputs = None
    conductors=2
    conductor_steps=2
    conductor_state_size=None # none => same as decoder
    initial_state_from_dense=True
    initial_state_activation='tanh'

    ##### Training Config ####
    batch_size = 64
    lr = 0.0001
    epochs = 300
    monitor = 'loss'
    # musicvae used 48 for 2-bars, 256 for 16 bars (see https://arxiv.org/pdf/1803.05428.pdf)
    free_bits=0
    clipvalue = 1
    # loss = losses.vae_custom_loss
    loss = 'mse'
    kl_weight = 1
    metrics = ['accuracy', 'categorical_crossentropy', 'mse']

    #other
    continue_run = None
    log_tensorboard = False


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
                vel_cutoff,
                
                # network params
                hierarchical,
                variational,
                latent_size,
                hidden_state,
                dense_size,
                dense_layers,
                recurrent_dropout,
                encoder_lstms,
                z_activation,

                # sampling params
                epsilon_std,

                # decoder params
                decoder_lstms,
                ar_inputs,
                conductors,
                conductor_steps,
                conductor_state_size,
                initial_state_from_dense,
                initial_state_activation,
                
                # training params
                batch_size,
                lr,
                epochs,
                monitor,
                free_bits,
                clipvalue,
                loss,
                kl_weight,
                metrics,

                #other
                continue_run,
                log_tensorboard):
    
    no, path = exp_utils.set_up_path(_run._id)
    
    # save text file with the parameters used
    with open(f'{path}description.txt', 'w') as f:
        for key, value in locals().items():
            f.write(f'{key} = {value}\n')
        

    # get training data
    assert seq_length % 4 == 0, 'Sequence length must be divisible by 4'
    model_datas_train, seconds = data.folder2examples('training_data/midi_train', sparse=True, use_base_key=use_base_key, beats_per_ex=int(seq_length / 4), nth_file=nth_file, vel_cutoff=vel_cutoff)
    _run.info['seconds_train_data'] = seconds
    model_datas_val, seconds = data.folder2examples('training_data/midi_val', sparse=True, use_base_key=use_base_key, beats_per_ex=int(seq_length / 4))
    _run.info['seconds_val_data'] = seconds

    model_input_reqs, model_output_reqs = models.get_model_reqs(model_inputs, model_outputs)

    callbacks = []
    # train loss model checkpoint
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(path + f'{no}_best_train_weights.hdf5',
                                monitor='loss', verbose=1, save_best_only=True, save_weights_only=True))
    # val loss model checkpoint
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(path + f'{no}_best_val_weights.hdf5',
                                monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True))
    # early stopping, if needed
    # callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=5))
    # log keras info to sacred
    callbacks.append(exp_utils.KerasInfoUpdater(_run))
    # log to tensorboard
    if log_tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='experiments/tb/', histogram_freq = 1))


    z, model_inputs = models.create_LSTMencoder_graph(model_input_reqs,
                                                    hidden_state_size = hidden_state,
                                                    dense_size=dense_size,
                                                    latent_size=latent_size,
                                                    seq_length=seq_length,
                                                    recurrent_dropout=recurrent_dropout,
                                                    variational=variational)
    if variational:
        loss = loss(z, free_bits=free_bits, kl_weight=kl_weight)
        sampling_fn = models.sampling(batch_size, epsilon_std=epsilon_std)
        z = layers.Lambda(sampling_fn)(z)
        
    
    if hierarchical:
        pred, ar_inputs = models.create_hierarchical_decoder_graph(z,
                                                                model_output_reqs,
                                                                seq_length=seq_length,
                                                                ar_inputs=ar_inputs, 
                                                                # dense and lstm sizes
                                                                dense_size=dense_size,
                                                                hidden_state_size=hidden_state,
                                                                decoder_lstms=decoder_lstms,
                                                                conductor_state_size=conductor_state_size, # none => same as decoder
                                                                # conductor configuration
                                                                conductors=conductors,
                                                                conductor_steps=conductor_steps,
                                                                initial_state_from_dense=initial_state_from_dense,
                                                                initial_state_activation=initial_state_activation,
                                                                recurrent_dropout=recurrent_dropout)
    else:
        pred, ar_inputs = models.create_LSTMdecoder_graph_ar(z,
                                                            model_output_reqs,
                                                            seq_length=seq_length,
                                                            hidden_state_size=hidden_state,
                                                            dense_size=dense_size,
                                                            ar_inputs=ar_inputs,
                                                            recurrent_dropout=recurrent_dropout)
    autoencoder = tf.keras.Model(inputs=model_inputs + ar_inputs, outputs=pred, name=f'autoencoder')
    autoencoder.summary()

    if continue_run != None:
        autoencoder.load_weights(f'experiments/run_{continue_run}/{continue_run}_best_train_weights.hdf5')



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
    autoencoder.compile(optimizer=opt, loss=loss, metrics=metrics)
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
