from importlib import reload
import numpy as np
import os
import copy
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import LearningRateScheduler
# import my modules
import src.midi_utils as midi_utils
import src.data as data
import src.models as models
import src.ml_classes as ml_classes
import src.exp_utils as exp_utils
import src.losses as losses

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('like 282, beta 0.05 -> 0.08')
ex.observers.append(MongoObserver(db_name='sacred'))

### take care of output

ex.captured_out_filter = lambda captured_output: "Output capturing turned off."

from sacred.utils import apply_backspaces_and_linefeeds
ex.captured_out_filter = apply_backspaces_and_linefeeds


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
    sub_beats = 2
    use_base_key = True
    transpose = False
    st = 0
    nth_file = None
    vel_cutoff = 4
    data_folder_prefix = '_8'

    ##### Model Config ####
    ### general network params
    hierarchical = False
    variational = True
    latent_size = 512
    hidden_state = 1024
    dense_size = 1024
    dense_layers = 2
    recurrent_dropout=0.0

    ### encoder params
    encoder_lstms = 2
    z_activation = None
    conv = False
    # pitch_stride = 6
    # conv = {'F_n': [32, 32, 48, 48, 48, 24], # number of filters
    #         'F_s': [(8,12), (4,4), (4,4), (4,4), (4,4), (4,4)], # size of filters
    #         'strides': [(1, pitch_stride), (1, 1), (2, 1), (2,1), (2,1), (2,2)],  # strides
    #         'batch_norm': True # apply batch norm after each conv operation (after activation)
    #         }


    ### sampling params... if applicable.
    epsilon_std=1

    ### decoder params
    decoder_lstms=2
    # ar_inputs only works as parameter for non hierarchical graph, currently
    ar_inputs = None
    conductors=2
    conductor_steps= int(seq_length/16)
    conductor_state_size=None # none => same as decoder
    initial_state_from_dense=True
    initial_state_activation='tanh'

    ##### Training Config ####
    batch_size = 64
    lr = 0.001
    lr_decay_rate = 0.2**(1/1500)
    min_lr = 0.00005
    epochs = 2
    monitor = 'loss'
    loss_weights = [1, 3]
    clipvalue = 1
    loss = losses.vae_custom_loss
    # loss = 'categorical_crossentropy'
    metrics = ['accuracy', 'categorical_crossentropy']

    # musicvae used 48 free bits for 2-bars, 256 for 16 bars (see https://arxiv.org/pdf/1803.05428.pdf)
    # Variational specific parameters
    max_beta = 0.08
    beta_rate = 0.2**(1/1000) # at 1000 epochs, we want (1 - something) * max_beta
    free_bits=0
    kl_weight = 1
    
    #other
    continue_run = None
    log_tensorboard = False
    ar_inc_batch_shape = False # sometimes needed to make training work...
     


@ex.automain
def train_model(_run,
                # data params
                model_inputs,
                model_outputs,
                seq_length,
                sub_beats,
                use_base_key,
                transpose,
                st,
                nth_file,
                vel_cutoff,
                data_folder_prefix,
                
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
                conv,

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
                lr_decay_rate,
                min_lr,
                epochs,
                monitor,
                loss_weights,
                clipvalue,
                loss,
                max_beta,
                beta_rate,
                free_bits,
                kl_weight,
                metrics,

                #other
                continue_run,
                log_tensorboard,
                ar_inc_batch_shape):
    
    no, path = exp_utils.set_up_path(_run._id)
    
    # save text file with the parameters used
    with open(f'{path}description.txt', 'w') as f:
        for key, value in locals().items():
            f.write(f'{key} = {value}\n')
        

    # get training data
    assert seq_length % 4 == 0, 'Sequence length must be divisible by 4'
    model_datas_train, seconds = data.folder2examples('training_data/midi_train' + data_folder_prefix, sparse=True, use_base_key=use_base_key, beats_per_ex=int(seq_length / sub_beats), nth_file=nth_file, vel_cutoff=vel_cutoff, sub_beats=sub_beats)
    _run.info['seconds_train_data'] = seconds
    model_datas_val, seconds = data.folder2examples('training_data/midi_val' + data_folder_prefix, sparse=True, use_base_key=use_base_key, beats_per_ex=int(seq_length / sub_beats), sub_beats=sub_beats)
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
    # learning rate scheduler
    callbacks.append(LearningRateScheduler(exp_utils.decay_lr(min_lr, lr_decay_rate, _run)))
    # log to tensorboard
    if log_tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='experiments/tb/', histogram_freq = 1))

    # model kwargs - for the encoder/decoder builder functions, make a dictionary to pass as kwargs
    model_kwargs = {# general model parameters
                    'recurrent_dropout':recurrent_dropout,
                    'hidden_state':hidden_state,
                    'dense_size':dense_size,
                    'seq_length':seq_length,
                    
                    # encoder parameters
                    'latent_size':latent_size,
                    'z_activation':z_activation,
                    'variational':variational,
                    'conv':conv,
                    
                    # decoder parameters
                    'ar_inputs':ar_inputs, 
                    'decoder_lstms':decoder_lstms,
                    'batch_size':batch_size, # not used in encoder, currently...
                    'initial_state_from_dense':initial_state_from_dense,
                    'initial_state_activation':initial_state_activation,
                    'ar_inc_batch_shape':ar_inc_batch_shape,
                    'conv':conv,
                    # conductor configuration (only used if hierarchical)
                    'conductor_state_size':conductor_state_size, # none => same as decoder
                    'conductors':conductors,
                    'conductor_steps':conductor_steps,
                    }
    # if variational, z will be a list of [[means], [stds]]
    build_encoder_graph = models.create_LSTMencoder_graph
    z, model_inputs_tf = build_encoder_graph(model_input_reqs, **model_kwargs)

    if variational:
        beta_fn = exp_utils.beta_fn2(beta_rate, max_beta)
        loss_for_train, beta_cb = loss(z, beta_fn, free_bits=free_bits, kl_weight=kl_weight, run=_run)
        sampling_fn = models.sampling(batch_size, epsilon_std=epsilon_std)
        # z_input is the tensor that will be passed into the decoder
        z_input = layers.Lambda(sampling_fn)(z)
        if not isinstance(beta_fn, (int, float)):
            callbacks.append(beta_cb)
    else:
        loss_for_train = loss
        z_input = z
    
    if hierarchical:
        build_decoder_graph = models.create_hierarchical_decoder_graph
    else:
        build_decoder_graph =models.create_LSTMdecoder_graph_ar

    pred, ar_inputs_tf = build_decoder_graph(z_input, model_output_reqs, **model_kwargs)
    autoencoder = tf.keras.Model(inputs=model_inputs_tf + ar_inputs_tf, outputs=pred, name=f'autoencoder')
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
    autoencoder.compile(optimizer=opt, loss=loss_for_train, metrics=metrics, loss_weights=loss_weights)
    history = autoencoder.fit(dg, validation_data=dg_val, epochs=epochs, callbacks=callbacks, verbose=2)

    # save the model history
    with open(f'{path}history-{epochs}epochs.json', 'w') as f:
        json.dump(str(history.history), f)

    # add weights to sacred... Or not, they can exceed max size! 
    # exp_utils.capture_weights(_run)

    # save a graph of the training vs validation progress
    models.plt_metric(history.history)
    plt.savefig(f'{path}model_training')
    # clear the output
    plt.clf()


    ### Make some predictions ###
    # load best weights
    # models.load_weights_safe(autoencoder, path + f'{no}_best_train_weights.hdf5', by_name=False)
    # get some random examples from the validation data
    random_examples, idx = data.n_rand_examples(model_datas_val, n=batch_size)

    # currently, prediction is broken for my variational models
    # if variational, then for now need to circumvent autoencoder.predict()
    if variational:
        #grab encoder layers from above, make a model
        encoder = tf.keras.Model(inputs=model_inputs_tf, outputs=z, name=f'encoder')
        
        # create a new decoder
        z_in = tf.keras.Input(shape=(latent_size,), name='z')
        pred, ar_inputs_tf = build_decoder_graph(z_in, model_output_reqs, **model_kwargs)
        decoder = tf.keras.Model(inputs=[z_in] + ar_inputs_tf, outputs=pred, name=f'decoder')

        # load weights for decoder
        # models.load_weights_safe(decoder, path + f'{no}_best_train_weights.hdf5', by_name=True)

        # in_dict = dg_val.__getitem__(0)[0]

        ### predict
        # get paramerterization of latent space
        zp_param = encoder.predict(random_examples)
        # generate random values
        zp_latent = sampling_fn(zp_param)
        with tf.compat.v1.Session():
            random_examples['z'] = zp_latent.eval()
        pred = decoder.predict(random_examples)

    else:
        pred = autoencoder.predict(random_examples)
    
    # find axis that corresponds to velocity
    v_index = np.where(np.array(autoencoder.output_names) == 'V_out')[0][0]
    print('velocity index:', v_index)
    model_datas_pred, _ = data.folder2examples('training_data/midi_val' + data_folder_prefix, sparse=False, use_base_key=use_base_key, beats_per_ex=int(seq_length / sub_beats), sub_beats=sub_beats)
    model_datas = copy.deepcopy(model_datas_pred)
    model_datas_pred['V'].data[idx,...] = np.array(pred)[v_index,:,:,:]
    os.mkdir(path + 'midi/')
    for i in idx:
        pm_original = data.examples2pm(model_datas, i, sub_beats=sub_beats)
        pm_pred = data.examples2pm(model_datas_pred, i, sub_beats=sub_beats)
        pm_original.write(path + 'midi/' + f'ex{i}original.mid')
        pm_pred.write(path + 'midi/' + f'ex{i}prediction_teacher_forced.mid')