from importlib import reload
import traceback
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
ex = Experiment(f'32seq-cce-{sys.argv[2:]}')
# ex = Experiment(f'32seq-no ar V')
ex.observers.append(MongoObserver(db_name='sacred'))

### take care of output

# ex.captured_out_filter = lambda captured_output: "Output capturing turned off."

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
    model_inputs = ['Pn', 'TBn', 'TSBn']
    model_outputs = ['Vn']
    seq_length = 50
    sub_beats = 2
    use_base_key = False
    transpose = False
    st = 0
    nth_file = None
    vel_cutoff = 4
    data_folder_prefix = '_8'

    ##### Model Config ####
    ### general network params
    hidden_state = 512
    recurrent_dropout=0.0

    ### encoder params
    bi_encoder_lstms = 2
    uni_encoder_lstms = 2
    conv = False
    ar_inputs = None
    # pitch_stride = 6
    # conv = {'F_n': [32, 32, 48, 48, 48, 24], # number of filters
    #         'F_s': [(8,12), (4,4), (4,4), (4,4), (4,4), (4,4)], # size of filters
    #         'strides': [(1, pitch_stride), (1, 1), (2, 1), (2,1), (2,1), (2,2)],  # strides
    #         'batch_norm': True # apply batch norm after each conv operation (after activation)
    #         }

    ##### Training Config ####
    batch_size = 64
    lr = 0.001
    lr_decay_rate = 0.3**(1/1500)
    min_lr = 0.00005
    epochs = 1500
    monitor = 'loss'
    loss_weights = None
    clipvalue = 1
    loss = 'categorical_crossentropy'
    metrics = ['accuracy', 'categorical_crossentropy', 'mse']

    # musicvae used 48 free bits for 2-bars, 256 for 16 bars (see https://arxiv.org/pdf/1803.05428.pdf)
    # Variational specific parameters
    max_beta = 3
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
                hidden_state,
                recurrent_dropout,
                bi_encoder_lstms,
                uni_encoder_lstms,
                conv,
                ar_inputs,
                
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
        
    model_datas_train, seconds = folder2nbq('training_data/midi_train' + data_folder_prefix, 
                                            return_ModelData_object=True,
                                            seq_length=seq_length, 
                                            sub_beats=sub_beats, 
                                            example_bars_skip=example_bars_skip, 
                                            use_base_key=use_base_key, 
                                            nth_file=nth_file, 
                                            vel_cutoff=vel_cutoff)
    _run.info['seconds_train_data'] = seconds
    model_datas_train, seconds = folder2nbq('training_data/midi_val' + data_folder_prefix, 
                                            return_ModelData_object=True,
                                            seq_length=seq_length, 
                                            sub_beats=sub_beats, 
                                            example_bars_skip=example_bars_skip, 
                                            use_base_key=use_base_key, 
                                            vel_cutoff=vel_cutoff)
    _run.info['seconds_val_data'] = seconds

    model_input_reqs, model_output_reqs = models.get_model_reqs(model_inputs, model_outputs, sub_beats=sub_beats)

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
                    'seq_length':seq_length,
                    'uni_encoder_lstms':uni_encoder_lstms,
                    'bi_encoder_lstms':bi_encoder_lstms,
                    'conv':conv,
                    
                    # decoder parameters
                    'ar_inputs':ar_inputs,
                    'batch_size':batch_size, # not used in encoder, currently...
                    'ar_inc_batch_shape':ar_inc_batch_shape,
                    'conv':conv,
                    }

    model_inputs_tf, pred = create_nbq_bi_graph(model_input_reqs, model_output_reqs, **model_kwargs)


    model = tf.keras.Model(inputs=model_inputs_tf + ar_inputs_tf, outputs=pred, name=f'bi_uni_model')
    model.summary()





    # save a plot of the model
    # tf.keras.utils.plot_model(seq_model, to_file=f'{path}model_plot.png')

    dg = ml_classes.ModelDataGenerator([md for md in model_datas_train.values()],
                                        [model_in.name for model_in in model_input_reqs if model_in.md],
                                        [model_out.name for model_out in model_output_reqs if model_out.md],
                                        t_force=True, batch_size = batch_size, seq_length=seq_length,
                                        sub_beats=sub_beats, V_no_zeros=False)

    dg_val = ml_classes.ModelDataGenerator([md for md in model_datas_val.values()],
                                        [model_in.name for model_in in model_input_reqs if model_in.md],
                                        [model_out.name for model_out in model_output_reqs if model_out.md],
                                        t_force=True, batch_size = batch_size, seq_length=seq_length,
                                        sub_beats=sub_beats, V_no_zeros=False)

    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=clipvalue)
    

    
    model.compile(optimizer=opt, loss=loss, metrics=metrics, loss_weights=loss_weights)

    try:
        print('freshly initialized:')
        print(autoencoder.metrics_names)
        print(autoencoder.evaluate(dg_val.__getitem__(0)[0], dg_val.__getitem__(0)[1], batch_size=batch_size))
    except:
        print('evaluation failed')
        print(traceback.format_exc())

    if continue_run != None:
        autoencoder.load_weights(f'experiments/run_{continue_run}/{continue_run}_best_train_weights.hdf5')
    
    try:
        print('loaded weights:')
        print(autoencoder.metrics_names)
        print(autoencoder.evaluate(dg_val.__getitem__(0)[0], dg_val.__getitem__(0)[1], batch_size=batch_size))
    except:
        print('evaluation failed')
        print(traceback.format_exc())

    history = model.fit(dg, validation_data=dg_val, epochs=epochs, callbacks=callbacks, verbose=2)

    # attempt to evaluate the model
    try:
        print('freshly trained:')
        print(autoencoder.metrics_names)
        print(autoencoder.evaluate(dg_val.__getitem__(0)[0], dg_val.__getitem__(0)[1], batch_size=batch_size))
    except:
        print('evaluation failed')
        print(traceback.format_exc())

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

    pred = autoencoder.predict(random_examples)
    
    # now need function that returns nbq to midi...

    # # find axis that corresponds to velocity
    # v_index = np.where(np.array(autoencoder.output_names) == 'V_out')[0][0]
    # print('velocity index:', v_index)
    # model_datas_pred, _ = data.folder2examples('training_data/midi_val' + data_folder_prefix, sparse=False, use_base_key=use_base_key, beats_per_ex=int(seq_length / sub_beats), sub_beats=sub_beats)
    # model_datas = copy.deepcopy(model_datas_pred)
    # model_datas_pred['V'].data[idx,...] = np.array(pred)[v_index,:,:,:]
    # os.mkdir(path + 'midi/')
    # for i in idx:
    #     pm_original = data.examples2pm(model_datas, i, sub_beats=sub_beats)
    #     pm_pred = data.examples2pm(model_datas_pred, i, sub_beats=sub_beats)
    #     pm_original.write(path + 'midi/' + f'ex{i}original.mid')
    #     pm_pred.write(path + 'midi/' + f'ex{i}prediction_teacher_forced.mid')