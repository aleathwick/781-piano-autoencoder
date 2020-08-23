import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

seq_length = 64
n_notes = 88

# # see https://blog.keras.io/building-autoencoders-in-keras.html
# def sampling(args):
#     z_mean, z_log_sigma = args
#     epsilon = tf.keras.backend.random_normal(shape=(batch_size, latent_dim),
#                               mean=0., std=epsilon_std)
#     return z_mean + K.exp(z_log_sigma) * epsilon

########## model inputs and outputs ##########

def get_model_reqs(model_inputs, model_outputs, sub_beats=2, seq_length=50, **kwargs):
    """Given a list of model input and model output names, returns namedtuple objects with attributes for those in/outputs. 
    
    Notes:
    This function stores information on the standard way to deal with model in/outputs.
    That way, when building models, in/outputs only need to be referred to by name.
    In practice, this may hide some changes if I ever changed output activations between different models.

    """
    n_notes=88
    # I assume that data, aside from the sequential dimension, will never have more than 1 dimension
    # md - refers to whether or not an md object is normally supplied to the data generator
    # ohe - is this input one hot encoded? 
    model_input = namedtuple('input', 'name dim seq md ohe transposable') 
    model_output = namedtuple('output', 'name dim activation seq md ohe transposable') 

    # model input requirements
    model_input_reqs_unfiltered = [
                                ### HOV inputs
                                model_input('H', n_notes, True, True, ohe=False, transposable=True),
                                model_input('tempo', 1, False, True, ohe=False, transposable=False),
                                model_input('key', 12, False, True, ohe=True, transposable=True),
                                model_input('V_mean', 1, False, True, ohe=False, transposable=False),
                                model_input('beat_indicators', 4, True, False, ohe=True, transposable=False),
                                model_input('sub_beat_indicators', sub_beats, True, False, ohe=True, transposable=False),

                                ### nbq inputs
                                # TSn and TEn should really be ohe, but max sub beat is not determined ahead of time...
                                # could remedy this by taking sub beat mod 64, or something like that
                                model_input('TSn', seq_length, True, True, ohe=False, transposable=False), # note starts in sub beats
                                model_input('TEn', seq_length, True, True, ohe=False, transposable=False), # note ends in sub beats
                                model_input('TBn', 4, True, True, ohe=True, transposable=False), # note starts in beats of bar 
                                model_input('TMn', 16, True, True, ohe=True, transposable=False), # note starts in beats of bar 
                                model_input('TSBn', sub_beats, True, True, ohe=True, transposable=False), # note starts in sub beats of beat
                                model_input('Pn', n_notes, True, True, ohe=True, transposable=True), # pitch
                                model_input('PSn', 1, True, True, ohe=False, transposable=False), # pitch as continuous value
                                model_input('PCn', 12, True, True, ohe=True, transposable=True)] # pitch class

    # model output requirements
    model_output_reqs_unfiltered = [
                                    ### HOV inputs
                                    model_output('H', n_notes, 'sigmoid', True, True, ohe=False, transposable=True),
                                    model_output('O', n_notes, 'tanh', True, True, ohe=False, transposable=True),
                                    model_output('V', n_notes, 'sigmoid', True, True, ohe=False, transposable=True),
                                    
                                    # nbq outputs
                                    model_output('TEn', seq_length, 'softmax', True, True, ohe=False, transposable=False),
                                    model_output('Vn', 1, 'sigmoid', True, True, ohe=False, transposable=False)]

    if model_inputs == 'all' or model_outputs == 'all':
        model_reqs = {m_input.name: m_input for m_input in model_input_reqs_unfiltered}
        model_reqs.update({m_output.name: m_output for m_output in model_output_reqs_unfiltered})
        return model_reqs
                                    

    model_input_reqs = [m_input for m_input in model_input_reqs_unfiltered if m_input.name in model_inputs]
    model_output_reqs = [m_output for m_output in model_output_reqs_unfiltered if m_output.name in model_outputs]

    return model_input_reqs, model_output_reqs

def get_inputs(model_input_reqs, seq_length):
    """Given a list of named tuples that list model inputs and attributes, produces relevant keras model inputs
    
    Notes:
    Inputs may either be sequential or auxillary, i.e. non-sequential.
    """
    seq_inputs = [tf.keras.Input(shape=(seq_length,seq_in.dim), name=seq_in.name + '_in') for seq_in in model_input_reqs if seq_in.seq == True]
    aux_inputs = [tf.keras.Input(shape=(aux_in.dim,), name=aux_in.name + '_in') for aux_in in model_input_reqs if aux_in.seq == False]
    return seq_inputs, aux_inputs


########## loading weights ##########

# see https://github.com/keras-team/keras/issues/5397#issuecomment-583898997
def load_weights_safe(model, weights_path, by_name=True):
    """loads weights (in place), checks which layers had their weights change, prints names of any that aren't changed
    
    Arguments:
    model -- a keras model
    weights_path -- path to a weights file to load
    by_name -- bool, whether or not to load layer weights by name
    
    """
    initial_weights = [layer.get_weights() for layer in model.layers]
    model.load_weights(weights_path, by_name=by_name)
    missing=False
    for layer, initial in zip(model.layers, initial_weights):
        weights = layer.get_weights()
        if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
            print(f'Checkpoint contained no weights for layer {layer.name}!')
            missing=True
    
    
    if not missing:
        print('weights loaded successfully')
    
    # if any model weights are missing, we MAY wish to revert model back to original weights
    attempt_revert = False
    if missing and attempt_revert:
        revert = True
        for layer, initial in zip(model.layers, initial_weights):
            layer.set_weights(initial)
        for layer, initial in zip(model.layers, initial_weights):
            weights = layer.get_weights()
            if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
                print(f'returned {layer.name} weights to original value')
            else:
                revert = False
                print(f'FAILED to return {layer.name} to original value!')
        if revert:
            print('All layer weights reverted to original values')
    



def sampling(batch_size, epsilon_std=1, **kwargs):
    """sample from the latent space

    Arguments:
    args -- list containing z_mean and z_log_sigma vectors
    latent_size -- size of latent space. Maybe should just infer this from args?
    epsilon_std -- controls stddev of the sample distribution
    
    Returns:
    vector sampled from the latent space

    Notes:
    look here for excellent tutorial: https://blog.keras.io/building-autoencoders-in-keras.html

    """
    def sampling_fn(z):
        z_mean, z_log_sigma = z
        latent_size = z_mean.shape[-1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, latent_size),
                                mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon
    return sampling_fn


########## Basic Recurrent ##########

def create_simple_LSTM_RNN(model_input_reqs, model_output_reqs, seq_length=seq_length, lstm_layers=3, dense_layers=2, LSTM_state_size = 256,
                    dense_size = 128, recurrent_dropout = 0.0, ar_inputs=None):
    
    seq_inputs, aux_inputs = get_inputs(model_input_reqs, seq_length)
    # pass input through non final lstm layers, returning sequences each time
    repeated_inputs = []
    if len(aux_inputs) > 0:
        # layer for repeating aux inputs
        repeat_aux_inputs = layers.RepeatVector(seq_length, name=f'repeat{seq_length}Times')
        repeated_inputs = [repeat_aux_inputs(aux_input) for aux_input in aux_inputs]
    # we might need to concat inputs, if more than 1
    if len(seq_inputs) + len(repeated_inputs) > 1:
        x = layers.concatenate(seq_inputs + repeated_inputs, name='joinModelInput')
    else:
        x = seq_inputs[0]
    # pass through all but last lstm layers
    for _ in range(lstm_layers - 1):
        x = layers.Bidirectional(layers.LSTM(LSTM_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout))(x)
    # pass through final lstm layer
    x = layers.Bidirectional(layers.LSTM(LSTM_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout))(x)

    # pass through non final dense layers
    for _ in range(dense_layers - 1):
        x = layers.Dense(dense_size, activation='relu')(x)

    # pass through final dense layer for output
    outputs = []
    for output in model_output_reqs:
        outputs.append(layers.TimeDistributed(layers.Dense(output.dim, activation=output.activation, name=output.activation), name=output.name + '_out')(x))
    
    model = tf.keras.Model(inputs=seq_inputs + aux_inputs, outputs=outputs, name=f'simple_LSTM')

    return model


########## Encoders ##########

def create_LSTMencoder_graph(model_input_reqs, 
                        seq_length = seq_length,  
                        lstm_layers = 3,  
                        dense_layers = 2,  
                        hidden_state = 512,  
                        latent_size = 256,
                        dense_size = 512,  
                        recurrent_dropout = 0.0,
                        z_activation='relu',
                        variational=False,
                        conv=False, # dictionary containing filter parameters
                        **kwargs):
    """layers for LSTM encoder, returning latent vector as output
    
    Arguments:
    seq_inputs -- inputs that are sequential in nature
    aux_inputs -- other inputs
    seq_length -- time steps per training example
    hidden_state -- size of LSTM hidden state
    supplemental_inputs -- list ints, where each int is the dimension of an input. These inputs will be converted from int to one hot.
    
    """
    # pass input through non final lstm layers, returning sequences each time
    seq_inputs, aux_inputs = get_inputs(model_input_reqs, seq_length)

    repeated_inputs = []
    if len(aux_inputs) > 0:
        # layer for repeating aux inputs
        repeat_aux_inputs = layers.RepeatVector(seq_length, name=f'repeat{seq_length}Times')
        repeated_inputs = [repeat_aux_inputs(aux_input) for aux_input in aux_inputs]
    if len(seq_inputs) + len(repeated_inputs) > 1:
        x = layers.concatenate(seq_inputs + repeated_inputs, name='joinModelInput')
    else:
        x = seq_inputs[0]
    for i in range(lstm_layers - 1):
        x = layers.Bidirectional(layers.LSTM(hidden_state, return_sequences=True,
                                recurrent_dropout=recurrent_dropout, name=f'enc_lstm_{i}'), name=f'bi_enc_lstm_{i}')(x)
    # pass through final lstm layer
    x = layers.Bidirectional(layers.LSTM(hidden_state, return_sequences=False, recurrent_dropout=recurrent_dropout, name=f'enc_lstm_{lstm_layers - 1}'), name=f'bi_enc_lstm_{lstm_layers - 1}')(x)

    if conv != None and conv != False:
        convolved_h = convolve_H(seq_inputs, conv)
        x = layers.concatenate([convolved_h, x], axis=-1)

    # pass through non final dense layers
    for i in range(dense_layers - 1):
        x = layers.Dense(dense_size, activation='relu', name=f'enc_dense_{i}')(x)
    if variational:
        # see here https://github.com/tensorflow/magenta/blob/85ef5267513f62f4a40b01b2a1ee488f90f64a13/magenta/models/music_vae/base_model.py#L201
        # for how musicVAE do this
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None)
        z_mean = layers.Dense(latent_size, activation=z_activation, kernel_initializer=initializer, name='z_mean')(x)
        z_log_sigma = layers.Dense(latent_size, activation='softplus', kernel_initializer=initializer, name='z_log_sigma')(x)
        z = [z_mean, z_log_sigma]
    else:
        z = layers.Dense(latent_size, activation=z_activation, name='z')(x)

    return z, seq_inputs + aux_inputs


def convolve_H(seq_inputs, conv):
    assert len(seq_inputs) == 1, f'There should only be one sequential input, but there are {len(seq_inputs)}'
    x = seq_inputs[0]
    # need to sort out data_format... channels_last, probably, and add extra dimension there
    # default axis is -1
    x = layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, -1))(x)
    
    for i in range(len(conv['F_n'])):
        x = layers.Conv2D(conv['F_n'][i], conv['F_s'][i], strides=conv['strides'][i], padding='same', name=f'enc_conv_{i}')(x)
        x = layers.Activation('relu')(x)
        if conv.get('batch_norm', False):
            x = layers.BatchNormalization(name=f'enc_batch_norm_{i}')(x)
        # probs don't want max pooling...
        # x = layers.MaxPooling2D(pool_size=(2, 2))(x)


    # x = layers.MaxPooling2D(pool_size=(4, 4))(x)
    # could try seperable conv setting data format to channels first, so that pitch and time weights are learnt separately.
    # but then stride can't be designated for the channel dimension.
    # layers.SeparableConv2D(filters, kernel_size, strides=(), )
    # dilation rate could be used like stride...
    # x = layers.Conv2D(5, (16,12), dilation_rate=(16, 12), padding='valid')(x)

    flat_x = layers.Flatten()(x)

    return flat_x


def create_conv_encoder_graph(model_input_reqs, seq_length=64, latent_size = 64):
    seq_inputs, aux_inputs = get_inputs(model_input_reqs, seq_length)
    assert len(seq_inputs) == 1, f'There should only be one sequential input, but there are {len(seq_inputs)}'
    h = seq_inputs[0]
    # need to sort out data_format... channels_last, probably, and add extra dimension there
    # default axis is -1
    # this is ludicrous that I need to use squeeze here...
    x = layers.Lambda(lambda k: tf.squeeze(tf.keras.backend.expand_dims(k, axis=-1), axis=0))(h)
    # x = layers.Lambda(lambda k: k[:,:,tf.newaxis]))(h)
    x = layers.Conv2D(4, (8,12), strides=(4, 12), padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(6, (4,4), strides=(1, 1), padding='same')(x)
    x = layers.Activation('relu')(x)
    # x = layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = layers.Conv2D(8, (3,3), strides=(1, 1), padding='same')(x)
    x = layers.Activation('relu')(x)
    # x = layers.MaxPooling2D(pool_size=(4, 4))(x)
    # could try seperable conv setting data format to channels first, so that pitch and time weights are learnt separately.
    # but then stride can't be designated for the channel dimension.
    # layers.SeparableConv2D(filters, kernel_size, strides=(), )
    # dilation rate could be used like stride...
    # x = layers.Conv2D(5, (16,12), dilation_rate=(16, 12), padding='valid')(x)

    flat_x = layers.Flatten()(x)
    if len(aux_inputs) != 0:
        flat_x = layers.concatenate([flat_x] + aux_inputs, name='joinModelInput')

    z = layers.Dense(latent_size, activation='relu', name='z')(flat_x)

    return z, seq_inputs + aux_inputs
    

########## Decoders ##########

def create_LSTMdecoder_graph(latent_vector, model_output_reqs, seq_length=seq_length,
                    lstm_layers = 2, dense_layers = 2, hidden_state_size = 512, dense_size = 512,
                    n_notes=88, chroma=False, recurrent_dropout = 0.0, **kwargs):
    """creates an LSTM based decoder, NOT autoregressive - handles all outputs at once
    
    Arguments:
    seq_length -- time steps per training example
    hidden_state_size -- size of LSTM hidden state
    supplemental_inputs -- list ints, where each int is the dimension of an input. These inputs will be converted from int to one hot.
    
    """

    x = layers.Dense(dense_size, activation='relu')(latent_vector)
    x = layers.RepeatVector(seq_length)(x)

    # pass input through non final lstm layers, returning sequences each time
    for _ in range(lstm_layers - 1):
        x = layers.Bidirectional(layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout))(x)
    # pass through final lstm layer
    x = layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout)(x)
    
    # attempt to rebuild input
    outputs = []
    for output in model_output_reqs:
        outputs.append(layers.TimeDistributed(layers.Dense(output.dim, activation=output.activation, name=output.activation), name=output.name + '_out')(x))

    return outputs


def create_LSTMdecoder_graph_ar(z,
                            model_output_reqs,
                            seq_length=seq_length,
                            ar_inputs=None,
                            lstm_layers = 2,
                            hidden_state = 256,
                            dense_size = 256,
                            chroma=False,
                            recurrent_dropout = 0.0,
                            stateful=False,
                            initial_state_from_dense=True,
                            initial_state_activation=None,
                            batch_size=None,
                            ar_inc_batch_shape=False,
                            **kwargs):
    """creates an autoregressive LSTM based decoder

    
    Arguments:
    seq_length -- time steps per training example
    hidden_state -- size of LSTM hidden state
    ar_inputs -- either None, in which case all outputs are fed back in auto regressively, or a list of input names
    supplemental_inputs -- list ints, where each int is the dimension of an input. These inputs will be converted from int to one hot.

    Notes:
    I suppose all that would be needed for prediction, would be to use sequence length of 1, and use predictions as feed in for tf inputs?
    
    """
    # check autoregressive inputs - if a list hasn't been received, then ALL outputs will be passed in autoregressively
    if ar_inputs == None:
        ar_inputs = [model_output.name for model_output in model_output_reqs]
    # else:
        # assert set(ar_inputs) <= set([model_output.name for model_output in model_output_reqs]), f'ar_inputs contains invalid output names: {ar_inputs}'

    # initial dense layer to transform z
    x = layers.Dense(dense_size, activation='relu', name='initial_dense')(z)
    # if not stateful:
    x = layers.RepeatVector(seq_length)(x)
    
    # get teacher forced inputs
    # the model will only be stateful if it is being used for prediction
    if not stateful:
        if ar_inc_batch_shape:
            ar_inputs = [tf.keras.Input((seq_length,model_output.dim), batch_size=batch_size, name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True and model_output.name in ar_inputs]
        else:
            ar_inputs = [tf.keras.Input((seq_length,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True and model_output.name in ar_inputs]
    else:
        # if this executes, then the model is being used for prediction, and batch size is 1
        ar_inputs = [tf.keras.Input(batch_shape=(1,seq_length,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True and model_output.name in ar_inputs]
    

    if initial_state_from_dense:
        for i in range(lstm_layers - 1):
            h0 = layers.Dense(hidden_state, activation=initial_state_activation, name=f'lstm_h0_{i}')(z)
            c0 = layers.Dense(hidden_state, activation=initial_state_activation, name=f'lstm_c0_{i}')(z)
            x = layers.LSTM(hidden_state, return_sequences=True, recurrent_dropout=recurrent_dropout, name=f'dec_lstm_{i}')(x, initial_state=[h0, c0])
        h0 = layers.Dense(hidden_state, activation=initial_state_activation, name=f'lstm_h0_final')(z)
        c0 = layers.Dense(hidden_state, activation=initial_state_activation, name=f'lstm_c0_final')(z)
        x = layers.concatenate(ar_inputs + [x], axis=-1)
        x = layers.LSTM(hidden_state, return_sequences=True, recurrent_dropout=recurrent_dropout, stateful=stateful, name='dec_lstm_final')(x, initial_state=[h0, c0])

    else:
        # pass input through non final lstm layers, returning sequences each time
        for i in range(lstm_layers - 1):
            x = layers.LSTM(hidden_state, return_sequences=True, recurrent_dropout=recurrent_dropout, stateful=stateful, name=f'dec_lstm_{i}')(x)
        
        # concat teacher forced inputs with x
        # pass through final lstm layer
        x = layers.concatenate(ar_inputs + [x], axis=-1)
        x = layers.LSTM(hidden_state, return_sequences=True, recurrent_dropout=recurrent_dropout, stateful=stateful, name='dec_lstm_final')(x)

    # attempt to rebuild input
    outputs = []
    for output in model_output_reqs:
        outputs.append(layers.TimeDistributed(layers.Dense(output.dim, activation=output.activation, name=output.name + output.activation), name=output.name + '_out')(x))

    return outputs, ar_inputs

def create_LSTMdecoder_graph_ar_explicit(z, model_output_reqs, seq_length=seq_length, ar_inputs=None,
                    lstm_layers = 2, hidden_state = 256, dense_size = 256, n_notes=88, chroma=False, recurrent_dropout = 0.0,
                    stateful=False):
    """creates an autoregressive LSTM based decoder, with an explicit for loop for LSTM operations

    
    Arguments:
    seq_length -- time steps per training example
    hidden_state -- size of LSTM hidden state
    ar_inputs -- either None, in which case all outputs are fed back in auto regressively, or a list of input names
    supplemental_inputs -- list ints, where each int is the dimension of an input. These inputs will be converted from int to one hot.

    Notes:
    I suppose all that would be needed for prediction, would be to use sequence length of 1, and use predictions as feed in for tf inputs?
    
    """
    if ar_inputs == None:
        ar_inputs = [model_output.name for model_output in model_output_reqs]
    else:
        assert set(ar_inputs) <= set([model_output.name for model_output in model_output_reqs]), f'ar_inputs contains invalid output names: {ar_inputs}'

    x = layers.Dense(dense_size, activation='relu', name='initial_dense')(z)

    # get teacher forced inputs
    # the model will only be stateful if it is being used for prediction
    if not stateful:
        ar_inputs = [tf.keras.Input((seq_length,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True and model_output.name in ar_inputs]
    else:
        # if this executes, then the model is being used for prediction, and batch size is 1
        ar_inputs = [tf.keras.Input(batch_shape=(1,seq_length,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True and model_output.name in ar_inputs]
    
    ar_concat = layers.concatenate(ar_inputs, axis=-1)

    # make final dense layers
    output_fns = [layers.Dense(output.dim, activation=output.activation, name=output.name + output.activation) for output in model_output_reqs]

    LSTMs = []
    for i in range(lstm_layers - 1):
        LSTMs.append(layers.LSTM(hidden_state, return_sequences=True, recurrent_dropout=recurrent_dropout, stateful=stateful, name=f'dec_lstm_{i}'))
    final_LSTM = layers.LSTM(hidden_state, return_sequences=True, recurrent_dropout=recurrent_dropout, stateful=stateful, name='dec_lstm_final')
    
    #layer for concatenating x2 with ar_inputs 
    x2_ar_concat = layers.concatenate

    outputs = [[] for i in range(len(output_fns))]

    expand_dims = layers.Lambda(lambda x: tf.expand_dims(x, 1))
    for i in range(seq_length):
        # need to retain x for next time step
        x2 = x
        x2 = expand_dims(x2)
        # pass dense outputs through LSTMs
        for lstm in LSTMs:
            x2 = lstm(x2)
        # get ar_inputs for this time step
        ar_slice = layers.Lambda(lambda x: x[...,i,:], name=f'select_ar_{i + 1}')(ar_concat)
        ar_slice = expand_dims(ar_slice)
        x2 = x2_ar_concat([x2, ar_slice])
        for j in range(len(output_fns)):
            outputs[j].append(output_fns[j](x2))
    
    final_concat = layers.concatenate
    outputs = [final_concat(unconcat_out, axis=-2, name=raw_out.name + '_out') for unconcat_out, raw_out in zip(outputs, model_output_reqs)]
    
    return outputs, ar_inputs


def create_LSTMdecoder_pred(latent_vector, model_output_reqs, seq_length=seq_length,
                    lstm_layers = 3, dense_layers = 2, hidden_state = 256, dense_size = 256, n_notes=88, chroma=False, recurrent_dropout = 0.0):
    """creates an LSTM based decoder
    
    Arguments:
    seq_length -- time steps per training example
    hidden_state -- size of LSTM hidden state
    supplemental_inputs -- list ints, where each int is the dimension of an input. These inputs will be converted from int to one hot.
    
    """

    x = layers.Dense(dense_size, activation='relu')(latent_vector)
    x = layers.RepeatVector(seq_length)(x)
    
    # get teacher forced input - only H, this time
    ar_inputs = [tf.keras.Input((seq_length,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True and model_output.name == 'H']

    
    # pass input through non final lstm layers, returning sequences each time
    for i in range(lstm_layers - 1):
        x = layers.Bidirectional(layers.LSTM(hidden_state, return_sequences=True, recurrent_dropout=recurrent_dropout, name=f'dec_lstm_{i}'))(x)
    
    # concat teacher forced inputs with x
    # pass through final lstm layer
    x = layers.concatenate(ar_inputs + [x], axis=-1)
    x = layers.LSTM(hidden_state, return_sequences=True, recurrent_dropout=recurrent_dropout, name='dec_lstm_final')(x)



    # attempt to rebuild input
    outputs = []
    for i in range(seq_length):
        pass

    for output in model_output_reqs:
        outputs.append(layers.TimeDistributed(layers.Dense(output.dim, activation=output.activation, name=output.activation), name=output.name + '_out')(x))

    return outputs, ar_inputs


def create_hierarchical_decoder_graph(
                        # basic data parameters
                        z,
                        model_output_reqs,
                        seq_length=seq_length,
                        ar_inputs=None, # None => all 
                        # dense and lstm sizes
                        dense_size=256,
                        hidden_state=256,
                        decoder_lstms=2,
                        conductor_state_size=None, # None => same as decoder
                        # conductor configuration
                        conductors=2, # no. of lstm layers for conductor 
                        conductor_steps=16, # length of conductor outputs in time
                        initial_state_from_dense=True, # initial conductor and decoder activations from dense layer
                        initial_state_activation=None, # activation for aforementioned dense layer
                        recurrent_dropout=0.0,
                        stateful=False, # use True to make a prediction model
                        prediction_model=False,
                        batch_size=None,
                        ar_inc_batch_shape=False, #sometimes needed to make shapes mesh, in TF 2.0.0
                        **kwargs):
    """create a hierarchical decoder

    Arguments:
    z -- latent vector
    model_output_reqs -- list of named tuples, each containing information about the outputs required
    ar_inputs -- which inputs to pass back in autoregressively. None indicates all.

    Returns:
    outputs -- list of outputs, used for compiling a model
    ar_inputs -- keras inputs representing auto-regressive inputs - outputs moved one step to the right.
    decoder -- IF stateful is true, outputs and ar_inputs will be returned for compiling a model that ends
                with conductor outputs and initial states for decoder lstms.
                In addition, a separate stateful decoder model with seq_length of 1 will be returned.

    Notes:
    See https://nips2017creativity.github.io/doc/Hierarchical_Variational_Autoencoders_for_Music.pdf
    for inspiration on architecture.

    """
    print('z type:', type(z))
    print(z)
    # check autoregressive inputs - if a list hasn't been received, then ALL outputs will be passed in autoregressively
    if ar_inputs == None:
        ar_inputs = [model_output.name for model_output in model_output_reqs]
    else:
        assert set(ar_inputs) <= set([model_output.name for model_output in model_output_reqs]), f'ar_inputs contains invalid output names: {ar_inputs}'
    
    if conductor_state_size == None:
        conductor_state_size = hidden_state

    # calculate conductor substeps ('sub beats')
    conductor_substeps = int(seq_length / conductor_steps)
    print('conductor substeps:', conductor_substeps)

    ### conductor operations ###

    # the first layer conductors have no input! We need some dummy input.
    # dummy input will control the number of conductor time steps
    dummy_in = tf.keras.Input(shape=[0], name='dummy')
    repeat_dummy = layers.RepeatVector(conductor_steps, name='dummy_repeater')(dummy_in)

    # give to conductors to get sequence 'conductor_out'
    if initial_state_from_dense:
        # get the conductor initial state by passing z through dense layers
        h0 = layers.Dense(conductor_state_size, activation=initial_state_activation, name='conductor_lstm_h0_0')(z)
        c0 = layers.Dense(conductor_state_size, activation=initial_state_activation, name='conductor_lstm_c0_0')(z)
        # fire up conductor, getting 'c', the hidden states for the start of each decoder step
        conductor_out = layers.LSTM(conductor_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout, name=f'conductor_lstm_0')(repeat_dummy, initial_state=[h0, c0])
        for i in range(conductors - 1):
            h0 = layers.Dense(conductor_state_size, activation=initial_state_activation, name=f'conductor_lstm_h0_{i+1}')(z)
            c0 = layers.Dense(conductor_state_size, activation=initial_state_activation, name=f'conductor_lstm_c0_{i+1}')(z)
            conductor_out = layers.LSTM(conductor_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout, name=f'conductor_lstm_{i+1}')(conductor_out, initial_state=[h0, c0])

    else:
        conductor_out = layers.LSTM(conductor_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout)(repeat_dummy)
        for i in range(conductors - 1):
            conductor_out = layers.LSTM(conductor_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout)(conductor_out)
    
    # dense layers for decoder initial states
    dense_decode_h0s = [layers.Dense(hidden_state, activation=initial_state_activation, name=f'conductor_LSTM_h0_{i}') for i in range(decoder_lstms)]
    dense_decode_c0s = [layers.Dense(hidden_state, activation=initial_state_activation, name=f'conductor_LSTM_c0_{i}') for i in range(decoder_lstms)]


    ### set up layers for final decoding ###
    # concatenation layer for joining c repeated with ar input
    c_ar_concat = layers.concatenate
    
    # set up decoder lstms
    decoder_lstm_layers = []
    for i in range(decoder_lstms):
        decoder_lstm_layers.append(layers.LSTM(hidden_state, return_sequences=True, recurrent_dropout=recurrent_dropout, stateful=stateful, name=f'final_dec_LSTM_{i}'))

    # set up dense layers for final output
    output_fns = [layers.TimeDistributed(layers.Dense(output.dim, activation=output.activation, name=output.name + output.activation), name=output.name + '_unconcat') for output in model_output_reqs]


    if stateful:
        # if stateful, then this is a prediction model! 
        # that means autoregressive inputs can't be all passed in at once, but need to be sorted step by step.
        # return conductor out, decoder initial states out, and dummy in for building autoencoder/decoder model up to conductor outs
        # also return the decoder part, which will need to be manually used to get outputs

        ### initial states for decoder - completes conductor model.
        decoder_initial_h0s = [get_h0(conductor_out) for get_h0 in dense_decode_h0s]
        decoder_initial_c0s = [get_c0(conductor_out) for get_c0 in dense_decode_c0s]
        initial_states = [initial for pair in zip(decoder_initial_h0s, decoder_initial_c0s) for initial in pair]

        # conductor input
        c_input = tf.keras.Input(batch_shape=(1,1,conductor_state_size), name='c_in')
        # ar inputs
        if ar_inc_batch_shape:
            ar_inputs = [tf.keras.Input(batch_shape=(batch_size,1,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True]
        else:
            ar_inputs = [tf.keras.Input((1,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True]
        # must sort them so that this is consistent with trained model
        ar_inputs.sort(key=lambda x: x.name)
        ar_concat = layers.concatenate(ar_inputs, axis=-1)

        # concatenate c_input with ar input
        c_ar_concat = layers.concatenate([c_input, ar_concat])

        # Perform one step of decoder
        # initial states will need to be set OUTSIDE this model, in prediction code
        x = decoder_lstm_layers[0](c_ar_concat)
        for i in range(1, decoder_lstms):
            x = decoder_lstm_layers[i](x)
        # expand dimensions so that time distributed layer doesn't have to be removed from final dense layers
        x = layers.Lambda(lambda x: tf.expand_dims(x, 0), name='expand_dims')(x)

        # outputs doesn't need sublists - because we're only doing one step
        outputs = []
        for i in range(len(output_fns)):
            outputs.append(output_fns[i](x))

        decoder = tf.keras.Model(inputs=[c_input] + ar_inputs, outputs=outputs, name=f'decoder')
        return [conductor_out] + initial_states, dummy_in, decoder


    # all decoder (sub step) LSTM operations for step i have c_i passed to them 
    c_repeater = layers.RepeatVector(conductor_substeps)

    # list for appending outputs at each sub step, one sub list for each output
    outputs = [[] for i in range(len(output_fns))]

    # need teacher forced inputs (sequential outputs moved right by a sub step)
    if ar_inc_batch_shape:
        ar_inputs = [tf.keras.Input((seq_length,model_output.dim), batch_size=batch_size, name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True]
    else:
        ar_inputs = [tf.keras.Input((seq_length,model_output.dim), batch_size=None, name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True]
    ar_inputs.sort(key=lambda x: x.name)
    # concatenate teacher forced
    ar_concat = layers.concatenate(ar_inputs, axis=-1)

    # concatenation layer for joining c repeated with ar input
    c_ar_concat = layers.concatenate

    ### decoder operations ###
    assert conductor_steps == conductor_out.shape[1]
    for i in range(conductor_steps):
        # get the c_i for the relevant substep
        c_i = layers.Lambda(lambda x: x[:,i,:], name=f'select_c_{i + 1}')(conductor_out)
        # if this is a prediction model, then we need to do all the steps manually, so that t_n can be fed into t_n+1
        if prediction_model:
            for j in range(conductor_substeps):
                pass
            

        # repeat for each substep
        c_repeated = c_repeater(c_i)

        # append target outputs shifted to the right be a timestep
        ar_slice = layers.Lambda(lambda x: x[...,i*conductor_substeps:(i+1)*conductor_substeps,:], name=f'select_ar_{i + 1}')(ar_concat)

        # join c repeated with teach forced input
        c_ar_step = c_ar_concat([c_repeated, ar_slice], axis=-1)

        h0 = dense_decode_h0s[0](c_i)
        c0 = dense_decode_c0s[0](c_i)

        # at this point for training, I need to have targets for previous timesteps appended to c_repeated, for teacher forcing.
        # repeated c is given as input, and c as the initial state
        x = decoder_lstm_layers[0](c_ar_step, initial_state=[h0,c0])
        for j in range(1, decoder_lstms):
            h0 = dense_decode_h0s[j](c_i)
            c0 = dense_decode_c0s[j](c_i)
            x = decoder_lstm_layers[j](x, initial_state=[h0,c0])
        for j in range(len(output_fns)):
            outputs[j].append(output_fns[j](x))

    final_concat = layers.concatenate
    outputs = [final_concat(unconcat_out, axis=-2, name=raw_out.name + '_out') for unconcat_out, raw_out in zip(outputs, model_output_reqs)]
    
    return outputs, ar_inputs + [dummy_in]


def create_hierarchical_decoder_graph2(z, model_output_reqs, seq_length=seq_length, dense_size=256, hidden_state=256, conductor_state_size=None,
                conductors=2, conductor_steps=8, recurrent_dropout=0.0, initial_state_from_dense=True, ar_inputs=None, stateful=False):
    """create a hierarchical decoder

    Arguments:
    z -- latent vector
    model_output_reqs -- list of named tuples, each containing information about the outputs required

    Returns:
    outputs -- list of outputs, used for compiling a model
    ar_inputs -- teacher forced inputs - that is, outputs moved one step to the right


    Notes:
    still need to sort out initial states...

    """

    
    if conductor_state_size == None:
        conductor_state_size = hidden_state

    # calculate conductor substeps ('sub beats')
    conductor_substeps = int(seq_length / conductor_steps)
    print('conductor substeps:', conductor_substeps)

    ### conductor operations ###

    # the first layer conductors have no input! We need some dummy input.
    # dummy input will control the number of conductor time steps
    dummy_in = tf.keras.Input(shape=[0], name='dummy')
    repeat_dummy = layers.RepeatVector(conductor_steps, name='dummy_repeater')(dummy_in)

    # give to conductors to get sequence 'conductor_out'
    if initial_state_from_dense:
        # get the conductor initial state by passing z through dense layers
        h0 = layers.Dense(conductor_state_size, activation='tanh')(z)
        c0 = layers.Dense(conductor_state_size, activation='tanh')(z)
        
        # fire up conductor, getting 'c', the hidden states for the start of each decoder step
        conductor_out = layers.LSTM(conductor_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout)(repeat_dummy, initial_state=[h0, c0])
    
    else:
        conductor_out = layers.LSTM(conductor_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout)(repeat_dummy)

    ### set up layers for final decoding ###
    # concatenation layer for joining c repeated with ar input
    c_ar_concat = layers.concatenate

    # set up decoder lstm
    if stateful:
        seq_length
    decoder_lstm = layers.LSTM(hidden_state, return_sequences=True, recurrent_dropout=recurrent_dropout, name='final_dec_LSTM')

    # set up dense layers for final output
    output_fns = [layers.TimeDistributed(layers.Dense(output.dim, activation=output.activation, name=output.name + output.activation), name=output.name + '_unconcat') for output in model_output_reqs]

    # all decoder (sub step) LSTM operations for step i have c_i passed to them 
    c_repeater = layers.RepeatVector(conductor_substeps)

    # list for appending outputs at each sub step, one sub list for each output
    outputs = [[] for i in range(len(output_fns))]

    # check autoregressive inputs - if a list hasn't been received, then ALL outputs will be passed in autoregressively
    if ar_inputs == None:
        ar_inputs = [model_output.name for model_output in model_output_reqs]
    else:
        assert set(ar_inputs) <= set([model_output.name for model_output in model_output_reqs]), f'ar_inputs contains invalid output names: {ar_inputs}'
    
    # need teacher forced inputs (sequential outputs moved right by a sub step)
    ar_inputs = [tf.keras.Input((seq_length,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True]
    ar_inputs.sort(key=lambda x: x.name)
    # concatenate teacher forced
    ar_concat = layers.concatenate(ar_inputs, axis=-1)

    # concatenation layer for joining c repeated with ar input
    c_ar_concat = layers.concatenate

    ### decoder operations ###

    for i in range(conductor_out.shape[1]):
        # get the c_i for the relevant substep
        c = layers.Lambda(lambda x: x[:,i,:], name=f'select_c_{i + 1}')(conductor_out)
        # repeat for each substep
        c_repeated = c_repeater(c)

        # append target outputs shifted to the right be a timestep
        ar_slice = layers.Lambda(lambda x: x[...,i*conductor_substeps:(i+1)*conductor_substeps,:], name=f'select_ar_{i + 1}')(ar_concat)

        # join c repeated with teach forced input
        c_ar_step = c_ar_concat([c_repeated, ar_slice])

        # at this point for training, I need to have targets for previous timesteps appended to c_repeated, for teacher forcing.
        # repeated c is given as input, and c as the initial state
        x = decoder_lstm(c_ar_step, initial_state=[c,c])
        for i in range(len(output_fns)):
            outputs[i].append(output_fns[i](x))

    final_concat = layers.concatenate
    outputs = [final_concat(unconcat_out, axis=-2, name=raw_out.name + '_out') for unconcat_out, raw_out in zip(outputs, model_output_reqs)]
    
    return outputs, ar_inputs + [dummy_in]


def pred_from_h_decoder(conductor_out, decoder, model_output_reqs, model_datas, seq_length=seq_length, conductor_steps=8, initial_state_from_dense=True, ar_inputs=None, stateful=False):
    """create a hierarchical decoder

    Arguments:
    z -- latent vector
    model_output_reqs -- list of named tuples, each containing information about the outputs required

    Returns:
    outputs -- list of outputs, used for compiling a model
    ar_inputs -- teacher forced inputs - that is, outputs moved one step to the right


    Notes:
    still need to sort out initial states...

    """

    # calculate conductor substeps ('sub beats')
    conductor_substeps = int(seq_length / conductor_steps)
    print('conductor substeps:', conductor_substeps)
        

    # check autoregressive inputs - if a list hasn't been received, then ALL outputs will be passed in autoregressively
    if ar_inputs == None:
        ar_inputs = [model_output.name for model_output in model_output_reqs]
    else:
        assert set(ar_inputs) <= set([model_output.name for model_output in model_output_reqs]), f'ar_inputs contains invalid output names: {ar_inputs}'

    # list for appending outputs at each sub step, one sub list for each output
    outputs = [[] for i in range(len(output_fns))]

    ### decoder operations ###
    # conductor_out.shape[1] should be same as number of conductor_steps
    ar_slice = np.zeros((1,1,1,1))
    for conductor_step in conductor_out[0]:
        for i in range(conductor_substeps):
            conductor_step
            # join c repeated with teach forced input
            c_ar_step = np.concatenate([conductor_step, ar_slice])

            # at this point for training, I need to have targets for previous timesteps appended to c_repeated, for teacher forcing.
            # repeated c is given as input, and c as the initial state
            if i == 0:
                tf.keras.backend.set_value(decoder.get_layer('final_dec_LSTM').states[0], conductor_step)
                tf.keras.backend.set_value(decoder.get_layer('final_dec_LSTM').states[1], conductor_step)
            x = decoder.predict(c_ar_step)
            for i in range(len(output_fns)):
                outputs[i].append(output_fns[i](x))
        ### need to do ar stuff... Including sorting out H

    outputs = [np.concatenate(outputs, axis=-2) for outputs, raw_out in zip(outputs, model_output_reqs)]
    
    return outputs



######### nbq models #############

def create_nbq_bi_graph(model_input_reqs,
                        model_output_reqs,
                        seq_length = seq_length,  
                        bi_lstm_layers = 3,  
                        hidden_state = 512,  
                        recurrent_dropout = 0.0,
                        
                        ar_inputs=None,
                        uni_lstm_layers = 2,
                        dense_size = 256,
                        stateful=False,
                        batch_size=None,
                        ar_inc_batch_shape=False,
                        **kwargs):
    """layers for LSTM encoder, returning latent vector as output
    
    Arguments:
    seq_inputs -- inputs that are sequential in nature
    aux_inputs -- other inputs
    seq_length -- time steps per training example
    hidden_state -- size of LSTM hidden state
    supplemental_inputs -- list ints, where each int is the dimension of an input. These inputs will be converted from int to one hot.
    ar_inputs -- either None, in which case all outputs are fed back in auto regressively, or a list of input names

    """
    ### inputs
    # pass input through non final lstm layers, returning sequences each time
    seq_inputs, aux_inputs = get_inputs(model_input_reqs, seq_length)

    # inputs that aren't sequential will be repeated for each timestep
    repeated_inputs = []
    if len(aux_inputs) > 0:
        # layer for repeating aux inputs
        repeat_aux_inputs = layers.RepeatVector(seq_length, name=f'repeat{seq_length}Times')
        repeated_inputs = [repeat_aux_inputs(aux_input) for aux_input in aux_inputs]
    
    # concatenate repeated inputs with sequential inputs
    if len(seq_inputs) + len(repeated_inputs) > 1:
        x = layers.concatenate(seq_inputs + repeated_inputs, name='joinModelInput')
    else:
        x = seq_inputs[0]

    # pass through bidirectional lstms
    for i in range(bi_lstm_layers - 1):
        x = layers.Bidirectional(layers.LSTM(hidden_state, return_sequences=True,
                                recurrent_dropout=recurrent_dropout, name=f'enc_lstm_{i}'), name=f'bi_enc_lstm_{i}')(x)
    x = layers.Bidirectional(layers.LSTM(hidden_state, return_sequences=True, recurrent_dropout=recurrent_dropout, name=f'enc_lstm_{bi_lstm_layers - 1}'), name=f'bi_enc_lstm_{bi_lstm_layers - 1}')(x)

    ## at this point, x is the bidirectionally encoded lstm sequence

    # sort out autoregressive inputs - if a list hasn't been received, then ALL outputs will be passed in autoregressively
    if ar_inputs == None:
        ar_inputs = [model_output.name for model_output in model_output_reqs]
    # the model will only be stateful if it is being used for prediction
    if not stateful:
        if ar_inc_batch_shape:
            ar_inputs = [tf.keras.Input((seq_length,model_output.dim), batch_size=batch_size, name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True and model_output.name in ar_inputs]
        else:
            ar_inputs = [tf.keras.Input((seq_length,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True and model_output.name in ar_inputs]
    else:
        # if this executes, then the model is being used for prediction, and batch size is 1
        ar_inputs = [tf.keras.Input(batch_shape=(1,seq_length,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True and model_output.name in ar_inputs]
    
    # concat teacher forced inputs with x
    x = layers.concatenate(ar_inputs + [x], axis=-1)
    # pass input through non final lstm layers, returning sequences each time
    for i in range(uni_lstm_layers - 1):
        x = layers.LSTM(hidden_state, return_sequences=True, recurrent_dropout=recurrent_dropout, stateful=stateful, name=f'uni_lstm_{i}')(x)
    
    # pass through final lstm layer
    x = layers.LSTM(hidden_state, return_sequences=True, recurrent_dropout=recurrent_dropout, stateful=stateful, name='uni_lstm_final')(x)

    # predict outputs
    outputs = []
    for output in model_output_reqs:
        outputs.append(layers.TimeDistributed(layers.Dense(output.dim, activation=output.activation, name=output.name + output.activation), name=output.name + '_out')(x))

    return ar_inputs + seq_inputs + aux_inputs, outputs


########## Plotting model output ##########

def plt_metric(history, metric='loss'):
    """plots metrics from the history of a model
    Arguments:
    history -- history of a keras model
    metric -- str, metric to be plotted
    
    """
    plt.plot(history[metric])
    plt.plot(history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')



