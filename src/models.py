import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

seq_length = 64
n_notes = 88

########## Basic Recurrent ##########

def get_model_reqs(model_inputs=['H', 'V_mean'], model_outputs=['V']):
    n_notes=88
    # I assume that data, aside from the sequential dimentsion, will never have more than 1 dimension
    model_input = namedtuple('input', 'name dim seq')
    model_output = namedtuple('output', 'name dim activation seq')

    # model input requirements
    model_input_reqs_unfiltered = [model_input('H', n_notes, True),
                                model_input('tempo', 1, False),
                                model_input('key', 12, False),
                                model_input('V_mean', 1, False)]

    # model output requirements
    model_output_reqs_unfiltered = [model_output('H', n_notes, 'sigmoid', True),
                                    model_output('O', n_notes, 'tanh', True),
                                    model_output('V', n_notes, 'sigmoid', True)]

    model_input_reqs = [m_input for m_input in model_input_reqs_unfiltered if m_input.name in model_inputs]
    model_output_reqs = [m_output for m_output in model_output_reqs_unfiltered if m_output.name in model_outputs]

    return model_input_reqs, model_output_reqs

def get_inputs(model_input_reqs, seq_length):
    seq_inputs = [tf.keras.Input(shape=(seq_length,seq_in.dim), name=seq_in.name + '_in') for seq_in in model_input_reqs if seq_in.seq == True]
    aux_inputs = [tf.keras.Input(shape=(aux_in.dim,), name=aux_in.name + '_in') for aux_in in model_input_reqs if aux_in.seq == False]
    return seq_inputs, aux_inputs


def create_simple_LSTM_RNN(model_input_reqs, model_output_reqs, seq_length=seq_length, lstm_layers=3, dense_layers=2, LSTM_state_size = 256,
                    dense_size = 128, recurrent_dropout = 0.0):
    
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

def create_LSTMencoder_graph(model_input_reqs, seq_length = seq_length, lstm_layers = 3, dense_layers = 2, hidden_state_size = 512, latent_size = 256,
                    dense_size = 512, recurrent_dropout = 0.0):
    """layers for LSTM encoder, returning latent vector as output
    
    Arguments:
    seq_inputs -- inputs that are sequential in nature
    aux_inputs -- other inputs
    seq_length -- time steps per training example
    hidden_state_size -- size of LSTM hidden state
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
        x = layers.Bidirectional(layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout, name=f'enc_lstm_{i}'))(x)
    # pass through final lstm layer
    x = layers.Bidirectional(layers.LSTM(hidden_state_size, return_sequences=False, recurrent_dropout=recurrent_dropout, name='enc_lstm_final'))(x)

    # pass through non final dense layers
    for i in range(dense_layers - 1):
        x = layers.Dense(dense_size, activation='relu', name=f'enc_dense_{i}')(x)
    
    z = layers.Dense(latent_size, activation='relu', name='z')(x)
    
    return z, seq_inputs + aux_inputs

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
                    lstm_layers = 2, dense_layers = 2, hidden_state_size = 512, dense_size = 512, n_notes=88, chroma=False, recurrent_dropout = 0.0):
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


def create_LSTMdecoder_graph_ar(latent_vector, model_output_reqs, seq_length=seq_length,
                    lstm_layers = 2, hidden_state_size = 256, dense_size = 256, n_notes=88, chroma=False, recurrent_dropout = 0.0, stateful=False):
    """creates an autoregressive LSTM based decoder

    
    Arguments:
    seq_length -- time steps per training example
    hidden_state_size -- size of LSTM hidden state
    supplemental_inputs -- list ints, where each int is the dimension of an input. These inputs will be converted from int to one hot.

    Notes:
    I suppose all that would be needed for prediction, would be to use sequence length of 1, and use predictions as feed in for tf inputs?
    
    """

    x = layers.Dense(dense_size, activation='relu', name='initial_dense')(latent_vector)
    # if not stateful:
    x = layers.RepeatVector(seq_length)(x)
    
    # get teacher forced inputs
    # the model will only be stateful if it is being used for prediction
    if not stateful:
        ar_inputs = [tf.keras.Input((seq_length,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True]
    else:
        # if this executes, then the model is being used for prediction, and batch size is 1
        ar_inputs = [tf.keras.Input(batch_shape=(1,seq_length,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True]
    
    # pass input through non final lstm layers, returning sequences each time
    for i in range(lstm_layers - 1):
        x = layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout, stateful=stateful, name=f'dec_lstm_{i}')(x)
    
    # concat teacher forced inputs with x
    # pass through final lstm layer
    x = layers.concatenate(ar_inputs + [x], axis=-1)
    x = layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout, stateful=stateful, name='dec_lstm_final')(x)

    # attempt to rebuild input
    outputs = []
    for output in model_output_reqs:
        outputs.append(layers.TimeDistributed(layers.Dense(output.dim, activation=output.activation, name=output.activation), name=output.name + '_out')(x))

    return outputs, ar_inputs


def create_LSTMdecoder_pred(latent_vector, model_output_reqs, seq_length=seq_length,
                    lstm_layers = 3, dense_layers = 2, hidden_state_size = 256, dense_size = 256, n_notes=88, chroma=False, recurrent_dropout = 0.0):
    """creates an LSTM based decoder
    
    Arguments:
    seq_length -- time steps per training example
    hidden_state_size -- size of LSTM hidden state
    supplemental_inputs -- list ints, where each int is the dimension of an input. These inputs will be converted from int to one hot.
    
    """

    x = layers.Dense(dense_size, activation='relu')(latent_vector)
    x = layers.RepeatVector(seq_length)(x)
    
    # get teacher forced input - only H, this time
    ar_inputs = [tf.keras.Input((seq_length,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True and model_output.name == 'H']

    
    # pass input through non final lstm layers, returning sequences each time
    for i in range(lstm_layers - 1):
        x = layers.Bidirectional(layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout, name=f'dec_lstm_{i}'))(x)
    
    # concat teacher forced inputs with x
    # pass through final lstm layer
    x = layers.concatenate(ar_inputs + [x], axis=-1)
    x = layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout, name='dec_lstm_final')(x)



    # attempt to rebuild input
    outputs = []
    for i in range(seq_length):
        pass

    for output in model_output_reqs:
        outputs.append(layers.TimeDistributed(layers.Dense(output.dim, activation=output.activation, name=output.activation), name=output.name + '_out')(x))

    return outputs, ar_inputs


def create_hierarchical_decoder_graph(z, model_output_reqs, seq_length=seq_length, dense_size=256, hidden_state_size=256, conductor_state_size=None,
                conductors=2, conductor_steps=8, recurrent_dropout=0.0, initial_state_from_dense=True):
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
        conductor_state_size = hidden_state_size

    # calculate conductor substeps ('sub beats')
    conductor_substeps = int(seq_length / conductor_steps)
    print('conductor substeps:', conductor_substeps)

    ### conductor operations ###

    # the first layer conductors have no input! We need some dummy input.
    # dummy input will control the number of conductor time steps
    dummy_in = tf.keras.Input(shape=[0], name='dummy')
    repeat_dummy = layers.RepeatVector(conductor_steps, name='dummy_repeater')(dummy_in)

    if initial_state_from_dense:
        # get the conductor initial state by passing z through dense layers
        h0 = layers.Dense(conductor_state_size, activation='tanh')(z)
        c0 = layers.Dense(conductor_state_size, activation='tanh')(z)
        
        # fire up conductor, getting 'c', the hidden states for the start of each decoder step
        all_c = layers.LSTM(conductor_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout)(repeat_dummy, initial_state=[h0, c0])
    
    else:
        all_c = layers.LSTM(conductor_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout)(repeat_dummy)

    ### set up layers for final decoding ###

    # set up decoder lstm
    decoder_lstm = layers.LSTM(conductor_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout)

    # set up dense layers for final output
    output_fns = [layers.TimeDistributed(layers.Dense(output.dim, activation=output.activation, name=output.activation), name=output.name + '_unconcat') for output in model_output_reqs]

    # all decoder (sub step) LSTM operations for step i have c_i passed to them 
    c_repeater = layers.RepeatVector(conductor_substeps)

    # list for appending outputs at each sub step, one sub list for each output
    outputs = [[] for i in range(len(output_fns))]

    # need teacher forced inputs (sequential outputs moved right by a sub step)
    ar_inputs = [tf.keras.Input((seq_length,model_output.dim), name=f'{model_output.name}_ar') for model_output in model_output_reqs if model_output.seq == True]
    ar_inputs.sort(key=lambda x: x.name)
    # concatenate teacher forced
    ar_concat = layers.concatenate(ar_inputs, axis=-1)

    # concatenation layer for joining c repeated with ar input
    c_ar_concat = layers.concatenate

    ### decoder operations ###

    for i in range(all_c.shape[1]):
        # get the c_i for the relevant substep
        c = layers.Lambda(lambda x: x[:,i,:], name=f'select_c_{i + 1}')(all_c)
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



