import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

seq_length = 64
n_notes = 88


def create_inputs(n_notes=n_notes, seq_length = seq_length, aux_inputs_dict = {'tempo': 1, 'key': 12}, int_input=False):
    """create list of inputs (for compiling model) and input for first layer of encoder, if needed converting inputs to OHE
    
    Arguments:
    n_notes -- range of valid notes
    seq_length = number of timesteps in input
    seq_length -- timesteps in input
    aux_inputs_dict -- dictionary mapping auxillary input names to dimensions
    int_input -- bool, determining whether or not auxillary inputs will be given as vectors or integers

    Returns:
    main_input
    aux_inputs -- any auxillary inputs, which haven't been expanded to OHE yet
    aux_inputs_expanded -- any auxillary inputs, now expanded to OHE

    Notes:
    aux_inputs and aux_inputs_expanded may be the same, if the raw inputs to the model have already been processed so they are OHE,
    or if only only 1 dimensional inputs are given (on/off inputs, or continuous valued)
    
    """
    
    ### sort out inputs, including taking aux inputs to one hot if needed
    # add the main input
    main_input = tf.keras.Input(shape=(seq_length,n_notes), name='H')
    # list for aux inputs
    aux_inputs = []
    # print([f'{input_name}: {dim}' for input_name, dim in aux_inputs_dict.items()])
    
    if len(aux_inputs_dict) > 0:
        if int_input:
            aux_inputs_expanded = []
            # with int_input true, input shapes will be 1, but then need to be expanded to OHE (if input_dim > 1)
            aux_inputs = [tf.keras.Input(shape=(1,), name=input_name) for input_name, dim in aux_inputs_dict.items()]
            for aux_input in aux_inputs:
                # if the 'aux_input_dims' dimension of the input > 1, assume it needs to be reencoded as one hot
                # using .split('_')[0] is necessary, because keras will name layers 'name_you_wanted' + '_blah'
                dim = aux_inputs_dict[aux_input.name.split('_')[0]]
                if dim > 1:
                    #cast to int32, make OHE, and remove the aux dimension that results
                    aux_inputs_expanded.append(layers.Lambda(lambda x: tf.squeeze(tf.one_hot(tf.cast(x, dtype='int32'), dim), -2), name='makeOneHot')(aux_input))
                else:
                    aux_inputs_expanded.append(aux_input)
            raw_inputs = [main_input] + aux_inputs
            model_inputs = [main_input] + aux_inputs_expanded

            
        else:
            # with int_input false, input dimensions are the same as provided in aux_input_dims
            aux_inputs = [tf.keras.Input(shape=(dim,), name=input_name) for input_name, dim in aux_inputs_dict.items()]
            raw_inputs = [main_input] + aux_inputs
            model_inputs = raw_inputs

    return raw_inputs, model_inputs

def create_LSTMencoder(seq_inputs, aux_inputs=[], seq_length = seq_length, batch_size=128, lstm_layers = 3, dense_layers = 2, hidden_state_size = 256, latent_size = 256,
                    dense_size = 256, n_notes=88, aux_input_dims = [1], chroma=False, recurrent_dropout = 0.0):
    """layers for LSTM encoder, returning latent vector as output
    
    Arguments:
    seq_inputs -- inputs that are sequential in nature
    aux_inputs -- other inputs
    seq_length -- time steps per training example
    hidden_state_size -- size of LSTM hidden state
    batch_size -- batch size
    supplemental_inputs -- list ints, where each int is the dimension of an input. These inputs will be converted from int to one hot.
    
    """
    # pass input through non final lstm layers, returning sequences each time
    repeated_inputs = []
    if len(aux_inputs) > 0:
        # layer for repeating aux inputs
        repeat_aux_inputs = layers.RepeatVector(seq_length, name=f'repeat{seq_length}Times')
        repeated_inputs = [repeat_aux_inputs(aux_input) for aux_input in aux_inputs]
    if len(seq_inputs) + len(repeated_inputs) > 1:
        x = layers.concatenate(seq_inputs + repeated_inputs, name='joinModelInput')
    else:
        x = seq_inputs[0]
    for _ in range(lstm_layers - 1):
        x = layers.Bidirectional(layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout))(x)
    # pass through final lstm layer
    x = layers.Bidirectional(layers.LSTM(hidden_state_size, return_sequences=False, recurrent_dropout=recurrent_dropout))(x)

    # pass through non final dense layers
    for _ in range(dense_layers - 1):
        x = layers.Dense(dense_size, activation='relu')(x)
    
    z = layers.Dense(latent_size, activation='relu', name='z')(x)
    
    return z

def create_conv_encoder(h, aux_inputs = [], latent_size = 64):
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

    return z
    

def create_LSTMdecoder(latent_vector, model_output_reqs, seq_length=seq_length,
                    lstm_layers = 3, dense_layers = 2, hidden_state_size = 256, dense_size = 256, n_notes=88, chroma=False, recurrent_dropout = 0.0):
    """creates an LSTM based decoder
    
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


def create_hierarchical_decoder(z, model_output_reqs, dummy_in, seq_length=seq_length, dense_size=256, conductor_state_size=32, decoder_state_size=256,
                conductors=2, conductor_steps=8, recurrent_dropout=0.0):
    """create a hierarchical decoder

    Arguments:
    z -- latent vector
    model_output_reqs -- list of named tuples, each containing information about the outputs required
    dummy_in -- dummy input used for inputless LSTMs

    Returns:
    outputs -- list of outputs, used for compiling a model
    tf_inputs -- teacher forced inputs - that is, outputs moved one step to the right


    Notes:
    still need to sort out initial states...

    """

    # calculate conductor substeps ('sub beats')
    conductor_substeps = int(seq_length / conductor_steps)
    print('conductor substeps:', conductor_substeps)

    ### conductor operations ###

    # the first layer conductors have no input! We need some dummy input.
    # dummy input will control the number of conductor time steps
    repeat_dummy = layers.RepeatVector(conductor_steps, name='dummy_repeater')(dummy_in)

    # get the conductor initial state by passing z through dense layers
    h0 = layers.Dense(conductor_state_size, activation='tanh')(z)
    c0 = layers.Dense(conductor_state_size, activation='tanh')(z)
    
    # fire up conductor, getting 'c', the hidden states for the start of each decoder step
    all_c = layers.LSTM(conductor_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout)(repeat_dummy, initial_state=[h0, c0])

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
    tf_inputs = [tf.keras.Input((seq_length,model_output.dim), name=f'{model_output.name}_tf') for model_output in model_output_reqs if model_output.seq == True]
    tf_inputs.sort(key=lambda x: x.name)
    # concatenate teacher forced
    tf_concat = layers.concatenate(tf_inputs, axis=-1)

    # concatenation layer for joining c repeated with tf input
    c_tf_concat = layers.concatenate

    ### decoder operations ###

    for i in range(all_c.shape[1]):
        # get the c_i for the relevant substep
        c = layers.Lambda(lambda x: x[:,i,:], name=f'select_c_{i + 1}')(all_c)
        # repeat for each substep
        c_repeated = c_repeater(c)

        # append target outputs shifted to the right be a timestep
        tf_slice = layers.Lambda(lambda x: x[...,i*conductor_substeps:(i+1)*conductor_substeps,:], name=f'select_tf_{i + 1}')(tf_concat)

        # join c repeated with teach forced input
        c_tf_step = c_tf_concat([c_repeated, tf_slice])

        # at this point for training, I need to have targets for previous timesteps appended to c_repeated, for teacher forcing.
        # repeated c is given as input, and c as the initial state
        x = decoder_lstm(c_tf_step, initial_state=[c,c])
        for i in range(len(output_fns)):
            outputs[i].append(output_fns[i](x))

    
    
    

    final_concat = layers.concatenate
    outputs = [final_concat(unconcat_out, axis=-2, name=raw_out.name + '_out') for unconcat_out, raw_out in zip(outputs, model_output_reqs)]
    
    return outputs, tf_inputs


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



