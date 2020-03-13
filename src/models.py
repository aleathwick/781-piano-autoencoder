import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


def create_inputs(n_notes=88, seq_length = 32, aux_inputs_dict = {'tempo': 1, 'key': 12}, int_input=True):
    """create list of inputs (for compiling model) and input for first layer of encoder, if needed converting inputs to OHE
    
    Arguments:
    n_notes -- range of valid notes
    seq_length -- timesteps in input
    aux_input_dims -- dimensions of any auxillary inputs, such as key or tempo
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
    model_inputs = []
    # add the main input
    main_input = tf.keras.Input(shape=(seq_length,n_notes), name='mainInput')
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
            
        else:
            # with int_input false, input dimensions are the same as provided in aux_input_dims
            aux_inputs = [tf.keras.Input(shape=(seq_length, dim), name=input_name) for input_name, dim in aux_inputs_dict.items()]
    else:
        x = main_input
    
    raw_inputs = [main_input] + aux_inputs

    return main_input, aux_inputs, aux_inputs_expanded

def create_LSTMencoder(main_input, aux_inputs_expanded = [], seq_length = 32, batch_size=128, lstm_layers = 3, dense_layers = 2, hidden_state_size = 256, latent_size = 256,
                    dense_size = 256, n_notes=88, aux_input_dims = [1], chroma=False, recurrent_dropout = 0.0):
    """layers for LSTM encoder, returning latent vector as output
    
    Arguments:
    seq_length -- time steps per training example
    hidden_state_size -- size of LSTM hidden state
    batch_size -- batch size
    supplemental_inputs -- list ints, where each int is the dimension of an input. These inputs will be converted from int to one hot.
    
    """
    # pass input through non final lstm layers, returning sequences each time
    x = main_input
    if len(aux_inputs_expanded) != 0:
        # layer for repeating aux inputs
        repeat_aux_inputs = layers.RepeatVector(seq_length, name=f'repeat{seq_length}Times')
        aux_inputs_expanded = [repeat_aux_inputs(i) for i in aux_inputs_expanded]
        x = layers.concatenate([main_input] + aux_inputs_expanded, name='joinModelInput')
    for _ in range(lstm_layers - 1):
        x = layers.Bidirectional(layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout))(x)
    # pass through final lstm layer
    x = layers.Bidirectional(layers.LSTM(hidden_state_size, return_sequences=False, recurrent_dropout=recurrent_dropout))(x)

    # pass through non final dense layers
    for _ in range(dense_layers - 1):
        x = layers.Dense(dense_size, activation='relu')(x)
    
    z = layers.Dense(latent_size, activation='relu', name='z')(x)
    
    return z

def create_LSTMdecoder(latent_vector, seq_length=32, latent_size = 256, batch_size=128, lstm_layers = 3, dense_layers = 2, hidden_state_size = 256,
                    dense_size = 256, n_notes=88, chroma=False, recurrent_dropout = 0.0):
    """creates a simple model
    
    Arguments:
    seq_length -- time steps per training example
    hidden_state_size -- size of LSTM hidden state
    batch_size -- batch size
    supplemental_inputs -- list ints, where each int is the dimension of an input. These inputs will be converted from int to one hot.
    
    """

    ### sort out inputs, including taking aux inputs to one hot if needed
    # main_input = tf.keras.Input(shape=(latent_size,))

    x = layers.Dense(dense_size, activation='relu')(latent_vector)
    x = layers.RepeatVector(seq_length)(x)

    # pass input through non final lstm layers, returning sequences each time
    for _ in range(lstm_layers - 1):
        x = layers.Bidirectional(layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout))(x)
    # pass through final lstm layer
    x = layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout)(x)
    
    # attempt to rebuild input
    H = layers.TimeDistributed(layers.Dense(n_notes, activation='sigmoid', name='sigmoid'), name='H')(x)
    # attempt to predict offsets and velocities
    O = layers.TimeDistributed(layers.Dense(n_notes, activation='tanh', name='tanh'), name='O')(x)
    V = layers.TimeDistributed(layers.Dense(n_notes, activation='sigmoid', name='sigmoid'), name='V')(x)

    return H, O, V


def generate_ooremusic(model, num_generate=256, temperature=0.2, input_events=[34,0,0,3,3,16], chroma=False):
    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    # Number of notes to generate
    events_generated = []
    input_events = np.array(input_events)

    # Here batch size == 1
    model.reset_states()

    # prime the model with the input notes
    for i, input_event in enumerate(input_events[:-1]):
        events_generated.append(input_event)
        # I think I need to do this? batch size of 1...
        input_event = tf.expand_dims(input_event, 0)
        input_event = tf.expand_dims(input_event, 0)
        input_event = tf.expand_dims(input_event, 0)
        predictions = model(input_event)


    input_event = input_events[-1]
    input_event = np.array(input_event)
    input_event = tf.expand_dims(input_event, 0)
    input_event = tf.expand_dims(input_event, 0)
    input_event = tf.expand_dims(input_event, 0)
    for i in range(num_generate):
        prediction = model(input_event)

        # using a categorical distribution to predict the note returned by the model
        # have to do this for each output attribute of the note
            # remove the batch dimension
        prediction = tf.squeeze(prediction, 0)
        prediction = prediction / temperature
        predicted_id = tf.random.categorical(prediction, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_event = tf.expand_dims(tf.expand_dims(tf.expand_dims(predicted_id, 0), 0), 0)

        events_generated.append(predicted_id)

    return(events_generated)


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



