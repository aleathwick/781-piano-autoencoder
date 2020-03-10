import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


def create_LSTMencoder(seq_length = 32, batch_size=128, lstm_layers = 3, dense_layers = 2, hidden_state_size = 256, latent_size = 256,
                    dense_size = 256, n_notes=88, extra_input_dims = [1], chroma=False, recurrent_dropout = 0.0):
    """creates a simple model
    
    Arguments:
    seq_length -- time steps per training example
    hidden_state_size -- size of LSTM hidden state
    batch_size -- batch size
    supplemental_inputs -- list ints, where each int is the dimension of an input. These inputs will be converted from int to one hot.
    
    """

    ### sort out inputs, including taking extra inputs to one hot if needed
    main_input = tf.keras.Input(shape=(seq_length,n_notes))
    extra_inputs = []
    if len(extra_input_dims) > 0:
        extra_inputs = [tf.keras.Input(shape=(seq_length, 1)) for dim in extra_input_dims]
        inputs_expanded = [main_input]

        for dim, extra_input in zip(extra_input_dims, extra_inputs):
            # if the 'extra_input_dims' dimension of the input > 1, assume it needs to be reencoded as one hot
            if dim > 1:
                inputs_expanded.append(layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:], dtype='int32'), dim))(extra_input))
            else:
                inputs_expanded.append(extra_input)

        x = layers.concatenate(inputs_expanded)
    else:
        x = main_input


    # pass input through non final lstm layers, returning sequences each time
    for _ in range(lstm_layers - 1):
        x = layers.Bidirectional(layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout))(x)
    # pass through final lstm layer
    x = layers.Bidirectional(layers.LSTM(hidden_state_size, return_sequences=False, recurrent_dropout=recurrent_dropout))(x)

    # pass through non final dense layers
    for _ in range(dense_layers - 1):
        x = layers.Dense(dense_size, activation='relu')(x)
    
    z = layers.Dense(latent_size, activation='relu')(x)


    # model = tf.keras.Model(inputs=[main_input] + extra_inputs, outputs=z, name=f'LSTM Encoder')

    # model.summary()
    
    return ([main_input] + extra_inputs, z)


def create_LSTMdecoder(latent_vector, seq_length=32, latent_size = 256, batch_size=128, lstm_layers = 3, dense_layers = 2, hidden_state_size = 256,
                    dense_size = 256, n_notes=88, chroma=False, recurrent_dropout = 0.0):
    """creates a simple model
    
    Arguments:
    seq_length -- time steps per training example
    hidden_state_size -- size of LSTM hidden state
    batch_size -- batch size
    supplemental_inputs -- list ints, where each int is the dimension of an input. These inputs will be converted from int to one hot.
    
    """

    ### sort out inputs, including taking extra inputs to one hot if needed
    # main_input = tf.keras.Input(shape=(latent_size,))

    x = layers.Dense(dense_size, activation='relu')(latent_vector)
    x = layers.RepeatVector(seq_length)(x)

    # pass input through non final lstm layers, returning sequences each time
    for _ in range(lstm_layers - 1):
        x = layers.Bidirectional(layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout))(x)
    # pass through final lstm layer
    x = layers.LSTM(hidden_state_size, return_sequences=True, recurrent_dropout=recurrent_dropout)(x)
    
    # attempt to rebuild input
    H = layers.TimeDistributed(layers.Dense(n_notes, activation='sigmoid'))(x)
    # attempt to predict offsets and velocities
    O = layers.TimeDistributed(layers.Dense(n_notes, activation='tanh'))(x)
    V = layers.TimeDistributed(layers.Dense(n_notes, activation='sigmoid'))(x)

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



