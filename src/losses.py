import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def note_categorical_crossentropy(y_true, y_pred):
    """calculates cc only for entries where a note exists
    
    Notes:
    If used with offsets, an offset COULD be zero! Which means the loss for that offset wouldn't be counted.
    Looking at a few files, out of 9693 notes, this was true for 20 of them.

    Results with this so far are poor - I suppose weights are updated irrespective of what they do
    to non note indices, which are most indices, leading to very unstable updates.
    
    """
    indices = tf.where(y_true != 0)
    y_true_notes = K.cast(tf.gather_nd(y_true, indices), dtype='float32')
    y_pred_notes = K.cast(tf.gather_nd(y_pred, indices), dtype='float32')
    return tf.keras.losses.categorical_crossentropy(y_true_notes, y_pred_notes)