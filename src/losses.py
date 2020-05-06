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

# for custom loss: looks like extra variables can be reffered by using custom loss wrapper?
# i.e. function within function, with outer function taking VOI as input, then passing to inner function
def vae_custom_loss(z, free_bits=0):
    # for implementation of free_nats, see https://github.com/tensorflow/magenta/blob/master/magenta/models/music_vae/base_model.py
    free_nats = free_bits * tf.math.log(2.0)
    z_mean, z_log_sigma = z
    def vae_loss(y_true, y_pred):
        xent_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        kl_loss = tf.maximum(- 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1) - free_nats, 0)
        # need this for training on batches, see here: https://github.com/keras-team/keras/issues/10155
        kl_loss = K.mean(xent_loss)
        return xent_loss + kl_loss
    return vae_loss