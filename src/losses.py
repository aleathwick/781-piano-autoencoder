import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LambdaCallback
from src.models import sampling
import src.midi_utils as midi_utils
import tensorflow_probability as tfp
ds = tfp.distributions


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

def note_mse(y_true, y_pred):
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
    return tf.keras.losses.mse(y_true_notes, y_pred_notes)

# for custom loss: looks like extra variables can be reffered by using custom loss wrapper?
# i.e. function within function, with outer function taking VOI as input, then passing to inner function
# look here for how MusicVAE does kl loss: https://github.com/tensorflow/magenta/blob/master/magenta/models/music_vae/base_model.py
def vae_custom_loss(z, beta_fn, free_bits=0, kl_weight=1, run=None):
    """Function for getting function to calculate kl loss + xent loss
    
    Arguments:
    z -- list containing [z_mean, z_log_sigma]
    beta_update -- function of epochs that returns beta, for updating beta in keras callback, OR an int/float
    free_bits -- alowance of free bits before kl loss starts impacting loss
    kl_weight -- weight to give to kl part of loss
    
    Returns:
    vae_loss -- function for evaluating loss
    beta_cb -- keras callback for updating beta

    Notes:
    Made with wonderful help of this: https://blog.keras.io/building-autoencoders-in-keras.html
    I had trouble with this function - I was using it with a sampling function that produced z_mean
    and z_LOG_sigma, rather than z_sigma, and no activation on those in the dense layers that produced them.


    """
    
    # for implementation of free_nats, see https://github.com/tensorflow/magenta/blob/master/magenta/models/music_vae/base_model.py
    free_nats = free_bits * tf.math.log(2.0)
    # for implementation of weighting kl loss according to epoch, see https://stackoverflow.com/questions/42787181/variationnal-auto-encoder-implementing-warm-up-in-keras/
    # beta is used for this
    if isinstance(beta_fn, (int, float)):
        beta = tf.keras.backend.variable(value=beta_fn)
    else:
        beta = tf.keras.backend.variable(value=0.0)
    z_mean, z_log_sigma = z
    # see https://stackoverflow.com/questions/42787181/variationnal-auto-encoder-implementing-warm-up-in-keras/
    
    beta_cb = None

    if not isinstance(beta_fn, (int, float)):
        def update_beta(epoch):
            """function for updating beta value according to epoch number"""
            value = beta_fn(epoch)
            print("beta:", value)
            K.set_value(beta, value)
            if run != None:
                run.log_scalar("beta", float(value))

        beta_cb = LambdaCallback(on_epoch_begin=lambda epoch, log: update_beta(epoch))

    def vae_loss(y_true, y_pred):
        xent_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        kl_loss = tf.maximum(kl_weight * (- 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)) - free_nats, 0)
        # need this for training on batches, see here: https://github.com/keras-team/keras/issues/10155
        kl_loss = K.mean(kl_loss)
        return beta * kl_loss + xent_loss
    return vae_loss, beta_cb

    ### 



def vae_custom_loss2(z, free_bits=0, kl_weight=1):
    """Function for getting function to calculate kl loss + xent loss
    
    Arguments:
    z -- list containing [z_mean, z_log_sigma]
    free_bits -- alowance of free bits before kl loss starts impacting loss
    kl_weight -- weight to give to kl part of loss
    
    Returns:
    function for evaluating loss

    Notes:

    """
    # for implementation of free_nats, see https://github.com/tensorflow/magenta/blob/master/magenta/models/music_vae/base_model.py
    free_nats = free_bits * tf.math.log(2.0)
    z_mean, z_sigma = z
    latent_size = z_mean.shape[-1]

    def vae_loss(y_true, y_pred):

        xent_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        q_z = ds.MultivariateNormalDiag(
            loc=z_mean, scale_diag=z_sigma)

        # Prior distribution.
        p_z = ds.MultivariateNormalDiag(
            loc=[0.] * latent_size, scale_diag=[1.] * latent_size)

        # KL Divergence (nats)
        kl_div = ds.kl_divergence(q_z, p_z)
        return kl_div + xent_loss
    return vae_loss