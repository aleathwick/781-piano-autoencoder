# utilities for running experiments

import os
import tensorflow as tf

def set_up_path(_run):
    no = _run._id
    path = f'experiments/run_{_run._id}/'
    os.mkdir(path)
    return no, path


# see https://github.com/pinae/Sacred-MNIST/blob/master/train_convnet.py for inspiration
class SacredLogMetrics(tf.keras.callbacks.Callback):
    """keras callback for logging metrics to sacred run, requires sacred _run to be passed in"""
    def __init__(self, _run):
        self._run = _run

    def log_performance(self, logs):
        self._run.log_scalar("loss", float(logs.get('loss')))
        self._run.log_scalar("accuracy", float(logs.get('accuracy')))
        self._run.log_scalar("val_loss", float(logs.get('val_loss')))
        self._run.log_scalar("val_accuracy", float(logs.get('val_accuracy')))
        self._run.result = float(logs.get('val_accuracy'))
    
    def on_epoch_end(self, _, logs={}):
        self.log_performance(logs=logs)


def capture_weights(_run):
    """captures weights from a run. assumes val and train weights stored in particular directory"""
    _run.add_artifact(f'experiments/run_{_run._id}/{_run._id}_best_train_weights.hdf5')
    _run.add_artifact(f'experiments/run_{_run._id}/{_run._id}_best_val_weights.hdf5')