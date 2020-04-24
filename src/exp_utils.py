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
    def __init__(self, run):
        self.run = run

    def log_performance(self, logs):
        self.run.log_scalar("loss", float(logs.get('loss')))
        self.run.log_scalar("accuracy", float(logs.get('accuracy')))
        self.run.log_scalar("val_loss", float(logs.get('val_loss')))
        self.run.log_scalar("val_accuracy", float(logs.get('val_accuracy')))
        self.run.result = float(logs.get('val_accuracy'))
    
    def on_epoch_end(self, _, logs={}):
        self.log_performance(logs=logs)

# another function, written by sacred developer:
#https://github.com/IDSIA/sacred/issues/110
class KerasInfoUpdater(tf.keras.callbacks.Callback):
    def __init__(self, run):
        super(KerasInfoUpdater, self).__init__()
        self.run = run
        # create a dictionary for logs, if not already in existence
        self.run.info['logs'] = self.run.info.get('logs', {})

    def on_epoch_end(self, epoch, logs={}):
        for k, v in logs.items():
            # get any values already assigned to k
            log_out = self.run.info['logs'].get(k, [])
            log_out.append(v)
            # add to run info
            self.run.info['logs'][k] = log_out
            # add to metrics (info is already stored, but makes for easy plotting in omniboard)
            self.run.log_scalar(k, float(v))


def capture_weights(_run):
    """captures weights from a run. assumes val and train weights stored in particular directory"""
    _run.add_artifact(f'experiments/run_{_run._id}/{_run._id}_best_train_weights.hdf5')
    _run.add_artifact(f'experiments/run_{_run._id}/{_run._id}_best_val_weights.hdf5')