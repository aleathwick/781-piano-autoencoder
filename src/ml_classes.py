import numpy as np
from scipy.sparse import csc_matrix
import src.midi_utils as midi_utils
import tensorflow as tf

 
class PianoPiece():
     def __init__(self, pm=None, data_dict=None):
        if pm != None:
            self.H, self.O, self.V = midi_utils.pm2HOV(pm)
        elif data_dict != None:
            self.H = data_dict('H')
        # and so on... probs won't use this 

class ModelData():
    def __init__(self, data, name, shape=(1,), sequential=True, is_input=True, transposable=True, sparse=True):
        """initializer

        Arguments:
        data -- list of training examples, each of which may be a list, np array, or scipy csc (sparse) array
        name -- name of input/output
        shape -- shape of the input, including sequence length
        sequential -- indicates whether the first number in shape refers to timesteps
        is_input -- bool, indicating whether data is an input or output
        transpose -- indicates whether or not to transpose the data
        sparse -- is the input sparse?

        """
        self.data = np.array(data)
        self.name = name
        self.is_input = is_input
        self.transposable = transposable
        self.is_csc = isinstance(data[0], csc_matrix)
        self.sparse = self.is_csc or sparse
        # shape should include sequence length (if part of input shape) + dimension
        self.shape = shape
        self.batch_data = None
    
    def data_generation(self, batch_size, indexes):
        batch_data = np.empty((batch_size) + self.shape)
        if self.is_csc:
            for i, index in enumerate(indexes):
                batch_data[i] = self.data[index].toarray()
        else:
            for i, index in enumerate(indexes):
                batch_data[i] = self.data[index]
        self.batch_data = batch_data
    
    def transpose(self, index, semitones):
        if not self.transposable:
            pass
        else:
            self.batch_data = np.concatenate((self.batch_data[...,-semitones:], self.batch_data[...,:-semitones]), axis=-1)
        


class MelDataGenerator(tf.keras.utils.Sequence):
    """NoteTuple data generator. Can transpose music on the fly, including chroma data."""
    def __init__(self, data, seq_length=64,  batch_size=64, shuffle=True, transpose_on = True, st = 4, epoch_per_dataset=1):
        """Initialization

        Arguments:
        data -- list of ModelDatas

        Notes:
        -check how you did transposition in your summer project... was it correct?

        """

        self.data = data
        self.input_data_batch = {}
        self.output_data_batch = {}
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        # if augment is true, data will be randomly transposed
        self.transpose_on = transpose_on
        self.on_epoch_end()
        # number of semitones to randomly transpose by
        self.st = st
        # this controls epoch
        self.epoch_per_dataset = epoch_per_dataset

    def __transpose(self):
        'Randomly transposes examples up or down by up to self.st semitones'
        for i in range(self.batch_size):
            semitones = np.random.randint(-self.st, (self.st + 1))
            for model_data in self.data:
                model_data.transpose(i, semitones)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int((len(self.input_data['H']) / self.epoch_per_dataset) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        self.__data_generation(indexes)

        # if augment is on, transpose
        if self.transpose_on:
            self.__transpose()

        return self.input_data_batch, self.output_data_batch

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, Tx)
        for model_data in self.data:
            model_data.data_generation(self.batch_size, indexes)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    

# very simple example of data generator
# taken from here: https://stackoverflow.com/questions/51057123/keras-one-hot-encoding-memory-management-best-possible-way-out
class MySequence(tf.keras.utils.Sequence): 
  def __init__(self, data, batch_size = 16):
    self.X = data[0]
    self.Y = data[1]
    self.batch_size = batch_size

  def __len__(self):
     return int(np.ceil(len(self.X) / float(self.batch_size)))

  def __getitem__(self, batch_id):
    # Get corresponding batch data...
    # one-hot encode
    return X, Y
