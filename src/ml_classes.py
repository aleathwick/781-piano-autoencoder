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
    def __init__(self, data, name, transposable, activation=None):
        """initializer

        Arguments:
        data -- list of training examples, each of which may be a list, np array, or scipy csc (sparse) array
        name -- name of input/output
        sequential -- indicates whether the first number in shape refers to timesteps
        is_input -- bool, indicating whether data is an input or output
        transposable -- indicates whether or not data is transposable
        activation -- activation that should be applied, should this be used as output for a model

        """

        self.data = np.array(data)
        self.name = name
        self.transposable = transposable
        self.is_csc = isinstance(self.data[0], csc_matrix)
        # shape of a single training example
        self.shape = self.data[0].shape
        if self.shape[0] == 1:
            self.shape = tuple([self.shape[-1]])
        print(self.shape)
        # dimension of one timestep
        self.dim = self.shape[-1]
        self.batch_data = None
    
    def data_generation(self, indexes):
        self.batch_data = np.empty((len(indexes),) + self.shape)
        if self.is_csc:
            for i, index in enumerate(indexes):
                self.batch_data[i] = self.data[index].toarray()
        else:
            for i, index in enumerate(indexes):
                self.batch_data[i] = self.data[index]

    def transpose(self, index, semitones):
        if not self.transposable:
            pass
        else:
            self.batch_data = np.concatenate((self.batch_data[...,-semitones:], self.batch_data[...,:-semitones]), axis=-1)
    
    def __len__(self):
        return len(self.data)
        


class ModelDataGenerator(tf.keras.utils.Sequence):
    """NoteTuple data generator. Can transpose music on the fly, including chroma data."""

    def __init__(self, data, inputs, outputs, seq_length=64,  batch_size=32, shuffle=True, transpose_on = True, st = 4, epoch_per_dataset=1):
        """Initialization

        Arguments:
        data -- list of ModelData objects
        inputs -- list of names, used select ModelData objects for model input
        outputs -- list of names, used select ModelData objects for model output

        Notes:
        -check how you did transposition in your summer project... was it correct?

        """

        # make a dictionary of the required inputs/outputs
        self.model_datas = {model_data.name: model_data for model_data in data if model_data.name in set(inputs + outputs)}
        # ensure that there aren't any required inputs/outputs with no corresponding ModelData objects
        assert set(self.model_datas.keys()) >= set(inputs), "inputs required that had no ModelData objects provided"
        assert set(self.model_datas.keys()) >= set(outputs), "outputs required that had no ModelData objects provided"
        self.inputs = inputs
        self.outputs = outputs
        # dictionary for a batch of data
        self.batch_data = {}
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
        self.on_epoch_end()

    def __transpose(self):
        'Randomly transposes examples up or down by up to self.st semitones'
        for i in range(self.batch_size):
            semitones = np.random.randint(-self.st, (self.st + 1))
            for model_data in self.model_datas.values():
                model_data.transpose(i, semitones)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int((len(list(self.model_datas.values())[0]) / self.epoch_per_dataset) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        self.__data_generation(indexes)

        # if augment is on, transpose
        if self.transpose_on:
            self.__transpose()
        
        input_data_batch = {input_data + '_in': self.model_datas[input_data].batch_data for input_data in self.inputs}
        output_data_batch = {output_data + '_out': self.model_datas[output_data].batch_data for output_data in self.outputs}

        return input_data_batch, output_data_batch

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        for data_name, model_data in self.model_datas.items():
            self.batch_data[data_name] = model_data.data_generation(indexes)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(list(self.model_datas.values())[0]))
        # print('indexes (in generator): ', self.indexes)
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
