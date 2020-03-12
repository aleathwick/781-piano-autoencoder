import numpy as np
import tensorflow as tf

class MelDataGenerator(tf.keras.utils.Sequence):
    """NoteTuple data generator. Can transpose music on the fly, including chroma data."""
    def __init__(self, input_data, output_data, data_shapes, aux_shapes, chroma=[], batch_size=64, seq_length=64, shuffle=True, augment=True, st = 4, epoch_per_dataset=1):
        """Initialization

        Arguments:
        data -- dictionary containing (potentially):
                -H, O, V, which are list of scipy sparse cscs arrays containing hits, offsets, and velocities for each training examples
                -chroma
                -key, a list of integers indicating the key of each example (0-11)
                -tempo, a list of floats indicating the tempo of each example


        """

        ''' Still to fix:

        -Some inputs need to be converted from cscs
            could just do an if check...
        -check how you did transposition in your summer project... was it correct?
        -check your transposition below actually works
        '''

        self.input_data = input_data
        self.output_data = output_data
        self.data_shapes = data_shapes
        self.input_data_batch = {}
        self.output_data_batch = {}
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        # if augment is true, data will be randomly transposed
        self.augment = augment
        self.on_epoch_end()
        # number of semitones to randomly transpose by
        self.st = st
        # this controls epoch
        self.epoch_per_dataset = epoch_per_dataset

    def __transpose(self):
        'Randomly transposes examples up or down by up to self.st semitones'
        for i in range(self.batch_size):
            semitones = np.random.randint(-self.st, (self.st + 1))
            # if this goes above or below the range of the piano, just use highest or lowest note
            self.input_data_batch['H'][i] = self.__concat_transpose(self.input_data_batch['H'][i], semitones)

            if 'chroma' in self.data_shapes.keys():
                    self.input_data_batch['chroma'][i] = self.__concat_transpose(self.input_data_batch['chroma'][i], semitones)
            if 'O' in self.data_shapes.keys():
                    self.input_data_batch['O'][i] = self.__concat_transpose(self.input_data_batch['O'][i], semitones)
            if 'V' in self.data_shapes.keys():
                    self.input_data_batch['V'][i] = self.__concat_transpose(self.input_data_batch['V'][i], semitones)
            if 'key' in self.data_shapes.keys():
                self.output_data_batch['key'][i] = (self.output_data_batch['key'][i] + semitones) % 12

    def __concat_transpose(self, data, semitones):
        """Transposition method for data in which position in last axis indicates pitch"""
        if semitones > 0:
            return np.concatenate((data[...,-semitones:], data[...,:-semitones]), axis=-1)
        if semitones < 0:
            return np.concatenate((data[...,-semitones:], data[...,:-semitones]), axis=-1)
        
    def __int_transpose(self, data, semitones):
        """Transposition method for data in which position in integer indicates pitch"""
        for j in range(self.seq_length):
           data[...,j] = min(max(data[...,j] + semitones, 0), 87)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int((len(self.input_data['H']) / self.epoch_per_dataset) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        self.__data_generation(indexes)
        # if chroma has been provided, fetch it

        # if augment is on, transpose
        if self.augment:
            self.__transpose()

        return self.input_data_batch, self.output_data_batch

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, Tx)
        for input_name, data in self.input_data.items():
            # hopefully this will work, getting the shape like this? 
            input_data = np.empty((self.batch_size,self.seq_length) + self.data_shapes[input_name])
            for i, index in enumerate(indexes):
                # Store sample, leaving off the last time step
                input_data[i] = data[index]
            self.input_data_batch[input_name] = input_data
        for input_name, data in self.output_data.items():
            # hopefully this will work, getting the shape like this? 
            input_data = np.empty((self.batch_size,self.seq_length) + self.data_shapes[input_name])
            for i, index in enumerate(indexes):
                # Store sample, leaving off the last time step
                input_data[i] = data[index]
            self.input_data_batch[input_name] = input_data
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    

class OoreDataGenerator(tf.keras.utils.Sequence):
    'Performance Representation data generator. Can transpose music on the fly, including chroma data.'
    def __init__(self, data, chroma=[], batch_size=64, dim=(601,), shuffle=True, augment=True, st = 4, epoch_per_dataset=1):
        """Initialization
        Note that data should be a list of X
        """
        self.X_data = data
        #the dimension of a single example
        self.dim = dim 
        self.batch_size = batch_size
        self.shuffle = shuffle
        # if augment is true, data will be randomly transposed
        self.augment = augment
        self.on_epoch_end()
        # number of semitones to randomly transpose by
        self.st = st
        self.chroma = chroma
        self.epoch_per_dataset = epoch_per_dataset

    def __transpose(self, X, Y):
        'Randomly transposes examples up or down by up to self.st semitones'
        for i in range(len(X)):
            semitones = np.random.randint(-self.st, (self.st + 1))
            # if this goes above or below the range of the piano, just use highest or lowest note
            for j in range(self.dim[0]):
                if 0 <= X[i,j] <= 87:
                    X[i,j] = min(max(X[i,j,0] + semitones, 0), 87)
                if 0 <= Y[i,j] <= 87:
                    Y[i,j] = min(max(Y[i,j,0] + semitones, 0), 87)
                if 88 <= X[i,j] <= 175:
                    X[i,j] = min(max(X[i,j,0] + semitones, 88), 175)
                if 88 <= Y[i,j] <= 175:
                    Y[i,j] = min(max(Y[i,j,0] + semitones, 88), 175)
            return semitones

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int((len(self.X_data) / self.epoch_per_dataset) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, Y = self.__data_generation(indexes)
        
        if len(self.chroma) != 0:
            C = self.__chroma_generation(indexes)
        if self.augment:
            semitones = self.__transpose(X, Y)
            if len(self.chroma) != 0:
                if semitones > 0:
                    C = np.concatenate((C[:,:,-semitones:], C[:,:,:-semitones]), axis=-1)
                elif semitones < 0:
                    C = np.concatenate((C[:,:,-semitones:], C[:,:,:-semitones]), axis=-1)
        if len(self.chroma) != 0:
            X = np.concatenate((X,C), axis=-1)

        
        # Different note attribute targets are separate outputs
        return X, Y

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, Tx)
        # Initialization
        X = np.empty((self.batch_size,) + self.dim)
        Y = np.empty((self.batch_size,) + self.dim) #I think this is right...? Because I'll use sparse categorical cross entropy.
        # Generate data
        for i, index in enumerate(indexes):
            # Store sample, leaving off the last time step
            X[i,:] = self.X_data[index][:-1]
            # Store expected output, i.e. leave off the first time step
            Y[i,:] = self.X_data[index][1:]

        return np.expand_dims(X, axis=-1), np.expand_dims(Y, axis=-1)
    def __chroma_generation(self, indexes):
        C = np.empty((self.batch_size,) + (self.dim[0],12))
        for i, index in enumerate(indexes):
            # Store sample, leaving off the last time step
            C[i,:,:] = self.chroma[index,:-1,:]
        return C
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    

class DataGenerator_onehot(tf.keras.utils.Sequence):
    'Generates data for a model expecting one hot input in performance representation'
    def __init__(self, data, batch_size=8, dim=(256,333), n_channels=1,
                 n_classes=333, shuffle=True):
        """Initialization
        Note that data should be a tuple containing (X, Y)
        """
        self.X_data, self. Y_data = data
        self.dim = dim #the dimension of a single example. Should it be (256, 333), the shape of a training example?
        self.batch_size = batch_size
        # self.labels = labels
        # self.list_IDs = list_IDs
        # self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        return int(np.floor(len(self.X_data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(indexes)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, Tx)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size), dtype=int) #I think this is right...? Because I'll use sparse categorical cross entropy.

        # Generate data
        for i, index in enumerate(indexes):
            # Store sample
            X[i,] = keras.utils.to_categorical(self.X_data[index], num_classes=self.n_classes) # don't think I need to specify datatype of float32?

            # Store expected output
            Y[i] = keras.utils.to_categorical(self.Y_data[index], num_classes=self.n_classes)

        return X, Y.transpose(1,0,2)


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
