# 781-piano-autoencoder

Models:
initial_exploration.ipynb has lots of fiddling around with data and models, and src/models.py has some models that are set up. This includes a simple encoder/decoder setups (hierarchical, sequential, and convolutional) that perform the operation:
robotic input -> lower dimensional 'latent vector' -> reassemble a humanized version of input given the latent vector

Data:
src/ml_classes.py has the class I'm using to store data for feeding into the model, and src/data.py has the code for producing examples given midi data (midi comes from the output of my digital piano). I'm opting for sparse matrices using the scipy package, which saves a huge amount of space. I've just uploaded some of the training data, split into training examples 64 timesteps long (stored as numpy arrays, serialized as python pickle files) . This consists of:
H - matrix of shape 301 training examples x 64 timesteps x 88 notes. Ones indicate note starts (the note 'hits', hence 'H')
O - matrix of shape 301 training examples x 64 timesteps x 88 notes. The same as H, except instead of ones, there is a real value indicting how far away from the true beat that note was played (the note's 'offset', hence 'O').
V - matrix of shape 301 training examples x 64 timesteps x 88 notes. Real values indicating the velocity with which each note was played.
key - 301 training examples x 12 keys. Indicator variable, indicating which of the twelve possible keys the training example is in.
tempo - 301 training examples x 1. Real value for each training example, indicating the tempo it was played at (faster or slower). I normalized it so it is somewhere between -1 and 1, but this was done quite arbitrarily.
