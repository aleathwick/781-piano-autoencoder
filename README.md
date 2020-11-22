# 781-piano-autoencoder
This project contains material related to my BSc Honours project, including code, models, and audio/MIDI predictions from deep learning models trained to humanize piano music.

### Inspiration: GrooVAE
The inspiration for this project is [Google's GrooVAE class of models](https://magenta.tensorflow.org/groovae), in particular, the methods they used for collating a dataset appropriate for training models to humanize drum sequences, where humanizing involves predicting velocities (how hard each note is struck) and timing offsets (time offset in ms to absolute time of beat or sub-beat for each note). Previous work in humanizing piano music has suffered from the difficulty of obtaining appropriate datasets, in which note-wise velocity and timing information from a human performances can be matched to each note in a music score. The authors of GrooVAE found an elegant solution, which was to record drummers (in MIDI format) playing to a metronome; timestamps of notes and beats can then be compared, and notes assigned to the closes beat or 'sub-beat', allowing a score to be 'extracted' from the recordings. I recorded more than 20 hours of piano data using using this method, across two datasets (which I call *d1* and *d2*. The second of these is very restricted in the allowed textures/styles of playing, to make the learning task easier. I initially tried variational autoencoders, after GrooVAE; this failed, but a simple stacked LSTM model performed very well.

### Representation
I experimented different ways of representing piano music for deep learning models. Particularly in a low data situation (restricting the dataset to an eighth of its original size), very clear differences in performance and training stability emerged between representations. In particular, representing pitch using a combination of pitch class (12-bit OHE vector) + pitch height (continuous variable in \[0, 1]) worked far better than using an 88-bit OHE vector. Along with beat and sub-beat indicator OHE vectors, I called this representation 'NTC'.

### Repository
This repository contains:
* Three scripts for training models; `run_experiment_ntc.py` is the script used for training stacked LSTM models using NTC representation. The script produces predictions for the validation set, using the best validation weights from training. [Sacred](https://github.com/IDSIA/sacred) is used for logging experiments, hence the decorators in the scripts.
* Python source code in `src`.
* `Datasets` contains d1 and d2 datasets; these were recorded to a metronome, so that notes can be quantized (assigned to the nearest beats and sub-beats). *d1* contains music improvised in various light/popular/jazzy styles, with a great variety of textures. *d2* is very limited in comparison, with the RH containing only chord melody, and LH only alberti bass style accompaniment.
* `audio_examples` contains a few audio examples from model predictions.
* Many weights from training runs are available, and metrics from model training runs, found in `experiments`. Many of these have midi files produced by predicting on the validation set at the end of training.

### Midi files (datasets)
Midi files are recorded and then placed in `training_data/midi_train/`. Midi files are manipulated using the [pretty midi package](https://github.com/craffel/pretty-midi), which itself uses [Mido](https://github.com/mido/mido). Mido is lower level, and represents all midi events using relative timings, i.e. time since last midi event. Pretty midi stores things using their absolute time in seconds.

Filenames are formatted as follows: in `as_79_D.mid`, `as` is a counter (aa, ab, ac, ad, etc...) that gives every file a unique ID. `79` indicates the tempo in beats per minute, and `D` indicates the key of the piece (24 possible keys, an `m` suffix indicates the key is a minor key. No `m` suffix indicates a major key).  
