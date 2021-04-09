# 781-piano-autoencoder
This project contains code, trained models, and audio examples from models trained to humanize piano music in light pop / blues / jazz genres. Humanization here means the prediction of velocities (how loud each note is). Future work may expand this scope to capturing temporal expression. The experiments were part of my BSc Honours project, completed in 2020.

### Audio Examples
Audio examples of piano music humanized by the models from this project can be found [here on soundcloud](https://soundcloud.com/user-611170338/sets/deep-learning-and-music-humanizing-piano-scores-longer-examples).  
There is also a folder called 'audio_examples' in this repository. File names contain a prefix `ex***` which denotes the specific musical example in question. File names also have suffixes as follows:  
* `No Humanization`: indicates there is no human expression in the example, i.e. velocity = 64 for all notes.
* `prediction_val_weights`: indicates velocities were predicted by a model using its best weights as measured by validation loss
* `prediction_train_weights`: indicates velocities were predicted by a model using its best weights as measured by training loss
* `original`: indicates the velocities of notes come from the original pianist (me)  

### Try it yourself
Models are [available for use in browser](https://aleathwick.github.io/midi-predict-react/). Upload a midi file, and click 'predict'. Midi files must have notes that will quantize accurately to the nearest sixteenth note, as determined by the BPM of the midi file (i.e. the midi file should have been recorded to a metronome/click track). 

### Inspiration: GrooVAE
The inspiration for this project is [Google's GrooVAE class of models](https://magenta.tensorflow.org/groovae), in particular, the methods they used for collating a dataset appropriate for training models to humanize drum sequences, where humanizing involves predicting velocities (how hard each note is struck) and timing offsets (time offset in ms to absolute time of beat or sub-beat for each note). Previous work in humanizing piano music has suffered from the difficulty of obtaining appropriate datasets, in which note-wise velocity and timing information from human performances can be matched to each note in music scores. The authors of GrooVAE found an elegant solution, which was to record drummers (in MIDI format) playing to a metronome; timestamps of notes and beats can then be compared, and notes assigned to the closes beat or 'sub-beat', allowing a score to be 'extracted' from the recordings. I recorded more than 20 hours of piano data using using this method, across two datasets (which I call *d1* and *d2*). In an effort to make the learning task easier, the second of these is very restricted in the textures and styles of playing allowed. I aimed to predict velocities in piano playing, initially using variational autoencoders, as found in GrooVAE; this failed, but a simple stacked LSTM model performed very well, capturing short term human-like expression very well. Long term expression (capturing more distant dependencies) has much room for improvement.

### Representation
I experimented with different ways of representing piano music for deep learning models. Particularly in a low data situation (restricting the dataset to an eighth of its original size), very clear differences in performance and training stability emerged between representations. In particular, representing pitch using a combination of pitch class (12-bit OHE vector) + pitch height (continuous variable in \[0, 1]) worked far better than using an 88-bit OHE vector. Along with beat and sub-beat indicator OHE vectors, I called this representation 'NTC'.

### Repository
This repository contains:
* Three scripts for training models; `run_experiment_ntc.py` is the script used for training stacked LSTM models using NTC representation. The script produces predictions for the validation set, using the best validation weights from training. [Sacred](https://github.com/IDSIA/sacred) is used for logging experiments, hence the decorators in the scripts.
* Python source code in `src/`.
* `Datasets` contains d1 and d2 datasets; these were recorded to a metronome, so that notes can be quantized (assigned to the nearest beats and sub-beats). *d1* contains music improvised in various light/popular/jazzy styles, with a great variety of textures, with four sub-beats per beat. *d2* is very limited in comparison, with two sub-beats per beat, the RH containing only chord melody, and LH only allowed alberti bass style accompaniment.
* `audio_examples` contains a few audio examples from model predictions.
* Weights and metrics from training runs are available, found in `experiments`. Many of training runs contain midi files produced by predicting on the validation set at the end of training.

### Midi files (datasets)
Midi files are recorded and then placed in `training_data/midi_train/`. Midi files are manipulated using the [pretty midi package](https://github.com/craffel/pretty-midi), which itself uses [Mido](https://github.com/mido/mido). Mido is lower level, and represents all midi events using relative timings, i.e. time since last midi event. Pretty midi stores things using their absolute time in seconds.

Filenames are formatted as follows: in `as_79_D.mid`, `as` is a counter (aa, ab, ac, ad, etc...) that gives every file a unique ID. `79` indicates the tempo in beats per minute, and `D` indicates the key of the piece (24 possible keys, an `m` suffix indicates the key is a minor key. No `m` suffix indicates a major key).

### Dissertation
To give an idea of my work, here is the title and abstract from my dissertation.

#### Humanizing Piano Sequences Using Deep Learning: Inverse Sequence Transformations and Representations

Previous work in humanizing piano scores using machine learning has been hampered by a lack of appropriate datasets, which stems from the difficulty of extracting music score like information from the raw note timing and velocity information of piano performances. Gillick et al. (2019) generate training data for the equivalent drumming task by recording drummers playing in time with aural beat cues, enabling the recovery of rhythmic score information. This work extends this technique to piano music, posing the non-trivial humanization task as the inverse of a ‘dehumanization’ transform that is trivial to compute when the times of beats are known, and explores different ways of representing music data as inputs to deep learning models. Results clearly show that small RNN-LSTM models (< 50,000 parameters) are capable of predicting highly convincing note velocities for piano sequences in popular contemporary music styles. The choice of representation affected training stability and model performance, proving especially critical in low data situations; I propose a new representation that factors pitch into pitch class and a continuous variable representing pitch height.

