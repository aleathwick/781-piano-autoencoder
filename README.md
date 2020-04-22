# 781-piano-autoencoder

## Inspiration: GrooVAE
The inspiration for this project is [Google's GrooVAE class of models](https://magenta.tensorflow.org/groovae), in particular the methods they used for collating their dataset. I have previously worked with data from [the MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro), which consists of midi recorded live from classical piano performances. The MAESTRO dataset is impressive in size, but being collected as a live stream of notes, it is not so suitable for tasks like teaching a model to add human expression to an existing musical score.  

### Improvisation vs Performance
When building models that make musical predictions, we could pursue a purely generative model, in which a stream of notes, along with their timings, velocities (how hard they are struck), and durations are produced, one after the other. This is like asking an algorithm to **improvise** music. The MAESTRO dataset is a large collection of streams of notes, and is an effective resource for modelling the distribution of a note given the notes that have come before. But we may already have a set of notes and timings in mind - for example, a musical score, written by any composer you care to name - and we lack the velocities and fluctuations in timing that make a performance human, and would like a model to predict that for us. That is conditioned on notes and their attributes, predict some other attributes - this is what humans do when the **perform** a piece of music.  
  
### A breif explanation of rhythm
Musical attributes can be divided roughly into pitch and rhythm, and for every note in a musical score, it is these two attributes that would be defined, leaving a human (or algorithm) to determine other note attributes (velocity, tempo fluctuations, rhythmic subtleties). Pitch is easy to recover from a dataset like MAESTRO, but rhythm is decidedly more difficult.  
  
The absolute timings of notes (in ms) are recorded, but normally a score contains information on what phrase, bar, beat, and sub-beat a note belongs to, all of which has a bearing on a notes importance and relationship to other notes. It is all rather recursive, like russian dolls. It is arranged like so:

* Phrase: a collection of bars, that together form a 'musical sentence'. Often a phrase is a group of four bars.
* Bar: a collection of beats, in which the first beat (the 'down beat') is the most important. Each bar contains the same number of beats (often 3, like in a waltz, or 4, like in a march). Think of counting '1, 2, 3, 4, 1, 2, 3, 4', to a marching band. Each group of four is a bar, and the count of '1' receives special emphasis. 
* Beat: a collection of sub-beats. Like a bar being divided into 4, a beat can be divided into 4 sub-beats, in which the first is the most important - the first sub beat marks the start of each beat. The marching band might play notes *during* a beat rather than *exactly at the start* of a beat. If you were counting each beat exactly in time, as above in the marching example, the first sub beat is precisely when you would say each of '1, 2, 3, 4'.  
  

In reality, a note can be divided into 2, 4, 8, 16, or 64 sub beats: the main thing to notice with rhythm is that things tend to be recursively divided into two! There are, of course, exceptions - but in the majority of cases, it is true. Composers (in the western classical tradition, and in modern pop music) write music which places notes on this rhythmic grid. Assuming this recursive division by two greatly simplifies notation, as it is only exceptions to this assumption that require further clarification. We can ask a question about a note like: what bar, beat, and sub beat does this fall on? The answer might be something like the 4th bar, the 3rd beat, the 2nd sub beat. 

### Conditioning on Musical Attributes: MAESTRO, and the Problem of Rhythm
The issue with the MAESTRO dataset is that we only know the absolute timings of notes - we don't know what bars, beats, or sub beats they belong to. We therefore cannot use the MAESTRO dataset to train a model that conditions on this rhythmic information to produce a sensible performance of a score. Efforts have been made to produce algorithms take such a stream of notes, and given the original written score for the music, attempt to work out which note in the stream comes from which note in the score, thus recovering the position of each note in bars and beats.  
  
### Another approach: Recording to a Meteronome
The authors of GrooVAE had another approach (in a very different context to classical piano music: drumming). Rather than relying on algorithms to align the notes from a performance to the notes in a score, they recorded drummers, storing the times and velocities for the hits of each drum. The drummers played to a meteronome (a device that produces a click or tone at regular intervals) that marked every beat. Within the constraints of the meteronomic beats, the drummers played freely as regards velocity, microtiming differences as compared to the beats as defined by the meteronome (not playing mechanicaly **exactly** on the beat, or sub beat - only a machine can do that!), and what beat or sub beats to hit drums on.  
  
The exact timings of the beats (and sub beats - allowing each note to be divided into four) are known - and the notes as played by the drummers can then be assigned to the closest sub beat to which they fall. If the drummers are playing very 'tight' and accurately, then the notes will not fall very far away from the sub beats at all, and this process will be smooth. If a drummer is playing sloppily, then sometimes notes may be assigned to the wrong sub beat. This could seriously misinform a learning algorithm about the correlations between velocity and sub beat - the first sub beat of a beat is normally the strongest, and such errors in the training data could lead, for example, to strong drum hits that fall on the first sub beat to be assigned to the second sub beat.  
  
Drums recorded in this way can be factored into several matrices, each of them of dimension Tn\*Dn, where Tn is the number of time steps (measured in sub beats), and Dn is the number of drums:

* H, Hits: the matrix showing what drums were hit at what times (which beat and which sub beat), with a 1 at position (Ti, Dj) indicating a hit on the jth drum at time i, and a 0 indicating no hit.
* V, Velocities: like H, but instead of ones indicating hits, this matrix contains values in \[0, 1\] indicating the velocities of those hits.
* O, Offsets: like V, but contains values in \[-0.5,0.5\], indicating time (measured in the smallest sub beat unit) to the closest sub beat.  
  
The primary prediction task is predicting V and O, given H. This prediction task, if accomplished successfuly, is of great use, for example to those in amateur music production who do not have access to a drummer.
  
## Data
In a similar vein to GrooVAE, one of the contributions of this project is a data set of light popular style piano music recorded to a meteronome. This way, no music score is needed to assign each note to its proper beat and sub beat. I improvised the music myself on a digital piano, sometimes sticking to preset chord progressions, sometimes moving freely.  
  
### Potential issues
Compared to the drum dataset, there are some issues that arise when creating a piano dataset in this way:

* **Drums vs notes**: There were only 9 unique drums used in the final dataset for GrooVAE, but the piano has 88 keys. Furthermore, while each drum of a drum kit has its own idiomatic function (which may change according to musical style), each note of a piano may take on a very different function depending on the key the music is written in alone, not to mention its place in relation to the currently used register (e.g. bottom third of the keys, or middle third of the keys). The same note may contribute tension or relaxation to a phrase, depending on the context. Therefore, identifying the role of a note in piano music requires much more complex reasoning about context than does drum music.
* **Note durations**: Drum music does not have note durations, but piano music does. In a piano score, notes not only have a starting beat and sub beat, but also an ending sub beat. In this piano dataset, the note-starts are precisely what one would expect to see in the musical score of the improvisations, but the note-ends are not. Although the note-starts in the data set can be moved to the closest sub beat and treated like a musical score (because they were played with a meteronome), the note ends were not placed so precisely - and therefore any models trained using this dataset predict a humanized version of the score will only be able to utilize the note-starts to make this prediction. Note-ends as found in a score are a factor that inform the interpretation of the music by the pianist; a musical score may indicate quite clearly that *this* note is meant be ended quite early, long before *that* note starts.
* **Durations again**: predicting the velocities and offsets, as achieved in drum music, is not enough to produce a convincing performance of a score. The lengths of the notes are a vital part of the interpretation of a pianist.
* **Phrase length**: Drum music often operates in phrases two bars long, with an established pattern that repeats over and over. An autoencoder (autoencoders are excellent at finding lower dimensional representations of complex data), as used in GrooVAE, can thus sensibly work with a two bar phrase. Piano phrases are often longer, more varied, and it is therefore a more complex task to find a meaningful lower dimensional space that encapsulates piano phrases.
* **Potential lack of data**: the above issues, particurly regarding phrase length and number of notes, may result in a much larger dataset being needed than used in the GrooVAE project. This will have to be investigated - it may be that the quantity of data I am able to record is a bottleneck!


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
