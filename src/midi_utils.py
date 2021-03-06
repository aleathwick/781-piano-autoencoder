import pretty_midi
import numpy as np
from scipy.sparse import csc_matrix
import src.data as data
import src.models as models
from collections import namedtuple
import copy

# cc64 is sustain
# cc66 is sostenuto
# cc67 is soft

def handy_functions():
    #note or instrument names to numbers
    pretty_midi.note_name_to_number('C4')
    pretty_midi.instrument_name_to_program('Cello')
    
    # shift pitches of notes
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch += 5


######## pretty midi functions ########

def trim_silence(pm):
    "trims any silence from the beginning of a pretty midi object"
    
    # get the time of the first event (note or cc)
    if pm.instruments[0].control_changes != []:
        delay = min(pm.instruments[0].notes[0].start, pm.instruments[0].control_changes[0])
    else:
        delay = pm.instruments[0].notes[0].start
    
    # subtract the delay from note objects
    for note in pm.instruments[0].notes:
        note.start = max(0, note.start - delay)
        note.end = max(0, note.end - delay)

    # subtract the delay from cc objects
    for cc in pm.instruments[0].control_changes:
        cc.time = max(0, cc.time - delay)


def sustain_only(pm):
    """Remove all non sustain cc messages from piano midi pm object"""
    filtered_cc = []
    # I'm going to assume that there is just one instrument, piano
    for cc in pm.instruments[0].control_changes:
        if cc.number == 64:
            filtered_cc.append(cc)
    pm.instruments[0].control_changes = filtered_cc


def pm_head(pm, seconds = 11):
    """Return the first seconds of a piano midi pm object"""
    pm.instruments[0].notes = [note for note in pm.instruments[0].notes if note.end < seconds]
    pm.instruments[0].control_changes = [cc for cc in pm.instruments[0].control_changes if cc.time < seconds]


def bin_sus(pm, cutoff = 64):
    """Set sustain so it is either completely on or completely off"""
    filtered_cc = []
    sustain = False
    # I'm going to assume that there is just one instrument, piano
    for cc in pm.instruments[0].control_changes:
        if cc.number == 64: # if it is sustain
            if sustain == False and cc.value >= cutoff:
                sustain = True
                cc.value = 127
                filtered_cc.append(cc)
            elif sustain == True and cc.value < cutoff:
                sustain = False
                cc.value = 0
                filtered_cc.append(cc)
        else:
            filtered_cc.append(cc)
    pm.instruments[0].control_changes = filtered_cc


def desus(pm, cutoff = 64):
    """(in place!) Remove sustain pedal, and lengthen notes to emulate sustain effect"""
    # collect intervals in which pedal is down, and remove the pedal messages
    filtered_cc = []
    sustain = False
    intervals = []
    downtime = -1
    for cc in pm.instruments[0].control_changes:
        if cc.number == 64: # if it is sustain
            if sustain == False and cc.value >= cutoff:
                sustain = True
                downtime = cc.time
            elif sustain == True and cc.value < cutoff:
                sustain = False
                uptime = cc.time
                intervals.append((downtime, uptime))
        else:
            filtered_cc.append(cc)
    pm.instruments[0].control_changes = filtered_cc
    # print(intervals)

    # Now, use the intervals to extend out notes in them
    # We can structure our code like this because notes are ordered by end time
    # If that wasn't the case, we would need to do some sorting first
    index = 0
    last = 0
    extended_notes = []
    for note in pm.instruments[0].notes:
        while index < len(intervals) and note.end > intervals[index][1]:
            index += 1
        if index >= len(intervals):
            break
        # at this point, we know that note.end < intervals[index][1]
        # we test whether the end of the note falls in a sustain period
        if note.end > intervals[index][0] and note.end < intervals[index][1]:
            note.end = intervals[index][1]
            extended_notes.append(note)
        
    # now, we need to check for extended notes that have been extended over their compatriots...
    # this is horribly inefficient. But it does the job.
    # Could set it so comparisons are done between lists of same notes.
    for long_note in extended_notes:
        for note in pm.instruments[0].notes:
            if note.pitch == long_note.pitch and note.start < long_note.end and note.end > long_note.end:
                long_note.end = note.start
                # or could set it to note.end. I don't know which is best. Both seem ok.




######## chroma and feature augmentation ########

# methods for getting chroma for all time steps of a score
def pm2chroma(pm, mode='normal'):
    '''generates chroma for each note of a pretty midi file'''
    roll = pm2roll(pm)
    chroma = roll2chroma(roll, mode=mode)
    return chroma


def roll2chroma(roll, mode='normal'):
    chroma = np.zeros((len(roll), 12))
    for i, sounding in enumerate(roll):
        chroma[i] = sounding2chroma(sounding, mode=mode)
    return chroma

# methods for going to roll representation
def pm2roll(pm):
    """returns a numpy array of shape (n_notes, 88), containing pitches active when each note is being played"""
    notes = pm.instruments[0].notes
    note_times = [0] * 88
    roll = np.zeros((len(notes), 88))
    for i, note in enumerate(pm.instruments[0].notes):
        pitchB = pitchM2pitchB(note.pitch)
        note_times[pitchB] = note.off
        for j in range(88):
            if note.start < note_times[j]:
                roll[i,j] = 1
    return roll
    
# method for getting chroma from a single snapshot of sounding notes to chroma
def sounding2chroma(sounding, mode='normal'):
    '''get chroma for a single snapshot of sounding notes (one step of roll)'''
    chroma = np.zeros(12)
    if mode == 'normal':
        chroma[np.where(sounding == 1)[0] % 12] = 1
    elif mode == 'weighted':
        for i in np.where(sounding == 1)[0]:
            if chroma[i % 12] == 0:
                chroma[i % 12] = 1 - i / 87
    elif mode == 'lowest':
        sounding_pitches = np.where(sounding==1)[0]
        if len(sounding_pitches) != 0:
            chroma[min(sounding_pitches) % 12] = 1
    else:
        raise ValueError('mode is not understood')
    return chroma

######## different representations and related functions ########

def snap_to_grid(event_time, size=8):
    """takes an event time (in seconds) and gives it back snapped to a grid.
    I.e. multiples by 1000, then rounds to nearest multiple of 8, if default grid is 8ms
    
    Parameters
    ----------
    event_time : float
        Time of event, in seconds
    
    Returns
    ----------
    grid_time : int
        Time of event, in miliseconds, rounded to nearest 8 miliseconds

    """
    ms_time = event_time * 1000
    # get the distance to nearest number on the grid
    distance = ms_time % size
    if distance < size / 2:
        ms_time -= distance
    else:
        ms_time -= (distance - 8)
    return int(ms_time)


key2int = {'C':0,'Am':0,'Db':1,'Bbm':1,'D':2,'Bm':2,'Eb':3,'Cm':3,'E':4,'C#m':4,'F':5,'Dm':5,'F#':6,'D#m':6,'Gb':6,'Ebm': 6,
                'G':7,'Em':7,'Ab':8,'Fm':8,'A':9,'F#m':9,'Bb':10,'Gm':10,'B':11,'G#m':11}
int2key = {value: key for key, value in key2int.items()}

def center_pm(pm, sub_beats=4):
    """shift all notes by some constant so that the average offset is zero"""
    beat_length = 60 / pm.get_tempo_changes()[-1][0]
    sub_beat_length = beat_length / sub_beats
    max_offset = sub_beat_length / 2
    offsets = []
    # get all offsets (distance to nearest subbeat)
    for note in pm.instruments[0].notes:
        # time since last sub beat
        offset = note.start % sub_beat_length
        # are we closer to the last sub beat, or next sub beat? 
        if offset > max_offset:
            offset = offset - sub_beat_length
        offsets.append(offset)

    # calculate mean
    mean_offset = np.mean(offsets)

    # use this to center notes and cc messages
    for note in pm.instruments[0].notes:
        note.start -= mean_offset
        note.end -= mean_offset
    for cc in pm.instruments[0].control_changes:
        cc.time -= mean_offset

def filter_notes(pm, vel_cutoff):
    """filter out any notes that are beneath a threshold in velocity"""
    pm.instruments[0].notes = [note for note in pm.instruments[0].notes if note.velocity >= vel_cutoff]


def pm2example(pm, key, beats_per_ex = 16, sub_beats = 4, sparse=True, use_base_key=False, vel_cutoff=4, V_no_zeros=False):
    """Converts a pretty midi file into sparse matrices of hits, offsets, and velocities
    
    Arguments:
    pm -- pretty midi file
    beats_per_ex -- number of beats per training example, measured in beats according to the midi file
    sub_beats -- number of sub beats used to quantize the training example
    sparse -- whether or not to use a sparse representation (scipy sparse array)
    use_base_key -- whether or not to transpose all examples to the same key (C or Am)
    V_no_zeros -- set all zero entries of V to the average example note velocity

    Returns:
    H, O, V, R -- sparse matrices (scipy) of shape (n_examples, beats_per_ex * sub_beats, 88):
            H - note starts
            O - offsets
            V - velocities
            R - roll (piano roll, indicating sounding notes for each timestep)
    
    """
    sustain_only(pm)
    bin_sus(pm)
    # filter_notes(pm, vel_cutoff)
    center_pm(pm)
    sustain = pm.instruments[0].control_changes
    desus(pm)

    # Get H, O, V for examples in a midifile
    n_examples = len(pm.get_beats()) // beats_per_ex
    
    # if midi file is shorter than minimum example length, return 
    if n_examples == 0:
        return None
    
    sub_beats_per_ex = sub_beats * beats_per_ex
    
    # initialize np array storage
    H = np.zeros((n_examples, sub_beats * beats_per_ex, 88))
    O = copy.deepcopy(H)
    V = copy.deepcopy(H)
    R = copy.deepcopy(H)


    if len(pm.get_tempo_changes()) > 2:
        print('Warning: tempo changes present')
    beat_length = 60 / pm.get_tempo_changes()[-1][0]
    sub_beat_length = beat_length / sub_beats
    
    max_offset = sub_beat_length / 2
    # sort notes by note start - by default they are sorted by note end
    notes_sorted = sorted(pm.instruments[0].notes, key = lambda x: x.start)
    current_sub_beat = 0
    sub_beat_times = [i + j * sub_beat_length for i in pm.get_beats() for j in range(sub_beats)]
    sub_beat_times = sub_beat_times[:n_examples * sub_beats_per_ex]

    # sort pedal
    S = np.zeros((n_examples * sub_beats_per_ex, 2))
    for message in sustain:
        timestep = int(round(message.time / sub_beat_length))
        if timestep >= len(S):
            break
        action = 1 if message.value > 64 else 0
        S[timestep,action] = 1
    S = np.reshape(S, (n_examples, sub_beats_per_ex, 2))


    end_times = np.zeros(88)
    for note in notes_sorted:
        # update note end times
        # # could snap to grid, but probably uneccessary?
        # end = snap_to_grid(note.end, sub_beat_length)
        # end_times[note.pitch - 21] = max(end_times[note.pitch - 21], end)
        end_times[note.pitch - 21] = max(end_times[note.pitch - 21], note.end)
        # iterate through sub-beats until we get to the one closest to the note start
        if note.start > sub_beat_times[-1] + max_offset:
            break
        while note.start > (sub_beat_times[current_sub_beat] + max_offset):
            timestep = current_sub_beat % sub_beats_per_ex
            example = current_sub_beat // sub_beats_per_ex
            R[example, timestep, np.where(end_times >= sub_beat_times[current_sub_beat] + max_offset)] = 1
            current_sub_beat += 1
        # calculate example and timestep index of this sub beat
        timestep = current_sub_beat % sub_beats_per_ex
        example = current_sub_beat // sub_beats_per_ex
        R[example, timestep, np.where(end_times >= sub_beat_times[current_sub_beat] + max_offset)] = 1
        # add information to H, O, and V
        H[example,timestep,note.pitch - 21] = 1
        O[example,timestep,note.pitch - 21] = (note.start - sub_beat_times[current_sub_beat]) / max_offset
        V[example,timestep,note.pitch - 21] = note.velocity / 127

    if use_base_key:
        semitones = -key2int[key]
        if semitones < -6:
            semitones += 12
        if semitones != 0:
            H = data.transpose_by_slice(H, semitones)
            O = data.transpose_by_slice(O, semitones)
            V = data.transpose_by_slice(V, semitones)
            R = data.transpose_by_slice(R, semitones)
            key = int2key[0]
    
    # get the mean velocity
    V_mean = np.array([[np.mean(v[np.where(v != 0)])] for v in V])
    
    if V_no_zeros:
        for i in range(len(V_mean)):
            V[i][np.where(V[i] == 0)] = V_mean[i]
    
    key_int = np.zeros((n_examples,12))
    key_int[...,key2int[key]] = 1

    # get vector with tempo for each example
    tempo = np.array([[data.normalize_tempo(pm.get_tempo_changes()[-1][0])] for _ in range(n_examples)])
    
    if sparse:
        key_int = [csc_matrix(k) for k in key_int]
        H = [csc_matrix(h) for h in H]
        O = [csc_matrix(o) for o in O]
        if not V_no_zeros:
            V = [csc_matrix(v) for v in V]
        R = [csc_matrix(r) for r in R]
        S = [csc_matrix(s) for s in S]

    return {'H': H, 'O': O, 'V': V, 'R': R, 'S': S, 'tempo': tempo, 'key': key_int, 'V_mean': V_mean}


def pm2notes_q(pm, sub_beats):
    """get the list of notes in pm object ready for use in NBC training data
    
    Arguments:
    pm -- pretty midi object
    sub_beats -- number of sub beats per beat
    
    Returns:
    notes -- list of pm note objects, but with pitch in [0,87], velocity in [0,1], and start/end
            expressed as closest sub_beat number
    
    """
    # grid resolution
    sub_beat_len = pm.get_beats()[1] / sub_beats
    # don't want to edit the original pm object, so we make a copy of notes
    notes = copy.deepcopy(pm.instruments[0].notes)
    for note in notes:
        note.pitch = pitchM2pitchB(note.pitch)
        note.start = int(round(note.start / sub_beat_len))
        note.end = int(round(note.end / sub_beat_len))
        note.velocity = note.velocity / 127

    return notes

def notes_q2nbq(notes, pm=None, seq_length=60, sub_beats=2, example_bars_skip=4, key=0, nth_example=None):
    """takes note_bin_q and returns examples of specified length
    
    Arguments:
    notes -- list of notes, note_bin_q style
        that is, pm notes with velocity and start/end times representing nearest sub beat.
        also with hand label (either 0 or 1)
    pm -- pretty midi object
    seq_length -- length of each example
    sub_beats -- number of sub beats per beat
    example_bars_skip -- examples start every example_skip bars
    key -- key of data in integer format
    nth_example -- take only every nth musical example

    Returns:
    Dictionary containing various information for each note of each training example.
    See comments in code for details. 
    
    
    Notes:
    four bars is around 50 notes, for the _8 dataset.
    
    """

    # normally pm notes are ordered by note end
    notes.sort(key = lambda note: note.start)

    # number of sub beats to skip forward by to get start of next example
    if nth_example == None:
        nth_example = 1
    
    sub_beat_skip = example_bars_skip * 4 * sub_beats * nth_example
    
    # n prefix indicates it is by note - not by sub beat, as in HOV format
    features = {'TSn': [], # starts, in sub beats (of whole example)
                'TEn': [], # ends, in sub beats (of whole example)
                # these three work together as indicator variables, together describing measure, beat, and sub beat
                'TBn': [], # note beat (of measure)
                'TMn': [], # note measure (of example, allow up to 16?)
                'TSBn': [], # note sub beat (of beat)
                'Pn': [], # pitches
                'PSn': [], # pitch as continuous value in [0,1]
                'PCn': [], # pitch class in [0,11]
                'Vn': [], # velocities (in [0, 1])
                'LRn': []} # left hand or right hand (in[0,1])
    
    
    # variable for starting sub beat of current example
    example_sub_beat_start = 0
    while len(notes) >= seq_length:
        # get the values for the relevant notes
        # note: LSTM inputs must have 3 dimensions! hence the extra [] around elements in list comprehensions
        features['TSn'].append([n.start - example_sub_beat_start for n in notes[:seq_length]])
        features['TEn'].append([n.end - example_sub_beat_start for n in notes[:seq_length]])

        # each element 
        features['TBn'].append([[ts % (sub_beats * 4) // sub_beats] for ts in features['TSn'][-1]])
        features['TMn'].append([[ts // (sub_beats * 4)] for ts in features['TSn'][-1]])
        features['TSBn'].append([[ts % sub_beats] for ts in features['TSn'][-1]])
        
        # pitch
        features['Pn'].append([n.pitch for n in notes[:seq_length]])
        # these aren't OHE, but LSTM needs 3 dimensional input, hence extra [] around every element
        features['PSn'].append([[p / 87] for p in features['Pn'][-1]])
        features['PCn'].append([[p % 12] for p in features['Pn'][-1]])
        
        # velocities
        # again, need extra dimension for LSTMs
        features['Vn'].append([[n.velocity] for n in notes[:seq_length]])

        features['LRn'].append([[n.hand] for n in notes[:seq_length]])
        
        example_sub_beat_start += sub_beat_skip
        
        # chop off example_skip beats from the start of the list of notes
        i = 0
        while i < (len(notes) - 1) and notes[i].start < example_sub_beat_start:
            i += 1
        notes = notes[i:]

    # print([k + str(max(v)) for k, v in features.items()])
    n_examples = len(features['TSn'])
    # print([(K, np.max([np.array(V).flatten()])) for K, V in features.items()])
    # convert to np arrays... a little more complex because we need to go to one hot encoding
    # get model input/output reqs, which stores dimension data for all outputs and inputs
    model_reqs = models.get_model_reqs(model_inputs='all', model_outputs='all', sub_beats=sub_beats, seq_length=seq_length)
    # convert to OHE np array if appropriate, otherwise convert to np array
    # use model input/output reqs to get OHE appropriateness
    for f in features.keys():
        features[f] = np.array(features[f])
        if model_reqs[f].ohe:
            # this is the number of variables the feature has for each timestep
            dim = model_reqs[f].dim
            # empty array of required shape
            expanded = np.zeros((n_examples,seq_length,dim))
            # print(f)
            # print(str(expanded.shape) + f)
            # make ones where needed
            # requires careful construction of indices!
            expanded[[i for i in range(n_examples) for _ in range(seq_length)],[j for i in range(n_examples) for j in range(seq_length)],features[f].flatten()] = 1
            features[f] = expanded
    
    # get tempi for each example (identical for each example in same pm file)
    features['tempo'] = np.array([[data.normalize_tempo(pm.get_tempo_changes()[-1][0])] for _ in range(n_examples)])
    
    # get key, one hot encoded
    key_int = np.zeros((n_examples,12))
    key_int[...,key2int[key]] = 1
    features['key'] = key_int

    # get mean velocity for each example
    features['V_mean'] = np.array([[np.mean(v)] for v in features['Vn']])

    return features

def pm2nbq(pm, seq_length=60, sub_beats=2, example_bars_skip=4, key='C', use_base_key=False, nth_example=None):
    """given pm, return nbq format examples (dictionary of features for each example)"""
    # don't want this in place
    pm = copy.deepcopy(pm)
    desus(pm)
    
    # add a hand attribute to pm note class
    setattr(pretty_midi.Note, 'hand', 0)
    if pm.instruments[0].name in ['LH', 'RH']:
        for note in pm.instruments[1].notes:
            # set note hand to 1 for all notes in second instrument note array
            note.hand = 1
        # pm2notes_q reads the first instrument's notes, so combine notes from both 'instruments' here
        pm.instruments[0].notes = pm.instruments[0].notes + pm.instruments[1].notes
        pm.instruments[0].sort(key=lambda x: x.start)
    elif len(pm.instruments) > 1:
        print('WARNING: more than one instrument in midi file, but instruments are not hands')
    
    # convert note information
    notes = pm2notes_q(pm, sub_beats)

    # if use_base_key, then transpose to C/Am
    if use_base_key and key != None:
        semitones = -key2int[key]
        if semitones < -6:
            semitones += 12
        for note in notes:
            note.pitch += semitones
    elif use_base_key and key == None:
        print('Warning: no key provided, but use base key set to True. Unable to transpose.')
        key=int2key[0]
    
    # include the pm, so that 
    nbq = notes_q2nbq(notes, pm, key=key, sub_beats=sub_beats, seq_length=seq_length, nth_example=nth_example)
    return nbq



def pitchM2pitchB(pitchM):
    """Maps midi notes to [0, 87]"""
    return pitchM - 21 # lowest A is 21


def pitchB2pitchM(pitchM):
    """Maps notes from [0, 87] to midi numbers"""
    return pitchM + 21 # lowest A is 21

def twinticks2sec(major_tick, minor_tick, major_ms=600, minor_ms=10):
    """Inverts sec2twinticks"""
    return (major_ms * major_tick + minor_ms * minor_tick) / 1000


def rebin(bin, a=128, b=32):
    """Maps from [0, a-1] to [0, b-1], useful for velocity"""
    if bin > a:
        print(f'Warning: input bin {bin} outside of input bin range of {a}')
        bin = a/2
    return round(bin * (b-1)/(a))


def note_bin2pm(note_bin, M_shift_ms = 600, m_shift_ms = 25,  M_duration_ms = 800, m_duration_ms = 50, n_velocity=16):
    """Performs inverse function of pm2notebin"""

    pm = pretty_midi.PrettyMIDI(resolution=125)
    pm.instruments.append(pretty_midi.Instrument(0, name='piano'))

    # define indexes for note_bins
    pitch = 0
    shift_major = 1
    shift_minor = 2
    duration_major = 3
    duration_minor = 4
    velocity = 5

    current_time = 0

    for noteB in note_bin:
        velocityM = rebin(noteB[velocity], n_velocity, 128)
        pitchM = pitchB2pitchM(noteB[pitch])
        current_time += twinticks2sec(noteB[shift_major], noteB[shift_minor], major_ms=M_shift_ms, minor_ms=m_shift_ms)
        duration = twinticks2sec(noteB[duration_major], noteB[duration_minor], major_ms=M_duration_ms, minor_ms=m_duration_ms)
        if duration == 0:
            duration += 0.02
        end = current_time + duration

        noteM = pretty_midi.Note(velocityM, pitchM, current_time, end)
        pm.instruments[0].notes.append(noteM)
    # sort by note offs, which is how pm objects are organized
    pm.instruments[0].notes.sort(key=lambda note: note.end)
    return pm










