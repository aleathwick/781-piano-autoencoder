import pretty_midi
import numpy as np
from scipy.sparse import csc_matrix
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
    """Remove sustain pedal, and lengthen notes to emulate sustain effect"""
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
    notes = pm.instruments[0].notes
    note_times = [0] * 88
    roll = np.zeros((len(notes), 88))
    for i, note in enumerate(pm.instruments[0].notes):
        pitchB = pitchM2pitchB(note.pitch)
        note_times[pitchB] = note.off
        for j in range(88):
            if note.start < note_times[j]:
                chroma_output[i,j] = 1
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
    """takes an event time (in seconds) and gives it back snapped to a grid with 8ms between each event.
    I.e. multiples by 1000, then rounds to nearest multiple of 8
    
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
                'G':7,'Em':7,'Ab':8,'Fm':8,'A':9,'F#m':9,'Bb':10,'Gm':10,'B':11,'G#':11}
int2key = {value: key for key, value in key2int.items()}

def transpose_by_slice(np1, semitones):
    np1 = np.concatenate((np1[...,-semitones:], np1[...,:-semitones]), axis=-1)
    return np1

def pm2example(pm, key, beats_per_ex = 16, sub_beats = 4, sparse=True, use_base_key=False):
    """Converts a pretty midi file into sparse matrices of hits, offsets, and velocities
    
    Arguments:
    pm -- pretty midi file
    beats_per_ex -- number of beats per training example, measured in beats according to the midi file
    sub_beats -- number of sub beats used to quantize the training example
    sparse -- whether or not to use a sparse representation (scipy sparse array)
    use_base_key -- whether or not to transpose all examples to the same key (C or Am)

    Returns:
    H, O, V -- sparse matrices (scipy) of shape (n_examples, beats_per_ex * sub_beats, 88):
            H - note starts
            O - offsets
            V - velocities
    
    
    """
    sustain_only(pm)
    desus(pm)

    # Get H, O, V for examples in a midifile
    n_examples = len(pm.get_beats()) // beats_per_ex
    sub_beats_per_ex = sub_beats * beats_per_ex
    
    # list of list of lists, of shape (n_examples, example length, none), where none is determined by the number of notes at that position
    H = np.zeros((n_examples, sub_beats * beats_per_ex, 88))
    O = copy.deepcopy(H)
    V = copy.deepcopy(H)


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


    for note in notes_sorted:
        # iterate through sub-beats until we get to the one closest to the note start
        if note.start > sub_beat_times[-1] + max_offset:
            break
        while note.start > (sub_beat_times[current_sub_beat] + max_offset):
            current_sub_beat += 1
        # calculate example and timestep index of this sub beat
        example = current_sub_beat // sub_beats_per_ex
        timestep = current_sub_beat % sub_beats_per_ex
        # add information to H, O, and V
        H[example,timestep,note.pitch - 21] = 1
        O[example,timestep,note.pitch - 21] = (note.start - sub_beat_times[current_sub_beat]) / max_offset
        V[example,timestep,note.pitch - 21] = note.velocity / 127

    if use_base_key:
        semitones = -key2int[key]
        if semitones < -6:
            semitones += 12
        if semitones != 0:
            H = transpose_by_slice(H, semitones)
            O = transpose_by_slice(O, semitones)
            V = transpose_by_slice(V, semitones)
            key = int2key[0]
    
    key_int = np.zeros((n_examples,12))
    key_int[...,key2int[key]] = 1

    # get vector with tempo for each example
    tempo = np.array([[pm.get_tempo_changes()[-1][0] / 100 - 1] for _ in range(n_examples)])
    
    if sparse:
        key_int = [csc_matrix(k) for k in key_int]
        H = [csc_matrix(h) for h in H]
        O = [csc_matrix(o) for o in O]
        V = [csc_matrix(v) for v in V]

    

    return {'H': H, 'O': O, 'V': V, 'tempo': tempo, 'key': key_int}



def pm2oore(pm):
    """Create event representation of midi. Must have sustain pedal removed.
    Will only have one note off for duplicate notes, even if multiple note offs are required.

    333 total possible events:
    0 - 87: 88 note on events, 
    88 - 175: 88 note off events
    176 - 300: 125 time shift events (8ms to 1sec)
    301 - 332: 32 velocity events

    Parameters:
    ----------
    pm : Pretty_Midi
        pretty midi object containing midi for a piano performance. Must have no sustain pedal.

    Returns:
    ----------
    events_with_shifts : list
        A list of events expressed as numbers between 0 and 332

    """
    # initially, store these in lists (time, int) tuples, where time is already snapped to 8ms grid,
    # and integers represent which event has taken place, from 0 to 332
    note_ons = []
    note_offs = []
    velocities = []
    n_velocities = 32
    for note in pm.instruments[0].notes:
        on = snap_to_grid(note.start)
        note_ons.append((snap_to_grid(note.start), note.pitch - 21)) # -21 because lowest A is note 21 in midi
        off = snap_to_grid(note.end)
        note_offs.append((snap_to_grid(note.end), note.pitch - 21 + 88))
        velocities.append((snap_to_grid(note.start), round(301 + note.velocity * (n_velocities - 1)/127))) #remember here we're mapping velocities to [0,n_velocities - 1]

    # remove duplicate consecutive velocities
    velocities.sort() #sort by time
    previous = (-1, -1)
    new_velocities = []
    for velocity in velocities:
        if velocity[1] != previous[1]: # check that we haven't just had this velocity
            new_velocities.append(velocity)
        previous = velocity
    velocities = new_velocities

    # Get all events, sorted by time
    # For simultaneous events, we want velocity change, then note offs, then note ons, so we
    # sort first by time, then by negative event number
    events = note_ons + note_offs + velocities
    events.sort(key = lambda x: (x[0], -x[1]))

    # add in time shift events. events 176 - 300.
    events_with_shifts = []
    previous_time = 0
    previous_event_no = -1
    for event in events:
        difference = event[0] - previous_time # time in ms since previous event
        previous_time = event[0] # update the previous event
        if difference != 0:
            shift = difference / 8 # find out how many 8ms units have passed
            seconds = int(np.floor(shift / 125)) # how many seconds have passed? (max we can shift at once)
            remainder = int(shift % 125) # how many more 8ms units do we need to shift?
            for seconds in range(seconds):
                events_with_shifts.append(300) # time shift a second
            if remainder != 0:
                events_with_shifts.append(remainder + 175)
        #append the event number only if it is not a repeated note off
        if 88 <= event[1] <= 175 and event[1] != previous_event_no or event[1] < 88 or event[1] > 175:
            events_with_shifts.append(event[1]) # append event no. only
            previous_event_no = event[1]
    return events_with_shifts        

def oore2pm(events):
    """Maps from a list of event numbers back to midi.

    333 total possible events:
    0 - 87: 88 note on events, 
    88 - 175: 88 note off events
    176 - 300: 125 time shift events (8ms to 1sec)
    301 - 332: 32 velocity events

    Parameters:
    ----------
    events_with_shifts : list
        A list of events expressed as numbers between 0 and 332

    Returns:
    ----------
    pm : Pretty_Midi
        pretty midi object containing midi for a piano performance.

    """
    pm = pretty_midi.PrettyMIDI(resolution=125)
    pm.instruments.append(pretty_midi.Instrument(0, name='piano'))

    notes_on = [] # notes for which there have been a note on event
    notes = [] # all the retrieved notes
    current_time = 0 # keep track of time (in seconds)
    current_velocity = 0

    for event in events:
        # sort note ons
        if 0 <= event <= 87:
            pitch = event + 21
            # set attributes of note, with end time as -1 for now
            note = pretty_midi.Note(current_velocity, pitch, current_time, -1)
            # add it to notes that haven't had their note off yet
            notes_on.append(note)
        # sort note offs
        elif 88 <= event <= 175:
            end_pitch = event + 21 - 88
            new_notes_on = []
            for note in notes_on:
                if note.pitch == end_pitch:
                    note.end = current_time
                    notes.append(note)
                else:
                    new_notes_on.append(note)
            notes_on = new_notes_on
        # sort time shifts
        elif 176 <= event <= 300:
            shift = event - 175
            current_time += (shift * 8 / 1000)
        # sort velocities
        elif 301 <= event <= 332:
            rescaled_velocity = np.round((event - 301) / 31 * 127)
            current_velocity = int(rescaled_velocity)
    notes.sort(key = lambda note: note.end)
    # Just in case there are notes for which note off was never sent, I'll clear notes_on
    
    pm.instruments[0].notes = notes
    return pm


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










