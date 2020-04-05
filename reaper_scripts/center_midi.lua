-- Script for finding average offset from semiquaver grid, and moving notes by their average distance from that grid
-- Very helpfl for getting a sense of syntax + iterating over notes:
-- https://forum.cockos.com/showthread.php?t=171015

-- Function for printing to reaper console
function Msg(str)
  reaper.ShowConsoleMsg(tostring(str) .. "\n")
end

reaper.ClearConsole();

-- Get HWND
hwnd = reaper.MIDIEditor_GetActive()

-- Get current take being edited in MIDI Editor
take = reaper.MIDIEditor_GetTake(hwnd)

-- Get ticks per quarter... this is quite hacky, but works
tps = reaper.MIDI_GetPPQPos_EndOfMeasure(take, 1) / 16

Msg("ticks per 16th note:" .. tps)

-- count all notes and ccs
retval, notes, ccs, sysex = reaper.MIDI_CountEvts(take) 

note_total = 0

-- Keep sum so that average can be calculated
offset_sum = 0
-- Might want offsets later? If I can figure out some graphing...
-- Note that reaper indexes notes starting with i, but lua prefers to start array indexing with 1
offsets = {}


for i=0, notes-1 do
   retval, sel, muted, startppq, endppq, chan, pitch, vel = reaper.MIDI_GetNote(take, i)
   if sel == true then -- find which notes are selected
       note_total = note_total + 1 -- if notes selected add it to total note count
       offset = startppq % tps
       if offset > tps / 2 then
          offset = offset - tps
       end
       offsets[i+1] = offset
       offset_sum = offset_sum + offset
   end
   i=i+1
end

-- no math.round function, use floor with + 0.5
mean_offset = math.floor(offset_sum / note_total + 0.5)

Msg("Mean Offset: " .. mean_offset)

if not (mean_offset == 0) then -- check mean offset isn't 0 
    -- move all note starts and note ends by offset
    Msg("Moving Notes")
    for i=0, notes-1 do
       retval, sel, muted, startppq, endppq, chan, pitch, vel = reaper.MIDI_GetNote(take, i)
       if sel == true then -- find which notes are selected
           -- calculate new start and end of note, and update it
           new_start = startppq - mean_offset 
           new_end = endppq - mean_offset
           reaper.MIDI_SetNote(take, i, nil, nil, new_start, new_end, nil, nil, nil, 1)
       end
       i=i+1
    end
    
    Msg("Moving CC")
    cc_total = 0
    for i=0, ccs-1 do
       retval, sel, muted, ppqpos, chanmsg, chan, msg2, msg3 = reaper.MIDI_GetCC(take, i)
       if sel == true then -- find which ccs are selected
           cc_total = cc_total + 1 -- if cc selected then shift it to the correct center
           new_pos = ppqpos - mean_offset
           reaper.MIDI_SetCC(take, i, nil, nil, new_pos, nil, nil, nil, nil, 1)
       end
       i=i+1
    end
    
    reaper.MIDI_Sort(take)
    
else
    Msg("Not moving notes")
end
