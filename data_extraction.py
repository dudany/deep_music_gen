import os
from typing import List

import music21
from music21 import note, chord


def run_data_extraction(path: str) -> List:
    notes_data = []
    for i, f in enumerate(os.listdir(path)):
        file_p = os.path.join(path, f)
        midi_file = music21.converter.parse(file_p)
        midi_notes = extract_notes_from_midi(midi_file)
        notes_data.append(midi_notes)
    return notes_data


def extract_notes_from_midi(midi_file: music21.stream.Stream) -> List:
    """
    This function extract all the notes out of the midi files in the signed data path

    :param midi_file: Midi file
    :return: return list of notes out of the midi files
    """

    notes = []
    parts = music21.instrument.partitionByInstrument(midi_file)
    if parts:  # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else:  # file has notes in a flat structure
        notes_to_parse = midi_file.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes
