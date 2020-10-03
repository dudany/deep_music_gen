import os
import numpy as np
import music21
from music21 import note, chord
from typing import List, Dict, Tuple


def run_data_extraction(path: str) -> List:
    """
    This function returns list of unvectorized prepared data
    :param path:
    :return:
    """

    notes_data = []
    for i, f in enumerate(os.listdir(path)):
        file_p = os.path.join(path, f)
        if os.path.isdir(file_p):
            temp_list = run_data_extraction(file_p)
            for n in temp_list:
                notes_data.append(n)
        elif os.path.splitext(file_p)[1] == '.mid':
            midi_file = music21.converter.parse(file_p)
            midi_notes = extract_notes_from_midi(midi_file)
            for n in midi_notes:
                notes_data.append(n)
    return notes_data


def get_notes_mapping_dict(notes_list: List) -> Tuple[Dict, np.array]:
    """
    Function get list of midi notes and returns mapping for each note

    :param notes_list:
    :return:
    """
    assert len(notes_list) > 0, 'Empty notes list !!'

    full_list = sorted(set(notes_list))
    notes2idx = {note_e: i for i, note_e in enumerate(full_list)}
    idx2note = np.array(full_list)
    return notes2idx, idx2note


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


def vectorize_notes_by_mapping(notes_list: List, mapping: Dict) -> np.array:
    vectorized_output = np.array([mapping[char] for char in notes_list])
    return vectorized_output
