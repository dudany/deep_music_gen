from data_extraction import run_data_extraction, get_notes_mapping_dict, vectorize_notes_by_mapping
from config import data_path


def test_data_extraction():
    list_test = run_data_extraction(data_path)
    notes2idx, idx2note = get_notes_mapping_dict(list_test)
    notes_vec = vectorize_notes_by_mapping(list_test, notes2idx)

    print('end')


if __name__ == '__main__':
    test_data_extraction()
