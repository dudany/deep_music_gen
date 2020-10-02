from data_extraction import run_data_extraction, get_notes_mapping_dict
from config import data_path


def test_data_extraction():
    list_test = run_data_extraction(data_path)
    mapping = get_notes_mapping_dict(list_test)

if __name__ == '__main__':
    test_data_extraction()
