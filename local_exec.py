from data_extraction import run_data_extraction
from config import data_path


def test_data_extraction():
    list_test = run_data_extraction(data_path)


if __name__ == '__main__':
    test_data_extraction()
