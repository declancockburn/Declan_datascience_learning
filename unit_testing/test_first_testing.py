import pytest
import numpy as np
from main import return_greeting, split_array_in_two


def test_print_hi():
    assert return_greeting("Declan") == "Hi, Declan"


def test_split_array_works():
    example_data = np.array([1,2,3,4,5,6])
    actual = split_array_in_two(example_data)
    expected = np.array([[1,2,3],
                        [4,5,6]])
    assert actual == pytest.approx(expected), f"Expected {expected}, got {actual}"

    example_data_str = str(example_data)
    # Todo: expand on this, what's it used for etc.
    with pytest.raises(ValueError):
        split_array_in_two(example_data_str)


test_split_array_works()