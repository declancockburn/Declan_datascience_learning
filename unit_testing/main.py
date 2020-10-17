import pandas as pd
import numpy as np
import pytest


def return_greeting(name):
    return f'Hi, {name}'

def split_array_in_two(a):
    if not isinstance(a, np.ndarray):
        raise ValueError
    halfway = len(a) // 2
    return np.array([a[:halfway], a[halfway:]])


# split_array_in_two(np.array([1,2,3,4,5,6]))
# split_array_in_two("np.array([1,2,3,4,5,6])")



if __name__ == '__main__':
    exit_code = pytest.main()

    return_greeting('PyCharm')


