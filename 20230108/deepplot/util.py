import numpy as np


def max_array(array_list):
    max_merge = [max(arr[i] for arr in array_list) for i in range(len(array_list[0]))]
    return np.array(max_merge)
