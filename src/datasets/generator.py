import random


def random_dataset(size, min_value=1, max_value=100000):
    """
    Generate a random dataset of specified size.
    """
    return [
        random.randint(min_value, max_value)
        for _ in range(size)
    ]


def sorted_dataset(size):
    """
    Generate an already sorted dataset.
    """
    return list(range(size))


def reverse_sorted_dataset(size):
    """
    Generate a reverse sorted dataset.
    """
    return list(range(size, 0, -1))


def nearly_sorted_dataset(size, swaps=10):
    """
    Generate a nearly sorted dataset by performing
    a small number of random swaps.
    """
    arr = list(range(size))

    if size <= 1:
        return arr

    for _ in range(min(swaps, size)):
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)

        arr[i], arr[j] = arr[j], arr[i]

    return arr


def duplicate_heavy_dataset(size, unique_values=20):
    """
    Generate a dataset with many duplicates.
    """
    return [
        random.randint(1, unique_values)
        for _ in range(size)
    ]


def all_equal_dataset(size, value=42):
    """
    Generate a dataset where all values are identical.
    """
    return [value] * size


def empty_dataset():
    """
    Generate an empty dataset.
    """
    return []


def single_element_dataset(value=42):
    """
    Generate a dataset containing one element.
    """
    return [value]