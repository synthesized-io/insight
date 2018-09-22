import numpy as np


def categorical_emd(a, b):
    space = sorted(list(set(a).union(set(b))))

    a_unique, counts = np.unique(a, return_counts=True)
    a_counts = dict(zip(a_unique, counts))

    b_unique, counts = np.unique(b, return_counts=True)
    b_counts = dict(zip(b_unique, counts))

    p = np.array([float(a_counts[x]) if x in a_counts else 0.0 for x in space])
    q = np.array([float(b_counts[x]) if x in b_counts else 0.0 for x in space])

    p /= np.sum(p)
    q /= np.sum(q)

    distances = 1 - np.eye(len(space))

    return emd(p, q, distances)