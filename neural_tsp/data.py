import hashlib
from pathlib import Path

import numpy as np

from .graph import route_length, validate_tour


def instances(path, tsplib=False):
    with Path(path).open() as handle:
        for index, raw in enumerate(handle):
            line = raw.strip()
            if not line:
                continue
            if tsplib:
                text = line.translate(str.maketrans("", "", "[]'"))
                name, reference, coords = text.split(",", 2)
                points = np.fromstring(coords, sep=",").reshape(-1, 2)
                yield name, points, float(reference), None
            else:
                coords, target = line.split("output", 1)
                points = np.fromstring(coords, sep=" ").reshape(-1, 2)
                tour = np.fromstring(target, sep=" ", dtype=np.int64) - 1
                if len(tour) == len(points) + 1 and tour[0] == tour[-1]:
                    tour = tour[:-1]
                tour = validate_tour(tour, len(points))
                yield str(index), points, route_length(points, tour), tour


def instance_key(points):
    # Invariant to indexing, and independent of file name and reference tour.
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    return hashlib.sha256(points.astype("<f8").tobytes()).hexdigest()[:20]


def synthetic(seed, size, distribution):
    rng = np.random.default_rng(seed)
    if distribution == 0:
        points = rng.random((size, 2))
    elif distribution == 1:
        centers = rng.random((8, 2))
        points = centers[rng.integers(len(centers), size=size)] + rng.normal(0, 0.045, (size, 2))
    elif distribution == 2:
        points = rng.random((size, 2))
        points[:, 0] **= 3
    elif distribution == 3:
        points = rng.random((size, 2))
        points[:, 1] *= 0.15
    else:
        raise ValueError("Expected one of four distributions")
    points -= points.min(axis=0)
    return points / max(float(points.max()), 1e-12)
