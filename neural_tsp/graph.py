from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, cKDTree

try:
    from scipy.spatial import QhullError
except ImportError:  # SciPy 1.6 in the course container
    from scipy.spatial.qhull import QhullError


def route_length(points, tour):
    ordered = points[tour]
    return float(np.linalg.norm(ordered - np.roll(ordered, -1, axis=0), axis=1).sum())


def validate_tour(tour, size):
    tour = np.asarray(tour)
    if tour.shape != (size,) or tour.dtype.kind not in "iu":
        raise ValueError("Expected a one-dimensional integer tour")
    if np.any(tour < 0) or np.any(tour >= size) or np.unique(tour).size != size:
        raise ValueError("Tour must visit each city once")
    return tour


def edge_keys(pairs, size):
    pairs = np.asarray(pairs).reshape(-1, 2)
    return pairs.min(axis=1) * size + pairs.max(axis=1)


def tour_edges(tour):
    return np.column_stack((tour, np.roll(tour, -1)))


@dataclass
class Graph:
    points: np.ndarray
    neighbors: np.ndarray
    distances: np.ndarray
    keys: np.ndarray

    @classmethod
    def build(cls, points, k=32):
        points = np.asarray(points, dtype=np.float64)
        n = len(points)
        if points.shape != (n, 2) or n < 3 or not np.isfinite(points).all():
            raise ValueError("Expected at least three finite 2D points")
        if k < 1:
            raise ValueError("At least one nearest neighbor is required")
        width = min(k, n - 1)
        _, indices = cKDTree(points).query(points, k=width + 1)
        # Ties from duplicate points can place the queried city anywhere or omit it.
        keep = indices != np.arange(n)[:, None]
        take = keep & (np.cumsum(keep, axis=1) <= width)
        neighbors = indices[take].reshape(n, width)
        distances = np.linalg.norm(points[:, None] - points[neighbors], axis=2)
        pairs = [np.column_stack((np.repeat(np.arange(n), width), neighbors.ravel()))]
        try:
            triangles = Delaunay(points).simplices
            pairs.extend(triangles[:, [a, b]] for a, b in ((0, 1), (0, 2), (1, 2)))
        except QhullError:
            pass
        # These edges guarantee a legal geometric fallback through every city.
        pairs.append(tour_edges(np.lexsort((points[:, 1], points[:, 0]))))
        keys = np.unique(edge_keys(np.concatenate(pairs), n))
        return cls(points, neighbors, distances, keys)

    def features(self, tour=None):
        n, width = self.neighbors.shape
        keys = self.keys
        if tour is not None:
            keys = np.union1d(keys, edge_keys(tour_edges(tour), n))
        a, b = np.divmod(keys, n)
        distances = np.linalg.norm(self.points[a] - self.points[b], axis=1)
        scale = max(float(np.mean(self.distances)), 1e-12)
        local = np.maximum(self.distances.mean(axis=1), scale * 1e-4)
        directed = np.arange(n)[:, None] * n + self.neighbors
        order = np.argsort(directed.ravel())
        sorted_keys = directed.ravel()[order]
        ranks = np.broadcast_to(np.arange(1, width + 1), directed.shape).ravel()[order]

        def rank(left, right):
            query = left * n + right
            idx = np.minimum(np.searchsorted(sorted_keys, query), len(sorted_keys) - 1)
            return np.where(sorted_keys[idx] == query, ranks[idx], width + 1) / (width + 1)

        r1, r2 = rank(a, b), rank(b, a)
        ratio_a, ratio_b = distances / local[a], distances / local[b]
        features = np.column_stack((np.log1p(distances / scale), np.minimum(r1, r2),
                                    np.maximum(r1, r2), np.log1p(np.minimum(ratio_a, ratio_b)),
                                    np.log1p(np.maximum(ratio_a, ratio_b))))
        state = np.zeros((len(keys), 4), dtype=np.float32)
        if tour is not None:
            position = np.empty(n, dtype=np.int64)
            position[tour] = np.arange(n)
            offset = np.abs(position[a] - position[b])
            incident = np.empty(n)
            step = np.linalg.norm(self.points[tour] - self.points[np.roll(tour, -1)], axis=1)
            incident[tour] = np.maximum(step, np.roll(step, 1))
            state[:, 0] = offset == 1
            state[offset == n - 1, 0] = 1
            state[:, 1] = 2 * np.minimum(offset, n - offset) / n
            state[:, 2] = np.log1p(np.minimum(incident[a], incident[b]) / scale)
            state[:, 3] = np.log1p(np.maximum(incident[a], incident[b]) / scale)
        nodes = np.log1p(local / scale).astype(np.float32)[:, None]
        return keys, np.stack((a, b)), nodes, features.astype(np.float32), state


def construct_tour(points, edges, scores):
    """Greedy path merging, followed by nearest endpoint completion."""
    n = len(points)
    parent = list(range(n))
    adjacency = [[] for _ in range(n)]
    used = 0

    def root(city):
        while parent[city] != city:
            parent[city] = parent[parent[city]]
            city = parent[city]
        return city

    def join(a, b):
        nonlocal used
        left, right = root(a), root(b)
        if left == right or len(adjacency[a]) == 2 or len(adjacency[b]) == 2:
            return
        parent[right] = left
        adjacency[a].append(b)
        adjacency[b].append(a)
        used += 1

    for idx in np.argsort(-scores, kind="stable"):
        join(*edges[:, idx])
        if used == n - 1:
            break

    # Space-ordered endpoint pairs guarantee completion in O(n log n) space/time
    # outside the union-find operations, even on completely degenerate inputs.
    if used < n - 1:
        endpoints = np.flatnonzero([len(row) < 2 for row in adjacency])
        order = endpoints[np.lexsort((points[endpoints, 1], points[endpoints, 0]))]
        tree = cKDTree(points[order])
        widths = min(8, len(order))
        _, neighbors = tree.query(points[order], k=widths)
        for city, row in zip(order, neighbors):
            for other in order[row[1:]]:
                join(city, other)
                if len(adjacency[city]) == 2:
                    break
        if used < n - 1:
            components = {}
            for city in order:
                if len(adjacency[city]) < 2:
                    components.setdefault(root(city), []).append(city)
            paths = iter(components.values())
            tail = next(paths)[-1]
            for path in paths:
                join(tail, path[0])
                tail = path[-1]

    tour = [next(i for i, row in enumerate(adjacency) if len(row) == 1)]
    while len(tour) < n:
        previous = tour[-2] if len(tour) > 1 else -1
        tour.append(next(city for city in adjacency[tour[-1]] if city != previous))
    return np.asarray(tour, dtype=np.int64)
