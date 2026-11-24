"""Build a large-instance incumbent by solving and joining spatial subproblems."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import subprocess

import numpy as np
from scipy.spatial import cKDTree

from .graph import route_length


def split(points, cities, limit, leaves):
    if len(cities) <= limit:
        leaves.append(cities)
        return len(leaves) - 1
    coords = points[cities]
    axis = int(np.argmax(np.ptp(coords, axis=0)))
    midpoint = len(cities) // 2
    order = np.argpartition(coords[:, axis], midpoint)
    return (split(points, cities[order[:midpoint]], limit, leaves),
            split(points, cities[order[midpoint:]], limit, leaves))


def join(points, first, second):
    """Join two cycles by cutting one edge in each and reconnecting the paths."""
    width = min(4, len(second))
    _, nearest = cKDTree(points[second]).query(points[first], k=width)
    a = np.repeat(np.arange(len(first)), width)
    b = nearest.reshape(-1)
    x, y = points[first[a]], points[second[b]]
    bridge = np.linalg.norm(x - y, axis=1)
    best_cost = np.inf
    for direction_a in (-1, 1):
        xa = points[first[(a + direction_a) % len(first)]]
        cost_a = np.linalg.norm(x - xa, axis=1)
        for direction_b in (-1, 1):
            yb = points[second[(b + direction_b) % len(second)]]
            change = bridge + np.linalg.norm(xa - yb, axis=1) - cost_a - np.linalg.norm(y - yb, axis=1)
            index = int(np.argmin(change))
            if change[index] < best_cost:
                best_cost = change[index]
                cut_a, cut_b = a[index], b[index]
                step_a, step_b = direction_a, direction_b
    # The first path ends at cut_a and the second begins at cut_b.
    path_a = first[(cut_a - step_a * np.arange(len(first) - 1, -1, -1)) % len(first)]
    path_b = second[(cut_b - step_b * np.arange(len(second))) % len(second)]
    return np.concatenate((path_a, path_b))


def solve_leaf(points, lkh, seed, seconds, item):
    index, cities = item
    if len(cities) <= 4:
        return cities
    coords = points[cities]
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    try:
        return cities[lkh.run(coords, seconds, seed + index, quality=True)]
    except (RuntimeError, ValueError, subprocess.TimeoutExpired):
        return cities[order]


def partition_tour(points, lkh, seconds, seed, limit=768, workers=8):
    leaves = []
    tree = split(points, np.arange(len(points)), limit, leaves)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tours = list(pool.map(partial(solve_leaf, points, lkh, seed, seconds), enumerate(leaves)))

    def stitch(node):
        if isinstance(node, int):
            return tours[node]
        return join(points, stitch(node[0]), stitch(node[1]))

    return stitch(tree)


def polish_tour(points, tour, max_passes=16):
    """Apply batches of disjoint improving 2-opt moves using geographic neighbors."""
    n = len(tour)
    _, neighbor = cKDTree(points).query(points, k=min(12, n))
    best = tour.copy()
    positions = np.arange(n)
    starts = np.broadcast_to(positions[:, None], neighbor.shape).ravel()
    for _ in range(max_passes):
        position = np.empty(n, dtype=np.int64)
        position[best] = positions
        ends = position[neighbor[best]]
        forward = np.column_stack((starts, ends.ravel()))
        pairs = np.sort(np.vstack((forward, (forward - 1) % n)), axis=1)
        first, second = pairs.T
        gap = second - first
        first, second = first[(gap > 1) & (gap < n - 1)], second[(gap > 1) & (gap < n - 1)]
        coords = points[best]
        outgoing = np.roll(coords, -1, axis=0)
        removed = np.linalg.norm(coords - outgoing, axis=1)
        change = (removed[first] + removed[second] - np.linalg.norm(coords[first] - coords[second], axis=1)
                  - np.linalg.norm(outgoing[first] - outgoing[second], axis=1))
        improving = change > 1e-12
        if not improving.any():
            break
        first, second, change = first[improving], second[improving], change[improving]
        order = np.argsort(second, kind="stable")
        first, second, change = first[order], second[order], change[order]
        previous = np.searchsorted(second, first)
        total = np.zeros(len(order) + 1)
        take = np.zeros(len(order), dtype=bool)
        for i, gain in enumerate(change):
            with_move = total[previous[i]] + gain
            take[i] = with_move > total[i]
            total[i + 1] = with_move if take[i] else total[i]
        end = len(order)
        while end:
            i = end - 1
            if take[i]:
                path = best[first[i] + 1:second[i] + 1]
                path[:] = path[::-1]
                end = previous[i]
            else:
                end -= 1
    return best


def solve_window(points, lkh, seconds, seed, item):
    index, cities = item
    size = len(cities)
    if size <= 4:
        return cities
    coords = points[cities]
    initial = np.arange(size)
    try:
        local = lkh.run(coords, seconds, seed + index, initial=initial, quality=True,
                        fixed_edges=((0, size - 1),))
    except (RuntimeError, ValueError, subprocess.TimeoutExpired):
        return cities
    local = np.roll(local, -int(np.flatnonzero(local == 0)[0]))
    if local[1] == size - 1:
        local[1:] = local[:0:-1]
    if local[-1] == size - 1 and route_length(coords, local) < route_length(coords, initial):
        return cities[local]
    return cities


def polish_windows(points, tour, lkh, seconds, seed, size=768, workers=8):
    """Optimize overlapping sections while preserving each section's endpoints."""
    current = tour
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in (0, size // 2):
            rotated = np.roll(current, offset)
            chunks = [rotated[start:start + size] for start in range(0, len(tour), size)]
            fixed = pool.map(partial(solve_window, points, lkh, seconds, seed + offset),
                             enumerate(chunks))
            current = np.roll(np.concatenate(list(fixed)), -offset)
    return current
