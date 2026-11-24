"""Hamiltonian cycle optimization on explicit graphs and full-instance bounds."""
import math
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy.spatial import cKDTree

from .graph import tour_edges, validate_tour


def cycle_components(size, pairs):
    """Return components as (vertex order, edge order), including parallel edges."""
    adjacent = [[] for _ in range(size)]
    for i, (a, b) in enumerate(pairs):
        adjacent[a].append((b, i))
        adjacent[b].append((a, i))
    if any(len(row) != 2 for row in adjacent):
        raise ValueError("Expected degree two at every vertex")
    seen = np.zeros(size, dtype=bool)
    components = []
    for first in range(size):
        if seen[first]:
            continue
        cities, edges = [], []
        city, previous = first, -1
        while not seen[city]:
            seen[city] = True
            cities.append(city)
            city, previous = next(item for item in adjacent[city] if item[1] != previous)
            edges.append(previous)
        components.append((np.asarray(cities), np.asarray(edges)))
    return components


@dataclass
class CycleSolution:
    selected: np.ndarray
    optimal: bool


def optimize_cycle(size, pairs, costs, incumbent, seconds, fixed=0):
    """Solve an explicit integer-weight graph by adding subtour cuts as needed.

    The leading fixed edges can include multiple paths with the same endpoints.
    On interruption, the result always includes a complete feasible incumbent.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    deadline = time.monotonic() + seconds
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    costs = np.asarray(costs, dtype=np.int64)
    best = np.asarray(incumbent, dtype=np.int64)
    best_cost = costs[best].sum()
    if len(pairs) == size:
        return CycleSolution(best, True)
    columns = np.arange(len(pairs))
    degrees = coo_matrix((np.ones(2 * len(pairs)), (pairs.T.ravel(), np.tile(columns, 2))),
                         shape=(size, len(pairs))).tocsc()
    edge_bounds = Bounds(columns < fixed, np.ones(len(columns)))
    cuts = []
    while time.monotonic() < deadline:
        constraints = [LinearConstraint(degrees, 2, 2)]
        if cuts:
            constraints.append(LinearConstraint(vstack(cuts, format="csc"), 2, np.inf))
        # The incumbent supplies a valid cutoff for the integer objective.
        constraints.append(LinearConstraint(costs[None, :], -np.inf, best_cost))
        result = milp(costs, integrality=np.ones(len(costs)), bounds=edge_bounds,
                      constraints=constraints,
                      options={"time_limit": max(.001, deadline - time.monotonic()),
                               "mip_rel_gap": 0.0})
        if result.x is None:
            break
        selected = np.flatnonzero(result.x > .5)
        if len(selected) != size or not np.array_equal(selected[:fixed], columns[:fixed]):
            break
        if np.any(np.bincount(pairs[selected].ravel(), minlength=size) != 2):
            break
        components = cycle_components(size, pairs[selected])
        if len(components) == 1:
            cost = costs[selected].sum()
            if cost <= best_cost:
                best, best_cost = selected, cost
            return CycleSolution(best, result.status == 0)
        for cities, _ in components:
            inside = np.zeros(size, dtype=bool)
            inside[cities] = True
            crossing = columns[inside[pairs[:, 0]] ^ inside[pairs[:, 1]]]
            cuts.append(coo_matrix((np.ones(len(crossing)), (np.zeros(len(crossing)), crossing)),
                                   shape=(1, len(columns))))
    return CycleSolution(best, False)


class DistanceGrid:
    """Dyadic quantization with rigorous lower and upper Euclidean edge costs."""

    def __init__(self, points, bits=26):
        # Form lattice coordinates with integer arithmetic, even for extreme offsets.
        diameter = float(np.ptp(points, axis=0).max())
        exponent = bits - math.frexp(diameter)[1] if diameter else 0
        ratios = [float(value).as_integer_ratio() for value in points.ravel()]
        scaled = [divmod(a << max(exponent, 0), b << max(-exponent, 0)) for a, b in ratios]
        integer = np.asarray([q for q, _ in scaled], dtype=object).reshape(-1, 2)
        self.points = np.asarray(integer - integer.min(axis=0), dtype=np.int64)
        self.padding = 2 if any(r for _, r in scaled) else 0
        self.exponent = exponent

    def costs(self, pairs):
        a, b = np.asarray(pairs).T
        delta = self.points[a] - self.points[b]
        squared = (delta * delta).sum(axis=1)
        root = np.sqrt(squared).astype(np.int64)
        root -= root * root > squared
        root += (root + 1) ** 2 <= squared
        return np.maximum(root - self.padding, 0), root + (root * root != squared) + self.padding

    def unscale(self, cost):
        return math.ldexp(float(cost), -self.exponent)

    def geometric_lower(self):
        # Each vertex uses two edges at least as costly as its two nearest
        # other vertices in the grid's Chebyshev metric.
        nearest, _ = cKDTree(self.points).query(self.points, k=3, p=np.inf)
        return self.unscale(int(np.maximum(nearest[:, 1:].astype(np.int64) - self.padding, 0).sum())) / 2

    def upper(self, tour):
        return self.unscale(self.costs(tour_edges(tour))[1].sum())


def small_exact_bound(grid, incumbent):
    """An exact DP lower bound for any tour different from the incumbent."""
    n = len(incumbent)
    a, b = np.indices((n, n))
    weights = grid.costs(np.column_stack((a.ravel(), b.ravel())))[0].reshape(n, n)
    same = {frozenset(edge) for edge in tour_edges(incumbent)}
    layers = {(1, 0, False): 0}
    for _ in range(1, n):
        following = {}
        for (mask, last, different), cost in layers.items():
            for city in range(1, n):
                if mask & (1 << city):
                    continue
                key = (mask | (1 << city), city, different or frozenset((last, city)) not in same)
                value = cost + int(weights[last, city])
                following[key] = min(value, following.get(key, value))
        layers = following
    alternatives = [cost + int(weights[last, 0]) for (_, last, different), cost in layers.items()
                    if different or frozenset((last, 0)) not in same]
    return grid.unscale(min(alternatives)) if alternatives else math.inf


def concorde_solve(executable, scratch, points, seconds, seed, initial=None):
    """Return a tour and the corrected global lower bound on successful proof."""
    bits = min(26, 29 - math.ceil(math.log2(len(points))))
    grid = DistanceGrid(points, bits)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="concorde-quality-") as directory:
        path = Path(directory)
        with (path / "problem.tsp").open("w") as output:
            output.write(f"NAME: problem\nTYPE: TSP\nDIMENSION: {len(points)}\n"
                         "EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n")
            output.writelines(f"{i} {x} {y}\n" for i, (x, y) in enumerate(grid.points, 1))
            output.write("EOF\n")
        command = [str(executable), "-s", str(seed % 2147483647), "-o", "result.sol"]
        if initial is not None:
            (path / "warm.sol").write_text(f"{len(initial)}\n" + " ".join(map(str, initial)) + "\n")
            command.extend(("-t", "warm.sol"))
        try:
            run = subprocess.run([*command, "problem.tsp"],
                                 cwd=path, timeout=seconds, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired:
            return None, None
        if run.returncode:
            raise RuntimeError(f"Concorde failed: {run.stdout[-1200:]}")
        result = np.fromstring((path / "result.sol").read_text(), sep=" ", dtype=np.int64)
        if len(result) != len(points) + 1 or result[0] != len(points):
            raise ValueError("Invalid Concorde solution file")
        tour = validate_tour(result[1:], len(points))
        match = re.search(r"Optimal Solution:\s*(\d+)(?:\.0+)?\s*$", run.stdout, flags=re.M)
        if not match:
            return tour, None
        integer_optimum = int(match[1])
        pairs = tour_edges(tour)
        delta = grid.points[pairs[:, 0]] - grid.points[pairs[:, 1]]
        rounded = np.floor(np.sqrt((delta * delta).sum(axis=1)) + .5).astype(np.int64)
        if int(rounded.sum()) != integer_optimum:
            raise ValueError("Concorde objective does not match the returned tour")
        # Nearest-integer edge rounding adds at most 1/2 grid unit.
        lower = grid.unscale(integer_optimum - (grid.padding + .5) * len(points))
        return tour, max(0., lower)
