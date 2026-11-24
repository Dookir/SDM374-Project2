"""Quality-first TSP portfolio, tour merging, and learned regional repair."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .exact import DistanceGrid, concorde_solve, cycle_components, optimize_cycle, small_exact_bound
from .graph import edge_keys, route_length, tour_edges, validate_tour
from .partition import partition_tour, polish_tour, polish_windows
from .regions import TourContext, predict
from .search import solve


class Portfolio:
    def __init__(self, points, save=None, capacity=16):
        self.points = points
        self.size = len(points)
        self.capacity = capacity
        self.save = save
        self.history = []
        self.elites = []
        self.started = time.monotonic()

    def add(self, tour, source):
        tour = validate_tour(tour, self.size)
        length = route_length(self.points, tour)
        keys = np.sort(edge_keys(tour_edges(tour), self.size))
        if any(np.array_equal(keys, row[2]) for row in self.elites):
            return
        improved = not self.elites or length < self.elites[0][0]
        self.elites.append((length, tour, keys))
        self.elites.sort(key=lambda row: row[0])
        if len(self.elites) > self.capacity:
            # Keep the best and reward uncommon edges among other high-quality tours.
            union, counts = self.support()
            best_length = max(self.elites[0][0], 1e-12)
            def redundancy(row):
                relative_loss = (row[0] - best_length) / best_length
                frequency = counts[np.searchsorted(union, row[2])].mean() / len(self.elites)
                return frequency + 30 * relative_loss
            removed = max(range(1, len(self.elites)), key=lambda index: redundancy(self.elites[index]))
            self.elites.pop(removed)
        if improved:
            seconds = time.monotonic() - self.started
            self.history.append({"source": source, "length": length, "seconds": seconds})
            if self.save is not None:
                self.save(tour, length, self.history)

    @property
    def best(self):
        length, tour, _ = self.elites[0]
        return tour, length

    def support(self):
        return np.unique(np.concatenate([row[2] for row in self.elites]), return_counts=True)

    def context(self, tree=None, tour=None):
        keys, counts = self.support()
        return TourContext(self.points, self.best[0] if tour is None else tour, keys, counts,
                           len(self.elites), tree)


def collect_parents(portfolio, lkh, network, device, eax, neurolkh, seconds, seeds, seed):
    points = portfolio.points
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    portfolio.add(order, "geometric")
    if n <= 4 or np.all(points == points[0]):
        return []
    if n >= 10000:
        warm = partition_tour(points, lkh, min(seconds, 4.), seed)
        portfolio.add(warm, "partition_lkh")
        portfolio.add(polish_tour(points, warm), "partition_2opt")
        windowed = polish_windows(points, portfolio.best[0], lkh, min(seconds, 1.5), seed)
        portfolio.add(windowed, "partition_windows")
    errors = []
    native_progress = eax_progress = None
    if portfolio.save is not None:
        native_progress = lambda tour: portfolio.add(tour, "lkh_progress")
        eax_progress = lambda tour: portfolio.add(tour, "eax_progress")
    parent_seed = np.random.SeedSequence(seed).generate_state(seeds)
    for index, random_seed in enumerate(parent_seed):
        random_seed = int(random_seed) % 2147483547
        transform = points[:, ::-1] if index % 2 else points
        transform = transform * np.array([(-1) ** (index // 2 % 2), (-1) ** (index // 4 % 2)])
        methods = [("lkh", lambda: lkh.run(transform, seconds, random_seed, quality=True,
                                         on_tour=native_progress,
                                         initial=portfolio.best[0] if n >= 10000 else None,
                                         candidate_type="NEAREST-NEIGHBOR" if n >= 10000 and index % 2 == 0 else None))]
        if n >= 10000:
            methods.insert(0, ("lkh_independent", lambda: lkh.run(
                transform, seconds, random_seed, candidate_type="NEAREST-NEIGHBOR",
                quality=True, on_tour=native_progress)))
        if network is not None:
            methods.append(("static", lambda: solve(transform, network, device, lkh, seconds,
                                                    random_seed, "static", "neural", 1,
                                                    quality=True,
                                                    seed_tour=portfolio.best[0] if n >= 10000 else None)[0]))
        if neurolkh is not None and n > 20:
            methods.append(("neurolkh", lambda: neurolkh.run(transform, seconds, random_seed)))
        if eax is not None and n > 50:
            methods.append(("eax", lambda: eax.run(transform, seconds, seed=random_seed,
                                                   on_tour=eax_progress)))
        for name, method in methods:
            try:
                portfolio.add(method(), name)
            except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
                errors.append({"source": name, "seed": random_seed, "error": str(error)[-250:]})
    threshold = portfolio.best[1] * 1.03
    portfolio.elites = [row for row in portfolio.elites if row[0] <= threshold]
    return errors


def make_progress(path, identity):
    def save(tour, length, history):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"id": identity, "tour": tour.tolist(), "length": length,
                                         "history": history}))
        temporary.replace(path)
    return save


def merge(portfolio, grid, seconds):
    keys, _ = portfolio.support()
    tour, _ = portfolio.best
    incumbent = np.searchsorted(keys, edge_keys(tour_edges(tour), portfolio.size))
    pairs = np.column_stack(np.divmod(keys, portfolio.size))
    solution = optimize_cycle(portfolio.size, pairs, grid.costs(pairs)[0], incumbent, seconds)
    ((merged, _),) = cycle_components(portfolio.size, pairs[solution.selected])
    portfolio.add(merged, "tour_merge")
    return solution.optimal


def quality_solve(points, network, region_model, device, lkh, eax, neurolkh, concorde, scratch,
                  seconds, seed, *, seeds=2, region_count=32, attempts=8, passes=3,
                  local_seconds=1., merge_seconds=20., exact_seconds=60., exact_limit=1200,
                  policy="hybrid", save=None, initial=None):
    start = time.monotonic()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Expected at least three finite 2D points")
    portfolio = Portfolio(points, save)
    if initial is not None:
        portfolio.add(initial, "resume")
    errors = collect_parents(portfolio, lkh, network, device, eax, neurolkh, seconds, seeds, seed)
    parents_length = portfolio.best[1]
    grid = DistanceGrid(points)
    merged_optimal = merge(portfolio, grid, merge_seconds)
    union_length = portfolio.best[1]
    best_gap_bound = grid.geometric_lower()
    bound_source = "two-nearest"
    count = 0
    local_optimal = 0
    proposal_improvements = 0
    if len(points) > 3 and union_length > 0:
        tree = cKDTree(points)
        stalls = 0
        while (stalls < 2 if passes == 0 else count < passes):
            tour, length = portfolio.best
            regions = list(portfolio.context(tree).regions(region_count, seed + count))
            if policy in ("model", "hybrid"):
                scores, links = predict(region_model, regions, device)
                order = np.argsort(-scores, kind="stable")[:attempts]
                if policy == "hybrid":
                    other = region_order(regions, "disagreement", seed + count)[:attempts]
                    order = list(dict.fromkeys([*order, *other]))
            else:
                order = region_order(regions, policy, seed + count)[:attempts if policy != "oracle" else None]
            for i in order:
                region = regions[i]
                # Keep an unrestricted search alongside the learned edge proposals.
                if policy in ("model", "hybrid"):
                    proposal = region.repair(local_seconds, length, region.extras(links[i], region.size))
                    proposal_improvements += int(proposal.length < length)
                    portfolio.add(proposal.tour, "learned_links")
                result = region.repair(local_seconds, length)
                portfolio.add(result.tour, "region_" + policy)
                local_optimal += int(result.proven)
            if policy == "hybrid" and portfolio.best[1] >= length:
                remaining = sorted(set(range(len(regions))).difference(order))
                for i in remaining:
                    result = regions[i].repair(local_seconds, length)
                    portfolio.add(result.tour, "region_sweep")
                    local_optimal += int(result.proven)
            count += 1
            stalls = stalls + 1 if portfolio.best[1] >= length else 0
    if concorde is not None and len(points) <= exact_limit and len(points) > 3 and union_length > 0:
        try:
            tour, lower = concorde_solve(Path(concorde).resolve(), scratch, points,
                                         exact_seconds, seed, portfolio.best[0])
            if tour is not None:
                portfolio.add(tour, "concorde")
            if lower is not None and lower > best_gap_bound:
                best_gap_bound, bound_source = lower, "concorde"
            if tour is None:
                errors.append({"source": "concorde", "error": "time limit reached"})
        except (RuntimeError, ValueError, OSError) as error:
            errors.append({"source": "concorde", "error": str(error)[-250:]})
    tour, length = portfolio.best
    upper = grid.upper(tour)
    certified = best_gap_bound == upper
    if len(points) <= 11 and not certified:
        certified = small_exact_bound(grid, tour) >= upper
        if certified:
            # This proves the returned route is optimal. The original length
            # still lies in the interval between its lower and upper edge sums.
            best_gap_bound = max(best_gap_bound, grid.unscale(grid.costs(tour_edges(tour))[0].sum()))
            bound_source = "integer-dynamic-programming"
    return tour, {"seconds": time.monotonic() - start, "parent_length": parents_length,
                  "union_length": union_length, "union_quantized_optimal": merged_optimal,
                  "passes": count, "proven_regions": local_optimal,
                  "proposal_improvements": proposal_improvements, "lower_bound": best_gap_bound,
                  "upper_bound": upper, "bound_source": bound_source, "exact_optimal": certified,
                  "optimality_gap_upper_percent": (100 * (upper / best_gap_bound - 1)
                                                   if best_gap_bound > 0 else None),
                  "history": portfolio.history, "solver_errors": errors}


def region_order(regions, policy, seed):
    if policy == "random":
        return np.random.default_rng(seed).permutation(len(regions))
    if policy == "disagreement":
        def score(region):
            support = region.support[region.incumbent[region.fixed:]]
            return float(np.sum(1 - support))
        return np.argsort([-score(region) for region in regions], kind="stable")
    return np.arange(len(regions))
