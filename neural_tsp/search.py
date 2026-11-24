import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .graph import Graph, construct_tour, route_length, validate_tour
from .model import candidate_groups, graph_inputs, ranked_candidates


def write_instance(path, points, fixed_edges=()):
    scaled = points - points.min(axis=0)
    scaled *= 1e6 / max(float(scaled.max()), 1e-12)
    with path.open("w") as fp:
        fp.write(f"NAME : instance\nTYPE : TSP\nDIMENSION : {len(points)}\n"
                 "EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
        fp.writelines(f"{i} {x:.9f} {y:.9f}\n" for i, (x, y) in enumerate(scaled, 1))
        if len(fixed_edges):
            fp.write("FIXED_EDGES_SECTION\n")
            fp.writelines(f"{a + 1} {b + 1}\n" for a, b in fixed_edges)
            fp.write("-1\n")
        fp.write("EOF\n")


def read_lkh_tour(path, size):
    text = path.read_text()
    _, marker, body = text.partition("TOUR_SECTION")
    if not marker:
        raise ValueError(f"Incomplete tour file: {path}")
    words = body.split()
    end = words.index("-1")
    return validate_tour(np.asarray([int(token) - 1 for token in words[:end]]), size)


def run_with_progress(command, *, cwd, timeout, callback=None, env=None):
    if callback is None:
        return subprocess.run(command, cwd=cwd, timeout=timeout, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, env=env)
    deadline = time.monotonic() + timeout
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, env=env) as process:
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    stdout, _ = process.communicate(timeout=min(1., remaining))
                except subprocess.TimeoutExpired:
                    callback()
                    continue
                callback()
                return subprocess.CompletedProcess(command, process.returncode, stdout)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


class LKH:
    def __init__(self, executable, scratch):
        self.executable = Path(executable).resolve(strict=True)
        if not os.access(self.executable, os.X_OK):
            raise ValueError(f"LKH is not executable: {self.executable}")
        self.scratch = Path(scratch).resolve()
        self.scratch.mkdir(parents=True, exist_ok=True)

    def run(self, points, seconds, seed, initial=None, candidates=None, trials=None, quality=False,
            on_tour=None, candidate_type=None, fixed_edges=()):
        if seconds <= 0:
            raise ValueError("LKH must receive a positive time budget")
        n = len(points)
        with tempfile.TemporaryDirectory(dir=self.scratch, prefix="lkh-") as directory:
            path = Path(directory)
            write_instance(path / "instance.tsp", points, fixed_edges)
            options = [
                "PROBLEM_FILE = instance.tsp", "OUTPUT_TOUR_FILE = result.tour",
                "TOUR_FILE = final.tour", "INITIAL_TOUR_ALGORITHM = SIERPINSKI",
                "PRECISION = 1", "RUNS = 1", "TRACE_LEVEL = 0", "STOP_AT_OPTIMUM = NO",
                f"MAX_TRIALS = {trials if trials is not None else max(20, n)}",
                f"TIME_LIMIT = {seconds:.6f}",
                f"SEED = {1 + (int(seed) % 2147483646)}",
            ]
            if quality:
                options += ["MOVE_TYPE = 5", "PATCHING_C = 3", "PATCHING_A = 2"]
                if candidates is None:
                    kind = candidate_type or ("ALPHA" if n <= 10000 else "POPMUSIC")
                    options += ["CANDIDATE_SET_TYPE = " + kind,
                                f"MAX_CANDIDATES = {min(20, n - 1)}"]
            if seconds < 3 and not quality:
                options += ["MOVE_TYPE = 3", "PATCHING_C = 0", "PATCHING_A = 0", "MAX_BREADTH = 5"]
            if not quality and candidates is None and (seconds < 3 or n > 5000):
                options += ["CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR", f"MAX_CANDIDATES = {min(12, n - 1)}"]
            if initial is not None:
                values = "\n".join(str(int(city) + 1) for city in validate_tour(initial, n))
                (path / "initial.tour").write_text(
                    f"NAME : initial\nTYPE : TOUR\nDIMENSION : {n}\nTOUR_SECTION\n{values}\n-1\nEOF\n")
                options += ["INITIAL_TOUR_FILE = initial.tour"]
            if candidates is not None:
                with (path / "candidates.txt").open("w") as fp:
                    fp.write(f"{n}\n")
                    for city, row in enumerate(candidates, 1):
                        pairs = " ".join(f"{int(other) + 1} {rank}" for rank, other in enumerate(row))
                        fp.write(f"{city} 0 {len(row)} {pairs}\n")
                    fp.write("-1\nEOF\n")
                options += ["CANDIDATE_FILE = candidates.txt", "CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR",
                            f"MAX_CANDIDATES = {max(map(len, candidates))}"]
            (path / "run.par").write_text("\n".join(options) + "\n")
            stamp = None

            def progress():
                nonlocal stamp
                result = path / "result.tour"
                try:
                    updated = result.stat().st_mtime_ns
                    if stamp != updated:
                        tour = read_lkh_tour(result, n)
                        stamp = updated
                        on_tour(tour)
                except (FileNotFoundError, ValueError):
                    pass

            try:
                process = run_with_progress([str(self.executable), "run.par"], cwd=path,
                                            timeout=seconds + max(0.2, seconds * 0.05),
                                            callback=progress if on_tour is not None else None)
            except subprocess.TimeoutExpired:
                for output in (path / "final.tour", path / "result.tour"):
                    try:
                        return read_lkh_tour(output, n)
                    except (FileNotFoundError, ValueError):
                        continue
                if initial is not None:
                    return initial.copy()
                if trials != 0:
                    return self.run(points, max(0.5, seconds), seed, trials=0, fixed_edges=fixed_edges)
                raise
            if process.returncode != 0:
                raise RuntimeError(f"LKH failed for {n} cities: {process.stdout[-2000:]}")
            for output in (path / "final.tour", path / "result.tour"):
                if output.exists():
                    return read_lkh_tour(output, n)
            if initial is not None:
                return initial.copy()
            if trials != 0:
                return self.run(points, max(0.5, seconds), seed, trials=0, fixed_edges=fixed_edges)
            raise RuntimeError(f"LKH did not return a tour for {n} cities: {process.stdout[-2000:]}")


@torch.no_grad()
def solve(points, network, device, lkh, seconds, seed=0, mode="dynamic",
          initialization="both", rounds=4, k=32, graph=None, quality=False, seed_tour=None):
    if seconds <= 0 or rounds < 1:
        raise ValueError("A positive budget and at least one round are required")
    start = time.monotonic()
    if mode == "native":
        tour = lkh.run(points, seconds, seed)
        return tour, {"seconds": time.monotonic() - start, "initial_lengths": {}, "rounds": []}

    graph = graph if graph is not None else Graph.build(points, k)
    _, edges, (edge_tensor, nodes, features, _) = graph_inputs(graph, device)
    h = network.encode(edge_tensor, nodes, features)
    logits = network.edge_outputs(edge_tensor, h, features).cpu().numpy()
    tours = {}
    if initialization in ("both", "neural"):
        tours["neural"] = construct_tour(points, edges, logits)
    if initialization in ("both", "lkh"):
        tours["lkh"] = lkh.run(points, min(1.0, max(0.05, seconds / 5)), seed, trials=1)
    if seed_tour is not None:
        tours["warm_start"] = validate_tour(seed_tour, len(points))
    initial_lengths = {name: route_length(points, tour) for name, tour in tours.items()}
    best = tours[min(initial_lengths, key=initial_lengths.get)]
    length = route_length(points, best)
    history = []
    rounds = min(rounds, max(1, int(seconds / 0.25)))

    for step in range(rounds):
        remaining = seconds - (time.monotonic() - start)
        if remaining <= 0:
            break
        _, edges, (edge_tensor, _, features, state) = graph_inputs(graph, device, best)
        logits = network.edge_outputs(edge_tensor, h, features).cpu().numpy()
        ranks = candidate_groups(edges, features.cpu().numpy(), state.cpu().numpy(), logits)
        groups = [ranked_candidates(edges, rank, graph.neighbors, best) for rank in ranks]
        if mode == "dynamic":
            representations = network.edge_outputs(edge_tensor, h, features, state)
            choice = int(network.group_scores(representations, state, [g[1] for g in groups]).argmax())
        elif mode == "random":
            choice = int(np.random.default_rng(seed + step).integers(len(groups)))
        else:
            choice = 0
        remaining = seconds - (time.monotonic() - start)
        if remaining <= 0:
            break
        tour = lkh.run(points, remaining / (rounds - step), seed + step, initial=best,
                       candidates=None if mode == "unguided" else groups[choice][0], quality=quality)
        new_length = route_length(points, tour)
        if new_length < length:
            best, length = tour, new_length
        history.append({"candidate_group": choice, "best_length": length,
                        "seconds": time.monotonic() - start})
    return best, {"seconds": time.monotonic() - start, "initial_lengths": initial_lengths,
                  "rounds": history}
