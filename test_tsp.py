"""Project 2 evaluator for the synthetic and TSPLib files used by TTPL."""
import argparse
import json
import math
import os
import subprocess
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from neural_tsp.competitors import EAX, NeuroLKH
from neural_tsp.data import instance_key, instances, synthetic
from neural_tsp.graph import Graph, route_length, validate_tour
from neural_tsp.model import EdgeNetwork, graph_inputs
from neural_tsp.quality import make_progress, quality_solve
from neural_tsp.search import LKH, solve
from neural_tsp.train_regions import load_model


DATA_FILES = {
    0: "TSPlib_scale_ge_1K_n33_ascending.txt",
    1000: "MCTS_tsp1000_test_concorde.txt",
    5000: "test_tsp5000_lkh3_n16.txt",
    10000: "MCTS_tsp10000_test_concorde.txt",
    50000: "test_tsp50000_lkh3_n16.txt",
    100000: "test_tsp100000_lkh3_n16.txt",
}


def flag(value):
    value = value.lower()
    if value not in ("true", "false"):
        raise argparse.ArgumentTypeError("Expected true or false")
    return value == "true"


def summarize(rows, tsplib, baseline=None):
    if not rows:
        raise ValueError("The selected dataset is empty")
    mean_length = float(np.mean([row["length"] for row in rows]))
    mean_reference = float(np.mean([row["reference"] for row in rows]))
    average_gap = float(np.mean([row["gap_percent"] for row in rows]))
    result = {
        "instances": len(rows),
        "mean_length": mean_length,
        "mean_reference": mean_reference,
        "avg_aug_gap": average_gap if tsplib or mean_reference == 0 else 100 * (mean_length / mean_reference - 1),
        "mean_instance_gap_percent": average_gap,
        "total_seconds": sum(row["seconds"] for row in rows),
    }
    if baseline is not None:
        by_key = {row["id"]: row for row in baseline}
        if len(by_key) != len(baseline) or not all(row["id"] in by_key for row in rows):
            raise ValueError("Baseline records must match the evaluated instances by coordinate hash")
        differences = [row["length"] - by_key[row["id"]]["length"] for row in rows]
        wins, ties = sum(x < -1e-8 for x in differences), sum(abs(x) <= 1e-8 for x in differences)
        result.update(wins=wins, ties=ties, losses=len(rows) - wins - ties,
                      win_rate=wins / len(rows), required_wins=math.ceil(0.70 * len(rows)))
    if "lower_bound" in rows[0]:
        bounded = [row["optimality_gap_upper_percent"] for row in rows
                   if row.get("optimality_gap_upper_percent") is not None]
        result.update(exact_optimal=sum(row["exact_optimal"] for row in rows),
                      bounded_instances=len(bounded),
                      mean_gap_upper_percent=float(np.mean(bounded)) if bounded else None)
    return result


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--dataset-root", type=Path)
    parser.add_argument("--problem_size", type=int, default=1000)
    parser.add_argument("--test_in_tsplib", type=flag, default=False)
    parser.add_argument("--lkh", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--eax")
    parser.add_argument("--neurolkh")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("native", "eax", "neurolkh", "unguided", "static",
                                           "random", "dynamic", "quality"), default="dynamic")
    parser.add_argument("--initialization", choices=("neural", "lkh", "both"), default="both")
    parser.add_argument("--seconds", type=float, help="Defaults to 30 in quality mode and 2.16 otherwise")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--augmentations", choices=(1, 8), type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--baseline-jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--region-checkpoint", type=Path)
    parser.add_argument("--concorde", type=Path)
    parser.add_argument("--quality-policy", choices=("hybrid", "model", "random", "disagreement", "oracle"),
                        default="hybrid")
    parser.add_argument("--quality-seeds", type=int, default=2)
    parser.add_argument("--quality-passes", type=int, default=3, help="Zero stops after two consecutive stalls")
    parser.add_argument("--quality-regions", type=int, default=32)
    parser.add_argument("--quality-attempts", type=int, default=8)
    parser.add_argument("--local-seconds", type=float, default=1)
    parser.add_argument("--merge-seconds", type=float, default=20)
    parser.add_argument("--exact-seconds", type=float, default=60)
    parser.add_argument("--exact-limit", type=int, default=1200)
    args = parser.parse_args()
    if args.seconds is None:
        args.seconds = 30.0 if args.mode == "quality" else 2.16
    if args.seconds <= 0 or args.rounds < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Use a positive budget, round count, and optional instance limit")
    if args.mode in ("eax", "neurolkh") and not getattr(args, args.mode):
        parser.error(f"The selected solver requires --{args.mode}")
    if args.mode not in ("native", "eax", "neurolkh", "quality") and not args.checkpoint:
        parser.error("The selected solver requires --checkpoint")
    if args.mode == "quality":
        if args.quality_policy in ("hybrid", "model") and args.region_checkpoint is None:
            parser.error("The selected policy requires --region-checkpoint")
        if args.quality_passes < 0 or min(args.quality_seeds, args.quality_regions, args.quality_attempts,
                                          args.local_seconds, args.merge_seconds, args.exact_seconds,
                                          args.exact_limit) <= 0:
            parser.error("Invalid quality search settings")
    if args.dataset_root and args.problem_size not in DATA_FILES:
        parser.error("Unsupported TTPL dataset size")
    data = args.dataset or args.dataset_root / DATA_FILES[args.problem_size]
    torch.set_num_threads(2)
    lkh = LKH(args.lkh, args.scratch)
    network = None
    device = torch.device(args.device)
    if args.checkpoint:
        network = EdgeNetwork().to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        network.load_state_dict(checkpoint["network"])
        if args.mode == "dynamic" and checkpoint["stage"] != "dynamic":
            parser.error("Dynamic mode requires a trained repair head")
        network.eval()

    eax = EAX(args.eax, args.scratch) if args.eax and args.mode in ("eax", "quality") else None
    neurolkh = NeuroLKH(args.neurolkh, args.scratch) if args.neurolkh and args.mode in ("neurolkh", "quality") else None
    region_model = load_model(args.region_checkpoint, device) if args.mode == "quality" and args.region_checkpoint else None
    warmup_start = time.monotonic()
    if network is not None:
        with torch.no_grad():
            warm_points = synthetic(0, 64, 0)
            graph = Graph.build(warm_points)
            _, _, (edges, nodes, features, state) = graph_inputs(graph, device)
            embedding = network.encode(edges, nodes, features)
            network.edge_outputs(edges, embedding, features)
            if args.mode == "dynamic":
                values = network.edge_outputs(edges, embedding, features, state)
                network.group_scores(values, state, [np.arange(min(8, len(features)))])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    warmup_seconds = time.monotonic() - warmup_start

    baseline = None
    if args.baseline_jsonl:
        baseline = [json.loads(line) for line in args.baseline_jsonl.read_text().splitlines()]
    tracked_cuda = any(model is not None for model in (network, neurolkh, region_model)) and device.type == "cuda"
    if tracked_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    rows = []
    timed_out = []
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / "routes.in-progress.jsonl"
    config = {key: os.fspath(value) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key != "resume"}
    progress_config = args.output / "run-config.json"
    if args.resume and progress_config.exists():
        saved_config = json.loads(progress_config.read_text())
        if any(config.get(key) != value for key, value in saved_config.items()):
            parser.error("Resume requires the original benchmark arguments")
    incumbent_dir = args.output / "incumbents"
    if args.mode == "quality":
        incumbent_dir.mkdir(exist_ok=True)
    completed = []
    if args.resume and temporary.exists():
        completed = [json.loads(line) for line in temporary.read_text().splitlines()]
        if any(row.get("id") is None for row in completed):
            parser.error("Cannot resume progress without per-instance IDs")
        rows = [row for row in completed if row.get("status") != "timeout"]
        timed_out = [row for row in completed if row.get("status") == "timeout"]
    already_done = len(completed)
    progress_config.write_text(json.dumps(config, indent=2) + "\n")
    with temporary.open("a" if already_done else "w") as output:
        for index, (name, coords, reference, _) in enumerate(islice(instances(data, args.test_in_tsplib), args.limit)):
            if index < already_done:
                if completed[index]["id"] != instance_key(coords):
                    raise ValueError("Saved results and dataset order differ")
                continue
            best, elapsed = None, 0.0
            for view in range(args.augmentations):
                started = time.monotonic()
                transformed = coords[:, ::-1] if view & 4 else coords
                transformed = transformed * np.array([-1 if view & 1 else 1, -1 if view & 2 else 1])
                seed = args.seed + index * 101 + view
                budget = args.seconds / args.augmentations
                try:
                    if args.mode == "quality":
                        identity = instance_key(coords)
                        path = incumbent_dir / f"{index:05d}-view{view}.json"
                        initial = None
                        if args.resume and path.exists():
                            progress = json.loads(path.read_text())
                            if progress["id"] != identity:
                                raise ValueError("Saved incumbent does not match the instance")
                            initial = np.asarray(progress["tour"])
                        tour, info = quality_solve(
                            transformed, network, region_model, device, lkh, eax, neurolkh,
                            args.concorde, lkh.scratch, args.seconds, seed, seeds=args.quality_seeds,
                            region_count=args.quality_regions, attempts=args.quality_attempts,
                            passes=args.quality_passes, local_seconds=args.local_seconds,
                            merge_seconds=args.merge_seconds, exact_seconds=args.exact_seconds,
                            exact_limit=args.exact_limit, policy=args.quality_policy,
                            save=make_progress(path, identity), initial=initial)
                    elif eax is not None:
                        tour, info = eax.run(transformed, budget), {}
                    elif neurolkh is not None:
                        tour, info = neurolkh.run(transformed, budget, seed), {}
                    else:
                        tour, info = solve(transformed, network, device, lkh, budget,
                                           seed, args.mode, args.initialization, args.rounds)
                except subprocess.TimeoutExpired:
                    elapsed += time.monotonic() - started
                    continue
                tour = validate_tour(tour, len(coords))
                length = route_length(coords, tour)
                elapsed += time.monotonic() - started
                if best is None or length < best["length"]:
                    best = {
                        **info, "id": instance_key(coords), "name": name, "length": length,
                        "reference": reference, "tour": tour.tolist(), "augmentation": view,
                        "gap_percent": 100 * (length / reference - 1) if reference > 0 else 0.0,
                    }
            if best is None:
                missed = {"name": name, "id": instance_key(coords), "seconds": elapsed, "status": "timeout"}
                timed_out.append(missed)
                output.write(json.dumps(missed) + "\n")
                output.flush()
                continue
            best["seconds"] = elapsed
            rows.append(best)
            output.write(json.dumps(best) + "\n")
            output.flush()
            if (index + 1) % 8 == 0:
                print(f"{index + 1} instances, gap={summarize(rows, args.test_in_tsplib)['avg_aug_gap']:.4f}%", flush=True)
    result = {"mode": args.mode, "initialization": args.initialization, "checkpoint": args.checkpoint,
              "budget_seconds": args.seconds, "augmentations": args.augmentations,
              "dataset": os.fspath(data), "timed_out": timed_out,
              "complete": not timed_out, "instances": 0, "model_warmup_seconds": warmup_seconds}
    if rows:
        result.update(summarize(rows, args.test_in_tsplib, baseline))
    result["wall_seconds_all_instances"] = (sum(row["seconds"] for row in rows) +
                                            sum(x["seconds"] for x in timed_out))
    if tracked_cuda:
        result["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    finished_routes = args.output / "routes.tmp.jsonl"
    finished_routes.write_text("".join(json.dumps(row) + "\n" for row in rows))
    finished_routes.replace(args.output / "routes.jsonl")
    finished_summary = args.output / "summary.tmp.json"
    finished_summary.write_text(json.dumps(result, indent=2) + "\n")
    finished_summary.replace(args.output / "summary.json")
    temporary.unlink()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
