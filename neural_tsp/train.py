import argparse
import csv
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .competitors import Concorde, EAX
from .data import instance_key, instances, synthetic
from .graph import Graph, construct_tour, edge_keys, route_length, tour_edges
from .model import EdgeNetwork, candidate_groups, graph_inputs, ranked_candidates
from .search import LKH, solve


def save_arrays(path, **arrays):
    staging = path.with_suffix(".tmp.npz")
    np.savez_compressed(staging, **arrays)
    staging.replace(path)


def prepare(args, lkh):
    args.data.mkdir(parents=True, exist_ok=True)
    previous = {}
    for dataset in args.exclude:
        for _, coords, _, _ in instances(dataset):
            previous[instance_key(coords)] = None
    eax = EAX(args.eax, args.scratch) if args.eax else None
    concorde = Concorde(args.concorde, args.scratch) if args.concorde else None
    supplied = {}
    for dataset in args.labels:
        for _, coords, _, tour in instances(dataset):
            key = instance_key(coords)
            if key in previous:
                raise ValueError(f"Excluded evaluation instance in training labels: {key}")
            supplied[key] = (coords, tour)

    manifest = {
        "sizes": args.sizes, "examples": args.examples, "validation": args.validation,
        "label_seconds": args.label_seconds, "teacher_runs": args.teacher_runs,
        "label_instances": list(supplied), "eax": args.eax, "concorde": args.concorde,
        "extra_teacher_seconds": args.extra_teacher_seconds,
    }
    manifest_path = args.data / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Use a new --data directory when changing the label-generation configuration")
    staging_manifest = manifest_path.with_suffix(".tmp")
    staging_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    staging_manifest.replace(manifest_path)

    def generate(i, split, size, distribution, coords, known):
        path = args.data / f"{split}-{i:06d}.npz"
        if path.exists():
            return
        seed = i + (1_000_000_000 if split == "validation" else 0)
        points = synthetic(seed, size, distribution) if coords is None else coords
        if instance_key(points) in previous:
            raise ValueError("Training instance overlaps the exclusion list")
        init = lkh.run(points, args.label_seconds, seed, trials=0)
        teachers = [lkh.run(points, args.label_seconds, seed + 10000 + j)
                    for j in range(args.teacher_runs)]
        if eax is not None and 51 <= len(points) <= 10000:
            try:
                teachers.append(eax.run(points, args.extra_teacher_seconds))
            except subprocess.TimeoutExpired:
                pass
        if concorde is not None and len(points) <= 1000:
            try:
                teachers.append(concorde.run(points, args.extra_teacher_seconds))
            except subprocess.TimeoutExpired:
                pass
        if known is not None:
            teachers.append(known)
        lengths = np.asarray([route_length(points, tour) for tour in teachers])
        tours = np.stack(teachers)[lengths <= lengths.min() * 1.001 + 1e-12]
        save_arrays(path, points=points, teachers=tours, initial=init, distribution=distribution)

    sizes = args.sizes
    jobs = [(i, "train", sizes[(i // 4) % len(sizes)], i % 4, None, None)
            for i in range(args.examples)]
    jobs += [(i, "validation", sizes[(i // 4) % len(sizes)], i % 4, None, None)
             for i in range(args.validation)]
    jobs += [(args.examples + i, "train", len(coords), -1, coords, tour)
             for i, (coords, tour) in enumerate(supplied.values())]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(lambda row: generate(*row), jobs), 1):
            if i == len(jobs) or i % 32 == 0:
                print(f"Teacher instances {i}/{len(jobs)}", flush=True)


def balanced_loss(logits, target):
    positive = target.sum()
    negative = len(target) - positive
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if positive.item() == 0 or negative.item() == 0:
        return loss.mean()
    weight = target * (0.5 / positive) + (1 - target) * (0.5 / negative)
    return (loss * weight).sum()


def static_example(network, device, entry):
    graph = Graph.build(entry["points"])
    keys, _, (edges, nodes, features, _) = graph_inputs(graph, device)
    h = network.encode(edges, nodes, features)
    logits = network.edge_outputs(edges, h, features)
    n = len(graph.points)
    label = np.mean([np.isin(keys, edge_keys(tour_edges(tour), n))
                     for tour in entry["teachers"]], axis=0)
    target = torch.as_tensor(label, device=device, dtype=logits.dtype)
    return balanced_loss(logits, target)


@torch.no_grad()
def label_repairs(network, device, lkh, path, output, seconds, seed):
    if output.exists():
        return
    with np.load(path) as entry:
        points, initial = entry["points"], entry["initial"]
    graph = Graph.build(points)
    _, pairs, (edges, nodes, features, _) = graph_inputs(graph, device)
    h = network.encode(edges, nodes, features)
    neural = construct_tour(points, pairs, network.edge_outputs(edges, h, features).cpu().numpy())
    initial = min((initial, neural), key=lambda tour: route_length(points, tour))
    middle = lkh.run(points, seconds, seed, initial=initial, trials=5)
    near = lkh.run(points, seconds * 3, seed + 1, initial=middle)
    states = np.stack((initial, middle, near))
    gains = []
    for tour in states:
        _, pairs, (edges, _, features, state) = graph_inputs(graph, device, tour)
        logits = network.edge_outputs(edges, h, features).cpu().numpy()
        ranks = candidate_groups(pairs, features.cpu().numpy(), state.cpu().numpy(), logits)
        before = route_length(points, tour)
        row = []
        for rank in ranks:
            adjacency, _ = ranked_candidates(pairs, rank, graph.neighbors, tour)
            outcome = lkh.run(points, seconds, seed, initial=tour, candidates=adjacency, trials=20)
            improvement = before - route_length(points, outcome)
            row.append(max(0.0, improvement) / max(before, 1e-12))
        gains.append(row)
    save_arrays(output, points=points, states=states, gains=np.asarray(gains))


def repair_example(network, device, entry, state_index):
    points, tour = entry["points"], entry["states"][state_index]
    gains = entry["gains"][state_index]
    graph = Graph.build(points)
    with torch.no_grad():
        _, _, (edges, nodes, features, _) = graph_inputs(graph, device)
        h = network.encode(edges, nodes, features)
        _, pairs, (edges, _, features, state) = graph_inputs(graph, device, tour)
        logits = network.edge_outputs(edges, h, features).cpu().numpy()
    ranks = candidate_groups(pairs, features.cpu().numpy(), state.cpu().numpy(), logits)
    groups = [ranked_candidates(pairs, rank, graph.neighbors, tour)[1] for rank in ranks]
    representation = network.edge_outputs(edges, h, features, state)
    prediction = network.group_scores(representation, state, groups)
    target = torch.as_tensor(gains * 100, device=device, dtype=prediction.dtype)
    left, right = torch.triu_indices(len(groups), len(groups), offset=1, device=device)
    relative = target[left] - target[right]
    ordering = (F.softplus(-(prediction[left] - prediction[right]) * relative.sign()) *
                relative.abs().clamp(max=1)).mean()
    return F.smooth_l1_loss(prediction, target) + ordering


@torch.no_grad()
def validation_gap(network, device, lkh, paths, args, mode):
    lengths, references, elapsed = [], [], []
    for path in paths:
        with np.load(path) as entry:
            coords = entry["points"]
            reference = min(route_length(coords, tour) for tour in entry["teachers"])
        tour, info = solve(coords, network, device, lkh, args.validation_seconds, mode=mode,
                           rounds=2)
        lengths.append(route_length(coords, tour))
        references.append(reference)
        elapsed.append(info["seconds"])
    return 100 * (sum(lengths) / max(sum(references), 1e-12) - 1), sum(elapsed)


def fit(args, lkh):
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    network = EdgeNetwork().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    training = sorted(args.data.glob("train-*.npz"))
    validation = sorted(args.data.glob("validation-*.npz"))
    if len(validation) > args.validate_count:
        indices = np.linspace(0, len(validation) - 1, args.validate_count, dtype=int)
        validation = [validation[index] for index in indices]
    if not training or not validation:
        raise ValueError("Prepare both training and validation instances")
    rng = np.random.default_rng(args.seed)

    def stage(name, epochs, paths, loss_fn, mode):
        best = (float("inf"), float("inf"))
        checkpoint = args.output / f"{name}.pt"
        with (args.output / f"{name}.csv").open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "loss", "gap_percent", "validation_seconds"])
            for epoch in range(1, epochs + 1):
                network.train()
                total = torch.zeros((), device=device)
                sequence = list(rng.permutation(paths))
                if name == "static" and epoch <= epochs // 2:
                    small = []
                    for path in sequence:
                        with np.load(path) as entry:
                            if len(entry["points"]) <= 500:
                                small.append(path)
                    sequence = small or sequence
                for path in sequence:
                    optimizer.zero_grad(set_to_none=True)
                    with np.load(path) as entry:
                        loss = loss_fn(entry)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1)
                    optimizer.step()
                    total += loss.detach()
                network.eval()
                gap, elapsed = validation_gap(network, device, lkh, validation, args, mode)
                mean_loss = float(total) / len(sequence)
                print(f"{name} epoch={epoch} loss={mean_loss:.4f} validation_gap={gap:.4f}%", flush=True)
                writer.writerow([epoch, mean_loss, gap, elapsed])
                handle.flush()
                if (gap, mean_loss) < best:
                    best = (gap, mean_loss)
                    torch.save({"network": network.state_dict(), "epoch": epoch, "gap": gap,
                                "stage": name}, checkpoint.with_suffix(".tmp"))
                    checkpoint.with_suffix(".tmp").replace(checkpoint)
        network.load_state_dict(torch.load(checkpoint, map_location=device)["network"])

    if args.resume_static:
        network.load_state_dict(torch.load(args.output / "static.pt", map_location=device)["network"])
    else:
        stage("static", args.static_epochs, training, lambda entry: static_example(network, device, entry),
              "static")
    fingerprint = hashlib.sha256((args.output / "static.pt").read_bytes()).hexdigest()[:16]
    repair_dir = args.output / "repair_labels" / fingerprint
    repair_dir.mkdir(parents=True, exist_ok=True)
    repairs = [repair_dir / path.name for path in training]
    # Existing repair labels are valid only for the same frozen static checkpoint.
    def build_repair(row):
        i, (path, target) = row
        if not args.resume_static:
            target.unlink(missing_ok=True)
        label_repairs(network, device, lkh, path, target, args.repair_seconds, args.seed + i)

    jobs = enumerate(zip(training, repairs), 1)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(build_repair, jobs), 1):
            if i == len(training) or i % 32 == 0:
                print(f"Repair instances {i}/{len(training)}", flush=True)
    for layer in (network.embed, network.messages, network.updates, network.norms, network.static):
        layer.requires_grad_(False)
    stage("dynamic", args.repair_epochs, repairs,
          lambda entry: repair_example(network, device, entry, int(rng.integers(3))), "dynamic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "fit"))
    parser.add_argument("--lkh", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--validation", type=int, default=32)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 500, 1000, 5000])
    parser.add_argument("--labels", type=Path, nargs="*", default=[])
    parser.add_argument("--exclude", type=Path, nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eax")
    parser.add_argument("--concorde")
    parser.add_argument("--extra-teacher-seconds", type=float, default=20)
    parser.add_argument("--teacher-runs", type=int, default=4)
    parser.add_argument("--label-seconds", type=float, default=1.0)
    parser.add_argument("--repair-seconds", type=float, default=0.35)
    parser.add_argument("--resume-static", action="store_true")
    parser.add_argument("--static-epochs", type=int, default=10)
    parser.add_argument("--repair-epochs", type=int, default=5)
    parser.add_argument("--validate-count", type=int, default=12)
    parser.add_argument("--validation-seconds", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    lkh = LKH(args.lkh, args.scratch)
    if args.action == "prepare":
        prepare(args, lkh)
    else:
        if args.output is None or min(args.static_epochs, args.repair_epochs, args.validate_count) < 1:
            parser.error("fit requires an output directory and positive epoch/validation counts")
        fit(args, lkh)


if __name__ == "__main__":
    main()
