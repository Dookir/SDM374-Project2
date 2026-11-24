"""Generate paired repair labels, train both heads, and run frozen comparisons."""
import argparse
import inspect
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .competitors import EAX, NeuroLKH
from .data import instance_key, synthetic
from .exact import DistanceGrid
from .graph import edge_keys, route_length, tour_edges
from .model import EdgeNetwork
from .quality import Portfolio, collect_parents, merge, region_order
from .regions import RegionNetwork, feature_batch, predict
from .search import LKH


def label(region, length, seconds):
    restricted = region.repair(seconds, length, extra=[])
    opened = region.repair(seconds, length)
    # The restricted solution is feasible in the open graph.
    if restricted.length < opened.length:
        opened.tour, opened.length = restricted.tour, restricted.length
    pairs = region.cities[region.pairs]
    keys = edge_keys(pairs, len(region.context.tour))
    adopted = np.isin(keys, edge_keys(tour_edges(opened.tour), len(region.context.tour))) & ~region.base
    links = np.zeros((region.size, region.size), bool)
    eligible = np.zeros_like(links)
    a, b = region.pairs.T
    links[a, b] = adopted
    eligible[a, b] = ~region.base
    support = region.support[region.incumbent[region.fixed:]]
    return {
        "features": region.features(), "links": links, "eligible": eligible,
        "gain": length - opened.length, "new_edge_gain": restricted.length - opened.length,
        "restricted": restricted.length, "opened": opened.length,
        "proven": opened.proven and restricted.proven,
        "precision": opened.precision + restricted.precision,
        "consensus": bool(np.all(support == 1)),
    }


def load_model(path, device):
    model = RegionNetwork().to(device)
    model.load_state_dict(torch.load(path, map_location=device)["network"])
    return model.eval()


def additional_states(points, portfolio, parents, length, seed):
    if len(parents) > 1:
        donor_pool = Portfolio(points)
        for tour in parents[1:]:
            donor_pool.add(tour, "teacher")
        tour, score = donor_pool.best
        yield "best_parent_withheld", donor_pool, tour, score
    # Add small legal perturbations of near-optimal states so that the model
    # also gets supervised examples when all parent solvers converge together.
    rng = np.random.default_rng(seed)
    tour = portfolio.best[0].copy()
    if len(tour) < 8:
        return
    for index in rng.choice(len(tour) - 1, 2, replace=False):
        tour[index:index + 2] = tour[index:index + 2][::-1]
    score = route_length(points, tour)
    if not length < score < length * 1.08:
        return
    donor_pool = Portfolio(points)
    for alternative in parents[1:]:
        donor_pool.add(alternative, "teacher")
    donor_pool.add(tour, "perturbed")
    yield "perturbed", donor_pool, tour, score


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    lkh = LKH(args.lkh, args.scratch)
    device = torch.device(args.device)
    eax = EAX(args.eax, args.scratch) if args.eax else None
    neurolkh = NeuroLKH(args.neurolkh, args.scratch) if args.neurolkh else None
    search_model = None
    if args.static:
        search_model = EdgeNetwork().to(device)
        search_model.load_state_dict(torch.load(args.static, map_location=device)["network"])
        search_model.eval()
    trained = load_model(args.mine, device) if args.mine else None
    for i in range(args.count):
        size = args.sizes[(i // 4) % len(args.sizes)]
        distribution = i % 4
        seed = args.seed + i
        path = args.output / f"{seed}.pt"
        if path.exists():
            continue
        points = synthetic(seed, size, distribution)
        # On exact/paired benchmark comparisons, the baseline and all policies
        # must see exactly the same incumbent and candidate regions.
        portfolio = Portfolio(points)
        errors = collect_parents(portfolio, lkh, search_model, device, eax, neurolkh,
                                 args.parent_seconds, args.seeds, seed)
        parent_tour, parent_length = portfolio.best
        solver_tours = [row[1] for row in portfolio.elites
                        if row[0] <= parent_length * 1.03]
        portfolio.elites = [row for row in portfolio.elites if row[0] <= parent_length * 1.03]
        merged_optimal = merge(portfolio, DistanceGrid(points), args.merge_seconds)
        base_tour, base_length = portfolio.best
        states = [("merged", portfolio, base_tour, base_length)]
        if args.training and len(solver_tours) > 1:
            intermediate = solver_tours[min(len(solver_tours) - 1, 2)]
            states.append(("intermediate", portfolio, intermediate, route_length(points, intermediate)))
        if args.training and not np.array_equal(parent_tour, base_tour):
            states.append(("parent", portfolio, parent_tour, parent_length))
        if args.training:
            states.extend(additional_states(points, portfolio, solver_tours, base_length, seed))
        for state, (name, pool, tour, length) in enumerate(states):
            regions = list(pool.context(tour=tour).regions(args.regions, seed + state))
            examples = [label(region, length, args.local_seconds) for region in regions]
            if trained is not None:
                predicted, _ = predict(trained, regions, device)
                missed = set(np.argsort(predicted)[:max(1, len(predicted) // 2)])
                for j, example in enumerate(examples):
                    example["hard"] = (example["new_edge_gain"] > example["precision"] and
                                       (j in missed or example["consensus"]))
            else:
                for example in examples:
                    example["hard"] = False
            states[state] = {"name": name, "length": length, "examples": examples,
                             "order": {mode: region_order(regions, mode, seed + state)
                                       for mode in ("random", "disagreement")}}
        temporary = path.with_suffix(".tmp")
        torch.save({"id": instance_key(points), "size": size, "distribution": distribution,
                    "parent": parent_length, "merged": base_length, "merge_proven": merged_optimal,
                    "states": states, "solver_errors": errors}, temporary)
        temporary.replace(path)
        print(f"{seed} n={size} parents={len(solver_tours)} merged={base_length:.8f} "
              f"states={len(states)} errors={len(errors)}", flush=True)


def read_cases(directories):
    options = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
    return [torch.load(path, map_location="cpu", **options)
            for directory in directories for path in sorted(directory.glob("*.pt"))]


def evaluate(model, cases, device, attempts, all_states=False):
    rows = []
    for case in cases:
        if all_states:
            copies = [dict(case, states=[state]) for state in case["states"]]
            rows.extend(evaluate(model, copies, device, attempts))
            continue
        state = case["states"][0]
        examples = state["examples"]
        opened = np.asarray([row["opened"] for row in examples])
        restricted = np.asarray([row["restricted"] for row in examples])
        row = {"id": case["id"], "size": case["size"], "distribution": case["distribution"],
               "parent": case["parent"], "merged": case["merged"],
               "merge_proven": case["merge_proven"], "restricted_oracle": min(restricted),
               "oracle": min(opened), "proven_pairs": sum(ex["proven"] for ex in examples),
               "regions": len(examples), "solver_errors": case["solver_errors"],
               "novel_regions": sum(bool(ex["new_edge_gain"] > ex["precision"]) for ex in examples),
               "consensus_novel": sum(bool(ex["consensus"] and ex["new_edge_gain"] > ex["precision"])
                                      for ex in examples)}
        for policy, ordering in state["order"].items():
            row[policy] = float(min(opened[ordering[:attempts]]))
        if model is not None:
            with torch.no_grad():
                inputs = feature_batch([ex["features"] for ex in examples], device)
                scores, links = model(*inputs)
                order = torch.argsort(scores, descending=True).cpu().numpy()[:attempts]
                row["model"] = float(min(opened[order]))
                row["hybrid"] = min(row["model"], row["disagreement"])
                hits, total = 0, 0
                for i in order:
                    eligible = np.argwhere(examples[i]["eligible"])
                    wanted = examples[i]["links"]
                    if not len(eligible) or not wanted.any():
                        continue
                    a, b = eligible.T
                    rank = np.argsort(-links[i].cpu().numpy()[a, b], kind="stable")
                    rank = rank[:len(wanted)]
                    hits += int(wanted[a[rank], b[rank]].sum())
                    total += int(wanted.sum())
                row.update(link_hits=hits, link_targets=total)
        rows.append(row)
    return rows


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random = np.random.default_rng(args.seed)
    train_cases, validation = read_cases(args.training), read_cases([args.validation])
    ids = [case["id"] for case in train_cases]
    validation_ids = {case["id"] for case in validation}
    if not train_cases or not validation or len(ids) != len(set(ids)) or validation_ids.intersection(ids):
        raise ValueError("Training and validation must contain unique, disjoint instances")
    groups = [state["examples"] for case in train_cases for state in case["states"]]
    model = load_model(args.start, device) if args.start else RegionNetwork().to(device)
    if args.start:
        previous = torch.load(args.start, map_location="cpu")
        previous_train = set(previous["training_ids"])
        previous_validation = set(previous["validation_ids"])
        if previous_train.intersection(validation_ids) or previous_validation.intersection(ids):
            raise ValueError("Instances cannot change role when continuing from a checkpoint")
        all_training = sorted(previous_train.union(ids))
        all_validation = sorted(previous_validation.union(validation_ids))
    else:
        all_training, all_validation = sorted(ids), sorted(validation_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_quality = (math.inf, math.inf)
    for epoch in range(args.epochs):
        model.train()
        running = 0.
        for index in random.permutation(len(groups)):
            group = groups[index]
            inputs = feature_batch([item["features"] for item in group], device)
            predicted, edges = model(*inputs)
            gains = torch.as_tensor([np.log1p(1000 * row["gain"]) for row in group],
                                    device=device, dtype=predicted.dtype)
            novel = torch.as_tensor([np.log1p(1000 * row["new_edge_gain"]) for row in group],
                                    device=device, dtype=predicted.dtype)
            confident = torch.as_tensor([1. if row["proven"] else .2 for row in group], device=device)
            hard = torch.as_tensor([2. if row["hard"] else 1. for row in group], device=device)
            weights = confident * hard
            gain_loss = (F.smooth_l1_loss(predicted, gains + .25 * novel, reduction="none") * weights).mean()
            i, j = torch.triu_indices(len(group), len(group), 1, device=device)
            difference = gains[i] - gains[j]
            rank_loss = (F.softplus(-(predicted[i] - predicted[j]) * difference.sign()) *
                         difference.abs() * torch.minimum(weights[i], weights[j])).sum() / max(1, len(i))
            edge_loss = edges.new_zeros(())
            for k, example in enumerate(group):
                eligible = torch.as_tensor(example["eligible"], device=device)
                n = len(eligible)
                truth = torch.as_tensor(example["links"], device=device)
                logits, target = edges[k, :n, :n][eligible], truth[eligible].float()
                if target.numel() and target.any():
                    positive = (1 - target).sum() / target.sum()
                    edge_loss += weights[k] * F.binary_cross_entropy_with_logits(
                        logits, target, pos_weight=positive.clamp(1, 100))
            loss = gain_loss + rank_loss + edge_loss / len(group)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        model.eval()
        rows = evaluate(model, validation, device, args.attempts, all_states=True)
        regret = sum((row["model"] - row["oracle"]) / row["merged"] for row in rows) / len(rows)
        link_recall = sum(row["link_hits"] for row in rows) / max(1, sum(row["link_targets"] for row in rows))
        quality = (regret, -link_recall)
        print(f"epoch={epoch + 1} train={running / len(groups):.6f} "
              f"validation_relative_regret={regret:.8f} validation_link_recall={link_recall:.4f}", flush=True)
        if quality < best_quality:
            best_quality = quality
            temporary = args.output.with_suffix(".tmp")
            torch.save({"network": model.state_dict(), "epoch": epoch + 1, "kind": "region",
                        "training_ids": all_training, "validation_ids": all_validation},
                       temporary)
            temporary.replace(args.output)


def benchmark(args):
    device = torch.device(args.device)
    model = load_model(args.model, device) if args.model else None
    cases = read_cases([args.data])
    if not cases:
        raise ValueError("No paired regional cases")
    if args.model:
        stored = torch.load(args.model, map_location="cpu")
        used = set(stored["training_ids"]) | set(stored["validation_ids"])
        if used.intersection(case["id"] for case in cases):
            raise ValueError("Benchmarks must be disjoint from model training and validation")
    rows = evaluate(model, cases, device, args.attempts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    groups = {"all": rows}
    for row in rows:
        groups.setdefault(str(row["size"]), []).append(row)
        groups.setdefault(("uniform", "clustered", "mixed-density", "elongated")[row["distribution"]], []).append(row)
    for name, subset in groups.items():
        parent = sum(row["parent"] for row in subset)
        metrics = {key: 100 * (sum(row[key] for row in subset) / parent - 1)
                   for key in ("merged", "restricted_oracle", "random", "disagreement", "model", "hybrid", "oracle")
                   if key in subset[0]}
        print(name, len(subset), metrics, "novel regions", sum(row["novel_regions"] for row in subset),
              "proven pairs", sum(row["proven_pairs"] for row in subset))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for option in ("lkh", "scratch"):
        prepare_parser.add_argument("--" + option, required=True)
    for option in ("eax", "static", "neurolkh", "mine"):
        prepare_parser.add_argument("--" + option)
    prepare_parser.add_argument("--training", action="store_true")
    prepare_parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000])
    prepare_parser.add_argument("--count", type=int, default=32)
    prepare_parser.add_argument("--seed", type=int, required=True)
    prepare_parser.add_argument("--seeds", type=int, default=2)
    prepare_parser.add_argument("--regions", type=int, default=16)
    prepare_parser.add_argument("--parent-seconds", type=float, default=15)
    prepare_parser.add_argument("--local-seconds", type=float, default=2)
    prepare_parser.add_argument("--merge-seconds", type=float, default=15)
    train_parser = commands.add_parser("fit")
    train_parser.add_argument("--training", type=Path, nargs="+", required=True)
    train_parser.add_argument("--validation", type=Path, required=True)
    train_parser.add_argument("--start", type=Path)
    train_parser.add_argument("--epochs", type=int, default=16)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--attempts", type=int, default=3)
    benchmark_parser = commands.add_parser("benchmark")
    benchmark_parser.add_argument("--data", type=Path, required=True)
    benchmark_parser.add_argument("--model", type=Path)
    benchmark_parser.add_argument("--attempts", type=int, default=3)
    for command in (prepare_parser, train_parser, benchmark_parser):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "prepare":
        if min(args.sizes) < 3 or min(args.count, args.seeds, args.regions, args.parent_seconds,
                                     args.local_seconds, args.merge_seconds) <= 0:
            parser.error("Expected positive search parameters and at least three cities")
    elif args.attempts < 1:
        parser.error("Expected at least one repair attempt")
    torch.set_num_threads(2)
    {"prepare": prepare, "fit": train, "benchmark": benchmark}[args.command](args)


if __name__ == "__main__":
    main()
