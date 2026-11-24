"""Overlapping neighborhoods with exact preservation of exterior tour paths."""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from .exact import cycle_components, optimize_cycle
from .graph import edge_keys, route_length, tour_edges, validate_tour


class TourContext:
    def __init__(self, points, tour, parent_keys, parent_counts, parents, tree=None):
        self.points = points
        self.tour = tour
        self.tree = cKDTree(points) if tree is None else tree
        self.position = np.argsort(tour)
        self.parent_keys = parent_keys
        self.parent_counts = parent_counts
        self.parents = parents
        self.support = self.frequency(tour_edges(tour))
        step = np.linalg.norm(points[tour] - points[np.roll(tour, -1)], axis=1)
        self.step = step

    def frequency(self, pairs):
        keys = edge_keys(pairs, len(self.points))
        index = np.searchsorted(self.parent_keys, keys).clip(max=len(self.parent_keys) - 1)
        return np.where(self.parent_keys[index] == keys, self.parent_counts[index], 0) / self.parents

    def regions(self, count, seed):
        n = len(self.tour)
        rng = np.random.default_rng(seed)
        long = np.argsort(-self.step)
        disagreement = np.argsort(self.support, kind="stable")
        positions = rng.permutation(n)
        _, nearest = self.tree.query(self.points[self.tour], k=min(n, 6))
        starts = np.broadcast_to(np.arange(n)[:, None], nearest.shape)
        ends = self.position[nearest]
        near = np.linalg.norm(self.points[self.tour[starts]] - self.points[nearest], axis=2)
        left, right = self.tour[(starts + 1) % n], self.tour[(ends + 1) % n]
        two_opt = self.step[starts] + self.step[ends] - near - np.linalg.norm(
            self.points[left] - self.points[right], axis=2)
        offset = (ends - starts) % n
        two_opt[(offset <= 1) | (offset >= n - 1)] = -np.inf
        meaningful = np.argsort(-two_opt.ravel(), kind="stable")
        seen = set()
        sizes = [min(n, x) for x in (10, 18, 28, 44)]
        for i in range(max(count * 5, 20)):
            j, kind = divmod(i, 5)
            size = sizes[(j + kind) % len(sizes)]
            center = (long, disagreement, positions, long, positions)[kind][j % n]
            if kind == 3:
                selected = (center + np.arange(size) - size // 2) % n
            elif kind in (0, 4):
                # Cover both ends of long edges, including bridges between clusters.
                endpoints = self.tour[[center, (center + 1) % n]]
                if kind == 4:
                    index, slot = np.unravel_index(meaningful[j % len(meaningful)], nearest.shape)
                    endpoints = np.array([self.tour[index], nearest[index, slot]])
                _, neighbors = self.tree.query(self.points[endpoints], k=size)
                selected = self.position[np.unique(neighbors[:, :size // 2])]
            else:
                _, neighbors = self.tree.query(self.points[self.tour[center]], k=size)
                selected = self.position[neighbors]
            key = tuple(np.sort(selected))
            if len(key) < 3 or key in seen:
                continue
            seen.add(key)
            yield Region(self, np.asarray(key))
            if len(seen) == count:
                break


@dataclass
class Repair:
    tour: np.ndarray
    length: float
    proven: bool
    precision: float


class Region:
    def __init__(self, context, positions):
        self.context = context
        self.positions = positions
        self.cities = context.tour[positions]
        self.points = context.points[self.cities]
        self.size = len(positions)
        self.successor = np.roll(np.arange(self.size), -1)
        self.offsets = (positions[self.successor] - positions) % len(context.tour)
        self.fixed_start = np.flatnonzero(self.offsets > 1)
        fixed = np.column_stack((self.fixed_start, self.successor[self.fixed_start]))
        self.fixed = len(fixed)
        residual = 2 - np.bincount(fixed.ravel(), minlength=self.size)
        a, b = np.triu_indices(self.size, 1)
        free = (residual[a] > 0) & (residual[b] > 0)
        self.pairs = np.vstack((fixed, np.column_stack((a[free], b[free]))))
        self.support = context.frequency(self.cities[self.pairs])
        self.distance = np.linalg.norm(self.points[self.pairs[:, 0]] - self.points[self.pairs[:, 1]], axis=1)
        self.distance[:self.fixed] = 0
        self.scale = max(np.ptp(self.points, axis=0).max(), 1e-12)
        consecutive = np.flatnonzero(self.offsets == 1)
        keys = edge_keys(self.pairs[self.fixed:], self.size)
        current = edge_keys(np.column_stack((consecutive, self.successor[consecutive])), self.size)
        self.incumbent = np.concatenate((np.arange(self.fixed),
                                         self.fixed + np.searchsorted(keys, current)))
        self.base = self.support > 0
        self.base[self.incumbent] = True
        self.base[:self.fixed] = True
        self.boundary = (2 - residual) / 2

    def features(self):
        size = self.size
        xy = (self.points - self.points.mean(axis=0)) / self.scale
        nodes = np.column_stack((xy, self.boundary)).astype(np.float32)
        features = np.zeros((size, size, 6), dtype=np.float32)
        a, b = self.pairs.T
        features[a, b, 0] = self.distance / self.scale
        features[a, b, 1] = self.support
        features[a, b, 2] = 1
        features[self.pairs[self.incumbent, 0], self.pairs[self.incumbent, 1], 3] = 1
        features[a[:self.fixed], b[:self.fixed], 4] = 1
        features = np.maximum(features, features.transpose(1, 0, 2))
        span = np.abs(self.positions[:, None] - self.positions)
        n = len(self.context.tour)
        features[:, :, 5] = 2 * np.minimum(span, n - span) / n
        return nodes, features

    def repair(self, seconds, length, extra=None):
        allowed = np.ones(len(self.pairs), bool) if extra is None else self.base.copy()
        if extra is not None:
            allowed[extra] = True
        candidates = np.flatnonzero(allowed)
        quantum = self.scale / (1 << 25)
        weights = np.rint(self.distance[candidates] / quantum).astype(np.int64)
        choice = optimize_cycle(self.size, self.pairs[candidates], weights,
                                np.searchsorted(candidates, self.incumbent), seconds, self.fixed)
        selected = candidates[choice.selected]
        precision = (self.size - self.fixed) * quantum
        if self.distance[selected].sum() >= self.distance[self.incumbent].sum():
            return Repair(self.context.tour, length, choice.optimal, precision)
        tour = self.expand(selected)
        improved = route_length(self.context.points, tour)
        if improved < length:
            return Repair(tour, improved, choice.optimal, precision)
        return Repair(self.context.tour, length, choice.optimal, precision)

    def expand(self, selected):
        ((cities, edges),) = cycle_components(self.size, self.pairs[selected])
        parts = []
        for city, edge in zip(cities, selected[edges]):
            parts.append(self.cities[city:city + 1])
            if edge < self.fixed:
                start = self.fixed_start[edge]
                positions = self.positions[start] + np.arange(1, self.offsets[start])
                exterior = self.context.tour[positions % len(self.context.tour)]
                parts.append(exterior if city == start else exterior[::-1])
        return validate_tour(np.concatenate(parts), len(self.context.tour))

    def extras(self, scores, limit):
        eligible = np.flatnonzero(~self.base)
        a, b = self.pairs[eligible].T
        return eligible[np.argsort(-scores[a, b], kind="stable")[:limit]]


class RegionNetwork(nn.Module):
    """Joint edge and region scores conditioned on the exterior paths."""

    def __init__(self, width=48, depth=3):
        super().__init__()
        self.node = nn.Linear(3, width)
        self.edge = nn.Linear(6, width)
        self.updates = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(depth))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(depth))
        self.links = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 1))
        self.gain = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, nodes, features, mask):
        h = self.node(nodes)
        e = self.edge(features)
        size = mask.sum(dim=1, keepdim=True)
        for update, norm in zip(self.updates, self.norms):
            message = torch.nn.functional.silu(h[:, None, :, :] + e)
            pooled = (message * mask[:, None, :, None]).sum(dim=2) / size[:, :, None]
            h = norm(h + torch.nn.functional.silu(update(torch.cat((h, pooled), dim=-1))))
        combined = torch.cat((h[:, :, None, :] + h[:, None, :, :], e), dim=-1)
        links = self.links(combined).squeeze(-1)
        summary = (h * mask[..., None]).sum(dim=1) / size
        pair_mask = mask[:, :, None] & mask[:, None, :]
        edge_mean = (e * pair_mask[..., None]).sum(dim=(1, 2)) / (size * size)
        maximum = h.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)
        return self.gain(torch.cat((summary, maximum, edge_mean), dim=-1)).flatten(), links


def feature_batch(features, device):
    size = max(len(nodes) for nodes, _ in features)
    nodes = np.zeros((len(features), size, 3), dtype=np.float32)
    edges = np.zeros((len(features), size, size, 6), dtype=np.float32)
    mask = np.zeros((len(features), size), dtype=bool)
    for i, (node, edge) in enumerate(features):
        n = len(node)
        nodes[i, :n] = node
        edges[i, :n, :n] = edge
        mask[i, :n] = True
    return tuple(torch.as_tensor(x, device=device) for x in (nodes, edges, mask))


@torch.no_grad()
def predict(model, regions, device, batch_size=16):
    gains, links = [], []
    for start in range(0, len(regions), batch_size):
        batch = regions[start:start + batch_size]
        score, edges = model(*feature_batch([r.features() for r in batch], device))
        gains.extend(score.cpu().numpy())
        links.extend(value[:r.size, :r.size] for value, r in zip(edges.cpu().numpy(), batch))
    return np.asarray(gains), links
