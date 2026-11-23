import numpy as np
import torch
from torch import nn


def mlp(in_features, width, out_features):
    return nn.Sequential(nn.Linear(in_features, width), nn.SiLU(), nn.Linear(width, out_features))


class EdgeNetwork(nn.Module):
    def __init__(self, width=128, layers=6, chunk=4096):
        super().__init__()
        self.chunk = chunk
        self.embed = nn.Linear(1, width)
        self.messages = nn.ModuleList(mlp(width + 5, width, width) for _ in range(layers))
        self.updates = nn.ModuleList(mlp(2 * width, width, width) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.static = mlp(width + 5, width, 1)
        self.state = mlp(width + 9, width, 32)
        self.repair = mlp(32 + 4, 64, 1)

    def encode(self, edges, nodes, features):
        h = self.embed(nodes)
        parts = tuple(zip(edges.T.split(self.chunk), features.split(self.chunk)))
        degree = torch.bincount(edges.flatten(), minlength=len(h)).clamp_min(1)[:, None]
        for message, update, norm in zip(self.messages, self.updates, self.norms):
            aggregate = torch.zeros_like(h)
            for pair, attrs in parts:
                a, b = pair.T
                aggregate.index_add_(0, a, message(torch.cat((h[b], attrs), dim=1)))
                aggregate.index_add_(0, b, message(torch.cat((h[a], attrs), dim=1)))
            h = norm(h + update(torch.cat((h, aggregate / degree), dim=1)))
        return h

    def edge_outputs(self, edges, h, features, state=None):
        output = []
        for start in range(0, features.shape[0], self.chunk):
            sl = slice(start, start + self.chunk)
            a, b = edges[:, sl]
            z = torch.cat((h[a] + h[b], features[sl]), dim=1)
            output.append(self.static(z).flatten() if state is None else
                          self.state(torch.cat((z, state[sl]), dim=1)))
        return torch.cat(output)

    def group_scores(self, representations, state, groups):
        pooled = []
        for group in groups:
            idx = torch.as_tensor(group, device=representations.device)
            pooled.append(torch.cat((representations[idx].mean(dim=0), state[idx].mean(dim=0))))
        return self.repair(torch.stack(pooled)).flatten()


def graph_inputs(graph, device, tour=None):
    keys, edges, nodes, features, state = graph.features(tour)
    return keys, edges, tuple(torch.as_tensor(x, device=device) for x in (edges, nodes, features, state))


def candidate_groups(edges, features, state, logits):
    """Three deterministic rankings offered the same LKH budget."""
    distance = features[:, 0]
    unseen = 1 - state[:, 0]
    logits = np.clip(logits, -8, 8)
    return (
        logits,
        logits - 1.5 * distance + 0.5 * unseen,
        logits + unseen * (1.5 * state[:, 1] + state[:, 3]),
    )


def ranked_candidates(edges, scores, neighbors, tour, count=8):
    n = len(neighbors)
    # Rank incident edges independently for each endpoint.
    city = edges.ravel()
    target = edges[::-1].ravel()
    index = np.tile(np.arange(edges.shape[1]), 2)
    order = np.lexsort((target, -scores[index], city))
    boundaries = np.searchsorted(city[order], np.arange(n + 1))
    adjacent = [[] for _ in range(n)]
    chosen = []
    prev, after = np.roll(tour, 1), np.roll(tour, -1)
    for node, before, nxt in zip(tour, prev, after):
        adjacent[node] = [int(before), int(nxt)]
    for node in range(n):
        rows = order[boundaries[node]:boundaries[node + 1]][:count]
        chosen.extend(index[rows])
        adjacent[node] = list(dict.fromkeys([*target[rows], *adjacent[node], *neighbors[node, :2]]))
    return adjacent, np.unique(chosen)
