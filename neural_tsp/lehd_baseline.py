"""Capture final LEHD routes through its existing evaluation methods."""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import torch

from .data import instance_key, instances
from .graph import route_length, validate_tour


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("original", "ttpl"), required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tsplib", action="store_true")
    parser.add_argument("--projection", default="projection_1k")
    args = parser.parse_args()
    repo = args.repo.resolve()
    dataset = args.dataset.resolve()
    weights = args.weights.resolve()
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    records = list(instances(dataset, args.tsplib))[:args.limit]
    package = "LEHD.TSP.test" if args.variant == "original" else "lehd.TSP.test_tsp"
    sys.path.insert(0, str(repo))
    upstream = importlib.import_module(package)
    collected = []
    torch.manual_seed(0)

    if args.variant == "ttpl" and not hasattr(torch, "set_default_device"):
        # TTPL's tensors default to float32 on the selected CUDA device.
        def set_default_device(device):
            torch.set_default_tensor_type("torch.cuda.FloatTensor" if device.type == "cuda"
                                          else "torch.FloatTensor")
        torch.set_default_device = set_default_device

    class CaptureTester(upstream.Tester):
        def _test_one_batch(self, episode, batch_size, *positional, **keywords):
            captured = None
            original = self.env._get_travel_distance_2

            def capture(points, tour, *args, **kwargs):
                nonlocal captured
                if tour is not None and tour.shape[-1] == self.origin_problem.shape[1]:
                    captured = tour
                return original(points, tour, *args, **kwargs)

            started = time.monotonic()
            self.env._get_travel_distance_2 = capture
            try:
                result = super()._test_one_batch(episode, batch_size, *positional, **keywords)
                torch.cuda.synchronize()
            finally:
                self.env._get_travel_distance_2 = original
            elapsed = (time.monotonic() - started) / batch_size
            if captured is None:
                raise RuntimeError("The upstream tester did not expose its completed route")
            for (name, points, reference, _), tour in zip(records[episode:], captured.cpu().numpy()):
                validate_tour(tour, len(points))
                length = route_length(points, tour)
                collected.append({"name": name, "id": instance_key(points), "length": length,
                                  "reference": reference, "seconds": elapsed, "tour": tour.tolist()})
            return result

    env = dict(upstream.env_params, data_path=str(dataset), tsplib_path=str(dataset),
               RRC_budget=10 if args.variant == "original" else 0,
               test_in_tsplib=args.tsplib, random_insertion=False, RRC_range=1000)
    model = dict(upstream.model_params, knearest=True, k_nearest_nodes=100, coor_projection=True)
    config = dict(upstream.tester_params, use_cuda=True, cuda_device_num=0, test_episodes=len(records),
                  test_batch_size=1 if args.tsplib else args.batch,
                  model_load={"path": str(weights.parent), "epoch": int(weights.stem.split("-")[-1])})
    upstream.create_logger({"filepath": str(target.parent), "filename": "upstream.log"})
    tester = CaptureTester(env_params=env, model_params=model, tester_params=config)
    options = {} if args.variant == "original" else {
        "projection": getattr(upstream.projection, args.projection), "MVDF": True,
    }
    tester.run(**options)
    if len(collected) != len(records):
        raise RuntimeError("Expected one captured route per instance")
    temporary = target.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in collected))
    temporary.replace(target)


if __name__ == "__main__":
    main()
