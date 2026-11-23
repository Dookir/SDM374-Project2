import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .graph import route_length, validate_tour
from .search import read_lkh_tour, write_instance


class EAX:
    def __init__(self, executable, scratch):
        self.executable = Path(executable).resolve(strict=True)
        self.scratch = Path(scratch).resolve()

    def run(self, points, seconds, population=50, offspring=20):
        if len(points) < 51:
            raise ValueError("The upstream EAX implementation assumes at least 51 cities")
        with tempfile.TemporaryDirectory(dir=self.scratch, prefix="eax-") as tmp:
            folder = Path(tmp)
            write_instance(folder / "instance.tsp", points)
            process = subprocess.run(
                [str(self.executable), "1", "result", str(population), str(offspring), "instance.tsp"],
                cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=seconds, text=True, check=False)
            if process.returncode:
                raise RuntimeError(f"EAX failed: {process.stdout[-2000:]}")
            rows = (folder / "result_BestSol").read_text().splitlines()
            tours = [validate_tour(np.fromstring(line, sep=" ", dtype=np.int64) - 1, len(points))
                     for line in rows[1::2]]
            return min(tours, key=lambda tour: route_length(points, tour))


class Concorde:
    def __init__(self, executable, scratch):
        self.executable = Path(executable).resolve(strict=True)
        self.scratch = Path(scratch).resolve()

    def run(self, points, seconds):
        with tempfile.TemporaryDirectory(dir=self.scratch, prefix="concorde-") as tmp:
            folder = Path(tmp)
            write_instance(folder / "instance.tsp", points)
            result = subprocess.run([str(self.executable), "-o", "instance.sol", "instance.tsp"],
                                    cwd=folder, timeout=seconds, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, check=False)
            if result.returncode or not (folder / "instance.sol").exists():
                raise RuntimeError(f"Concorde failed: {result.stdout[-2000:]}")
            data = np.fromstring((folder / "instance.sol").read_text(), sep=" ", dtype=np.int64)
            if data[0] != len(points):
                raise ValueError("Concorde returned the wrong problem dimension")
            return validate_tour(data[1:], len(points))


class NeuroLKH:
    def __init__(self, directory, scratch, checkpoint=None):
        directory = Path(directory).resolve(strict=True)
        self.executable = directory / "LKH"
        self.scratch = Path(scratch).resolve()
        sys.path.insert(0, str(directory))
        from net.sgcn_model import SparseGCNModel

        self.model = SparseGCNModel().cuda().eval()
        saved = torch.load(checkpoint or directory / "pretrained/neurolkh.pt", map_location="cuda")
        self.model.load_state_dict(saved["model"])

    @torch.no_grad()
    def run(self, points, seconds, seed):
        n = len(points)
        if n <= 20:
            raise ValueError("NeuroLKH requires more than 20 cities")
        start = time.monotonic()
        with tempfile.TemporaryDirectory(dir=self.scratch, prefix="neurolkh-") as tmp:
            folder = Path(tmp)
            normalized = points - points.min(axis=0)
            normalized /= max(float(normalized.max()), 1e-12)
            write_instance(folder / "instance.tsp", points)
            feature_params = folder / "features.par"
            feature_params.write_text("PROBLEM_FILE = instance.tsp\nGerenatingFeature\nFeat_FILE = feature.txt\n")
            subprocess.run([str(self.executable), feature_params.name], cwd=folder, check=True,
                           timeout=seconds, stdout=subprocess.DEVNULL)
            with (folder / "feature.txt").open() as handle:
                values = np.asarray([np.fromstring(line, sep=" ") for line in list(handle)[:n]])
            values = values.reshape(n, 20, 3)
            edge_index = values[:, :, 0].astype(np.int64)
            edge_lengths = values[:, :, 1] / 1e6
            inverse = values[:, :, 2].astype(np.int64)

            def tensor(array, *shape):
                return torch.as_tensor(array, dtype=torch.float32, device="cuda").reshape(*shape)

            scores, _, penalties = self.model(tensor(normalized, 1, n, 2),
                                               tensor(edge_lengths, 1, -1, 1),
                                               tensor(edge_index, 1, -1),
                                               tensor(inverse, 1, -1), None, None, 20)
            order = np.argsort(-scores[..., 1].cpu().numpy().reshape(n, 20), axis=1)[:, :5]
            cand = np.take_along_axis(edge_index, order, axis=1)
            with (folder / "candidates.txt").open("w") as handle:
                handle.write(f"{n}\n")
                for node, row in enumerate(cand, 1):
                    edges = " ".join(f"{city + 1} {100 * rank}" for rank, city in enumerate(row))
                    handle.write(f"{node} 0 5 {edges}\n")
                handle.write("-1\nEOF\n")
            with (folder / "penalties.txt").open("w") as handle:
                handle.write(f"{n}\n")
                for node, penalty in enumerate(penalties.cpu().numpy().reshape(-1), 1):
                    handle.write(f"{node} {int(penalty * 1e6)}\n")
                handle.write("-1\nEOF\n")
            remaining = seconds - (time.monotonic() - start)
            if remaining <= 0:
                raise subprocess.TimeoutExpired("NeuroLKH feature/model inference", seconds)
            (folder / "solver.par").write_text(
                "PROBLEM_FILE = instance.tsp\nOUTPUT_TOUR_FILE = result.tour\n"
                "TOUR_FILE = result.tour\nCANDIDATE_FILE = candidates.txt\n"
                "PI_FILE = penalties.txt\nSUBGRADIENT = NO\nRUNS = 1\n"
                "MOVE_TYPE = 5\nPATCHING_C = 3\nPATCHING_A = 2\n"
                f"MAX_TRIALS = {n}\nTIME_LIMIT = {remaining:.6f}\nSEED = {seed + 1}\n")
            try:
                subprocess.run([str(self.executable), "solver.par"], cwd=folder, check=True,
                               timeout=remaining + .2, stdout=subprocess.DEVNULL)
            except subprocess.TimeoutExpired as deadline:
                try:
                    return read_lkh_tour(folder / "result.tour", n)
                except (FileNotFoundError, ValueError):
                    raise deadline
            return read_lkh_tour(folder / "result.tour", n)
