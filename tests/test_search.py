import tempfile
import unittest
from pathlib import Path

import numpy as np

from neural_tsp.graph import Graph, construct_tour, validate_tour
from neural_tsp.search import LKH


class SearchTests(unittest.TestCase):
    def test_tour_completion_with_duplicate_and_collinear_points(self):
        for points in (np.zeros((40, 2)), np.column_stack((np.arange(40), np.zeros(40)))):
            graph = Graph.build(points)
            _, edges, *_ = graph.features()
            # An empty proposal list forces geometric completion.
            tour = construct_tour(points, edges[:, :0], np.empty(0))
            validate_tour(tour, len(points))

    def test_deadline_with_incomplete_output_preserves_incumbent(self):
        points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        initial = np.arange(len(points))
        with tempfile.TemporaryDirectory() as temp:
            fake = Path(temp) / "test-lkh"
            fake.write_text("#!/bin/sh\nprintf 'NAME : incomplete\\n' > result.tour\nexec sleep 5\n")
            fake.chmod(0o700)
            solver = LKH(fake, temp)
            tour = solver.run(points, .05, seed=0, initial=initial)
        np.testing.assert_array_equal(tour, initial)


if __name__ == "__main__":
    unittest.main()
