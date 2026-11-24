import itertools
import unittest

import numpy as np
from scipy import optimize

from neural_tsp.exact import DistanceGrid, cycle_components, optimize_cycle, small_exact_bound
from neural_tsp.graph import edge_keys, route_length, tour_edges, validate_tour
from neural_tsp.partition import join, polish_tour
from neural_tsp.quality import Portfolio
from neural_tsp.regions import Region
from neural_tsp.train_regions import label


@unittest.skipUnless(hasattr(optimize, "milp"), "quality mode uses SciPy MILP")
class QualityTest(unittest.TestCase):
    def test_disconnected_degree_two_optimum_requires_subtour_cuts(self):
        pairs = np.asarray(list(itertools.combinations(range(6), 2)))
        incumbent = np.searchsorted(edge_keys(pairs, 6), edge_keys(tour_edges(np.arange(6)), 6))
        weights = np.where((pairs[:, 0] < 3) == (pairs[:, 1] < 3), 1, 20)
        result = optimize_cycle(6, pairs, weights, incumbent, 3)
        self.assertTrue(result.optimal)
        self.assertEqual(weights[result.selected].sum(), 44)
        self.assertEqual(len(cycle_components(6, pairs[result.selected])), 1)

    def test_region_matches_exhaustive_search_with_fixed_exterior_paths(self):
        points = np.random.default_rng(713).random((9, 2))
        initial = np.arange(9)
        portfolio = Portfolio(points)
        portfolio.add(initial, "initial")
        context = portfolio.context()
        region = Region(context, np.array([0, 1, 3, 4, 6, 7]))
        fixed = {1, 2, 4, 5, 7, 8}
        required = set(edge_keys(tour_edges(initial)[sorted(fixed)], 9))
        _, length = portfolio.best
        best = min(route_length(points, tour) for order in itertools.permutations(range(1, 9))
                   if required.issubset(edge_keys(tour_edges(tour := np.array((0, *order))), 9)))
        repair = region.repair(3, length)
        self.assertTrue(repair.proven)
        self.assertAlmostEqual(repair.length, best)
        self.assertTrue(required.issubset(edge_keys(tour_edges(repair.tour), 9)))
        paired = label(region, length, 3)
        self.assertAlmostEqual(paired["new_edge_gain"], length - best)

    def test_region_proposals_and_partition_polish_preserve_permutation(self):
        points = np.random.default_rng(42).random((80, 2))
        joined = join(points, np.arange(40), np.arange(40, 80))
        polished = validate_tour(polish_tour(points, joined), 80)
        self.assertLess(route_length(points, polished), route_length(points, joined))
        portfolio = Portfolio(points)
        portfolio.add(polished, "candidate")
        regions = list(portfolio.context().regions(32, 18))
        self.assertEqual(len(regions), 32)
        self.assertTrue(all(len(np.unique(region.cities)) == region.size for region in regions))

    def test_dyadic_distance_bounds_and_alternative_tours(self):
        square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        tour = np.arange(4)
        grid = DistanceGrid(square)
        self.assertEqual(grid.geometric_lower(), 4)
        self.assertEqual(grid.upper(tour), 4)
        self.assertGreater(small_exact_bound(grid, tour), 4)
        points = np.random.default_rng(5).random((4, 2)) * .6 + 1000.1234
        grid = DistanceGrid(points, bits=12)
        for order in itertools.permutations(range(4)):
            route = np.asarray(order)
            length = route_length(points, route)
            self.assertLessEqual(grid.geometric_lower(), length)
            self.assertLessEqual(length, grid.upper(route))
