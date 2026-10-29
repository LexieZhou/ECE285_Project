import numpy as np

from tabstruct.structure.graphs import Graph
from tabstruct.structure.random_dag import degree_matched_random_dag


def test_degree_matched():
    bn = Graph(order=[3, 1, 0, 2, 4], parents=[[], [0], [0, 1], [2], [1, 2, 3]], name="bn")
    for seed in range(10):
        g = degree_matched_random_dag(bn, np.random.default_rng(seed))
        assert g.order == bn.order
        assert g.in_degrees() == bn.in_degrees()
        g.validate()
