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


def test_random_dags_differ_across_seeds_and_are_reproducible():
    bn = Graph(order=list(range(8)), parents=[[]] + [[j - 1] for j in range(1, 8)], name="bn")
    draws = [degree_matched_random_dag(bn, np.random.default_rng(s)).parents for s in range(5)]
    assert len({str(p) for p in draws}) > 1
    assert degree_matched_random_dag(bn, np.random.default_rng(3)).parents == draws[3]
