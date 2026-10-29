import pytest

from tabstruct.structure.graphs import Graph


def test_full_prefix_and_marginal():
    order = [2, 0, 1]
    assert Graph.full_prefix(order).parents == [[], [0], [0, 1]]
    assert Graph.marginal(order).in_degrees() == [0, 0, 0]


def test_roundtrip():
    g = Graph(order=[1, 0, 2], parents=[[], [0], [1]], name="bn")
    assert Graph.from_dict(g.to_dict()) == g


@pytest.mark.parametrize(
    "order, parents",
    [
        ([0, 0, 1], [[], [], []]),  # order is not a permutation
        ([0, 1, 2], [[], [1], []]),  # parent not earlier in the order
        ([0, 1, 2], [[], [0], [0, 0]]),  # duplicate parent
        ([0, 1, 2], [[], [0]]),  # wrong number of parent lists
    ],
)
def test_validate_rejects_bad_graphs(order, parents):
    with pytest.raises(ValueError):
        Graph(order=order, parents=parents)
