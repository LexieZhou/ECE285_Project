"""Invariant 1 (CLAUDE.md): column j's prediction depends only on x_{C(j)}, at every layer."""

import torch

from tabstruct.models.masked_transformer import MaskedTabularTransformer
from tabstruct.models.masks import build_attention_mask

CARDS = [3, 4, 2, 5, 3]
PARENTS = [[], [0], [0, 1], [1], [2, 3]]


def test_mask_structure():
    d = len(PARENTS)
    expected = torch.zeros(2 * d, 2 * d, dtype=torch.bool)
    for k in range(d):
        expected[k, k] = True
    for j, pa in enumerate(PARENTS):
        expected[d + j, d + j] = True
        expected[d + j, pa] = True
    assert torch.equal(build_attention_mask(PARENTS), expected)


def test_prediction_depends_only_on_parents():
    torch.manual_seed(0)
    model = MaskedTabularTransformer(CARDS, dim=16, n_heads=2, n_layers=3, dropout=0.0).eval()
    mask = build_attention_mask(PARENTS)
    x = torch.stack([torch.randint(0, c, (64,)) for c in CARDS], dim=1)
    base = model(x, mask)

    for m, card in enumerate(CARDS):
        x_perturbed = x.clone()
        x_perturbed[:, m] = (x_perturbed[:, m] + 1) % card
        out = model(x_perturbed, mask)
        for j, pa in enumerate(PARENTS):
            if m in pa:
                assert not torch.allclose(out[j], base[j], atol=1e-6), (j, m)
            else:  # includes m == j: a column never sees its own value
                assert torch.allclose(out[j], base[j], atol=1e-6), (j, m)


def test_log_prob_shape():
    model = MaskedTabularTransformer(CARDS, dim=16, n_heads=2, n_layers=1)
    x = torch.stack([torch.randint(0, c, (8,)) for c in CARDS], dim=1)
    log_p = model.log_prob(x, build_attention_mask(PARENTS))
    assert log_p.shape == (8, len(CARDS))
    assert (log_p <= 0).all()
