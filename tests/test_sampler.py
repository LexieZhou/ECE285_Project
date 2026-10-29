import itertools

import torch

from tabstruct.models.masked_transformer import MaskedTabularTransformer
from tabstruct.models.masks import build_attention_mask
from tabstruct.models.sampler import ancestral_sample

CARDS = [2, 3, 2]
PARENTS = [[], [0], [0, 1]]


def peaked_model() -> MaskedTabularTransformer:
    """Random model with sharpened heads, so its joint is far from uniform."""
    torch.manual_seed(0)
    model = MaskedTabularTransformer(CARDS, dim=16, n_heads=2, n_layers=2, dropout=0.0)
    with torch.no_grad():
        for head in model.heads:
            head.weight.mul_(10)
    return model


def test_shape_range_and_reproducibility():
    model, mask = peaked_model(), build_attention_mask(PARENTS)
    a = ancestral_sample(model, mask, 100, torch.Generator().manual_seed(1), batch_size=32)
    b = ancestral_sample(model, mask, 100, torch.Generator().manual_seed(1), batch_size=32)
    assert a.shape == (100, len(CARDS)) and a.dtype == torch.long
    assert torch.equal(a, b)
    for j, card in enumerate(CARDS):
        assert a[:, j].min() >= 0 and a[:, j].max() < card


def test_samples_follow_model_joint():
    model, mask = peaked_model(), build_attention_mask(PARENTS)
    model.eval()
    cells = torch.tensor(list(itertools.product(*(range(c) for c in CARDS))))
    with torch.no_grad():
        exact = model.log_prob(cells, mask).sum(1).exp()
    assert torch.isclose(exact.sum(), torch.tensor(1.0), atol=1e-5)

    samples = ancestral_sample(model, mask, 20_000, torch.Generator().manual_seed(0))
    empirical = torch.stack([(samples == cell).all(1).float().mean() for cell in cells])
    tvd = 0.5 * (empirical - exact).abs().sum()
    assert tvd < 0.02, tvd  # sampling noise ~0.006; a reversed-order sampler gives ~0.08


def test_restores_training_mode():
    model = peaked_model().train()
    ancestral_sample(model, build_attention_mask(PARENTS), 10, torch.Generator().manual_seed(0))
    assert model.training
