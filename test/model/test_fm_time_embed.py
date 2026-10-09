import torch

from nucleus.layers.adaptive_layernorm import AdaptiveLayerNorm
from nucleus.layers.fm_time_embed import FMTimeEmbed

EMBED_DIM = 32


def build_time_conditioned_norm():
    torch.manual_seed(0)
    time_embed = FMTimeEmbed(EMBED_DIM)
    t_norm = AdaptiveLayerNorm(EMBED_DIM, EMBED_DIM, dtype=torch.float32)
    # AdaptiveLayerNorm starts as identity; perturb it so the modulation depends on its input
    torch.nn.init.normal_(t_norm.modulation[-1].weight, std=0.1)
    return time_embed, t_norm


def test_output_shape():
    flow_time = torch.rand(5, 1)
    assert FMTimeEmbed(EMBED_DIM)(flow_time).shape == (5, EMBED_DIM)


def test_distinct_times_give_distinct_embeddings():
    flow_time = torch.tensor([[0.0], [0.25], [0.5], [0.99]])
    embeddings = FMTimeEmbed(EMBED_DIM)(flow_time)
    pairwise_distance = torch.cdist(embeddings, embeddings)
    off_diagonal = pairwise_distance[~torch.eye(4, dtype=torch.bool)]
    assert off_diagonal.min() > 1e-3


def test_film_modulation_depends_on_flow_time():
    time_embed, t_norm = build_time_conditioned_norm()
    tokens = torch.randn(1, 2, 4, 4, EMBED_DIM).expand(2, -1, -1, -1, -1)
    flow_time = torch.tensor([[0.1], [0.9]])
    output = t_norm(tokens, time_embed(flow_time))
    assert not torch.allclose(output[0], output[1])
