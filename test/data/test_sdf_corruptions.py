import torch

from nucleus.noise import SDFBubbleResize, SDFSpuriousSpecks, SmoothFieldNoise, SmoothFieldWarp

SDF_MEAN, SDF_STD, DOMAIN_SIZE = -2.5, 2.13, 16.0
SIZE = 32
CELL = DOMAIN_SIZE / SIZE
CONSTANTS = dict(sdf_mean=SDF_MEAN, sdf_std=SDF_STD, domain_size=DOMAIN_SIZE)


def context_with_raw_sdf(raw_sdf: torch.Tensor, batch: int = 4) -> torch.Tensor:
    normalized_sdf = (raw_sdf - SDF_MEAN) / SDF_STD
    other_fields = torch.randn(batch, 4, SIZE, SIZE, 3)
    return torch.cat([normalized_sdf.expand(batch, 4, SIZE, SIZE)[..., None], other_fields], dim=-1)


def raw_sdf(context: torch.Tensor) -> torch.Tensor:
    return context[..., 0] * SDF_STD + SDF_MEAN


def planar_interface_sdf() -> torch.Tensor:
    # vapor for columns right of the middle, |grad| = 1 like a real distance field
    columns = (torch.arange(SIZE) + 0.5) * CELL
    return (columns - DOMAIN_SIZE / 2).expand(SIZE, SIZE)


def test_specks_add_vapor_only_in_bulk_liquid():
    torch.manual_seed(0)
    context = context_with_raw_sdf(torch.full((SIZE, SIZE), -5.0))
    corrupted = SDFSpuriousSpecks(**CONSTANTS, active_prob=1.0, radius_cells=(1.0, 1.5))(context)
    assert (raw_sdf(corrupted) > 0).any()
    assert (corrupted[..., 0] >= context[..., 0] - 1e-6).all()
    torch.testing.assert_close(corrupted[..., 1:], context[..., 1:])


def test_specks_skip_liquid_next_to_bubbles():
    context = context_with_raw_sdf(torch.full((SIZE, SIZE), -0.1))
    corrupted = SDFSpuriousSpecks(**CONSTANTS, active_prob=1.0)(context)
    torch.testing.assert_close(corrupted, context)


def test_bubble_resize_moves_interfaces_by_at_most_the_shift():
    torch.manual_seed(0)
    context = context_with_raw_sdf(planar_interface_sdf())
    corrupted = SDFBubbleResize(**CONSTANTS, active_prob=1.0, max_shift_cells=1.0)(context)
    shift = raw_sdf(corrupted) - raw_sdf(context)
    assert shift.abs().max() <= CELL + 1e-4
    assert shift.abs().max() > 0.1 * CELL


def test_zero_warp_is_identity():
    context = context_with_raw_sdf(planar_interface_sdf())
    warped = SmoothFieldWarp(**CONSTANTS, active_prob=1.0, max_shift_cells=0.0)(context)
    torch.testing.assert_close(warped, context, atol=1e-5, rtol=0)


def test_smooth_noise_is_spatially_correlated():
    torch.manual_seed(0)
    context = torch.zeros(2, 4, SIZE, SIZE, 4)
    noise = SmoothFieldNoise(**CONSTANTS, active_prob=1.0, min_std=0.05, max_std=0.05)(context)
    neighbour_difference = (noise[:, :, 1:] - noise[:, :, :-1]).pow(2).mean()
    # white noise would give a ratio of 2; correlated noise changes slowly between neighbours
    assert neighbour_difference / noise.pow(2).mean() < 0.5


def test_inactive_corruption_leaves_context_unchanged():
    context = context_with_raw_sdf(torch.full((SIZE, SIZE), -5.0))
    for corruption in (SDFSpuriousSpecks, SDFBubbleResize, SmoothFieldWarp, SmoothFieldNoise):
        torch.testing.assert_close(corruption(**CONSTANTS, active_prob=0.0)(context), context)
