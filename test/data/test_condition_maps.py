import pathlib

import pytest
import torch
import yaml

from nucleus.data.conditioning import NUM_CONDITION_CHANNELS, condition_maps, initial_thermal_layer
from nucleus.data.normalize import get_normalizer

NORMALIZER_CONFIG = pathlib.Path(__file__).parents[2] / "config" / "normalizer_cfg" / "op250_64_nondim.yaml"
SITES_X = [-3.5, 1.5, 4.0]
SIM_PARAMS = {
    "x_min": -8.0, "x_max": 8.0, "y_min": 0.0, "y_max": 16.0,
    "heater": {"xMin": -5.25, "xMax": 5.25, "yMin": 0.0, "yMax": 1e-13, "rcdAngle": 45.0},
    "nuc_sites_x": SITES_X, "nuc_sites_y": [5e-14] * 3, "nuc_seed_radii": [0.1] * 3,
}


def raw_maps(size: int):
    normalizer = get_normalizer(yaml.safe_load(open(NORMALIZER_CONFIG)))
    maps = condition_maps(SIM_PARAMS, normalizer, size, size)
    seeds = normalizer.unnormalize_sdf(maps[..., 2])
    thermal = normalizer.unnormalize_temp(maps[..., 3], 0.0)
    return maps, seeds, thermal


def column_of(x: float, size: int) -> int:
    return int((x + 8.0) / (16.0 / size))


def test_shape_and_coordinate_channels():
    maps, _, _ = raw_maps(64)
    assert maps.shape == (64, 64, NUM_CONDITION_CHANNELS)
    assert maps[..., 0].min() > -1 and maps[..., 0].max() < 1
    assert (maps[0, 1:, 0] > maps[0, :-1, 0]).all()  # x grows along the width
    assert (maps[1:, 0, 1] > maps[:-1, 0, 1]).all()  # y grows up from the heater row


def test_seed_field_matches_simulator_seeds_at_full_resolution():
    _, seeds, _ = raw_maps(512)
    for site_x in SITES_X:
        assert seeds[0, column_of(site_x, 512)] > 0  # inside the planted semicircular seed
    assert seeds[0, column_of(-7.5, 512)] < 0


def test_seed_field_stays_informative_at_64():
    _, seeds, _ = raw_maps(64)
    bottom_row = seeds[0]
    for site_x in SITES_X:
        column = column_of(site_x, 64)
        assert bottom_row[column] >= bottom_row[max(column - 2, 0)]
        assert bottom_row[column] > -0.2  # within about a cell of the seed surface
    assert seeds[20].max() < -4.0  # far from the wall the field is a plain distance


def test_thermal_layer_matches_simulation_profile_at_full_resolution():
    # point values at cell centers, as the simulation stores them (the maps cell-average instead)
    thermal = initial_thermal_layer(SIM_PARAMS, 512, 512, fine_factor=1)
    # frame 0 of the BubbleML_2 simulations: 0.922, 0.766, 0.609 ... = 1 - y / 0.2 at the cell centers
    expected = torch.tensor([0.921875, 0.765625, 0.609375, 0.453125, 0.296875, 0.140625, 0.0])
    torch.testing.assert_close(thermal[:7, column_of(0.0, 512)], expected, atol=1e-4, rtol=0)


def test_geometry_maps_replace_thermal_layer_with_heater_coverage():
    normalizer = get_normalizer(yaml.safe_load(open(NORMALIZER_CONFIG)))
    initial_state = condition_maps(SIM_PARAMS, normalizer, 64, 64, kind="initial_state")
    geometry = condition_maps(SIM_PARAMS, normalizer, 64, 64, kind="geometry")
    torch.testing.assert_close(geometry[..., :3], initial_state[..., :3])
    mask = geometry[..., 3]
    assert mask[0, column_of(0.0, 64)] == 1.0
    assert mask[0, column_of(-7.5, 64)] == 0.0
    assert mask[1:].abs().max() == 0.0

    # 5.25 is a cell boundary at 64x64, so move the edge to the middle of a cell
    narrower_heater = {**SIM_PARAMS, "heater": {**SIM_PARAMS["heater"], "xMax": 5.125}}
    edge_mask = condition_maps(narrower_heater, normalizer, 64, 64, kind="geometry")[..., 3]
    torch.testing.assert_close(edge_mask[0, column_of(5.125, 64)], torch.tensor(0.5))


def test_unknown_map_kind_is_rejected():
    normalizer = get_normalizer(yaml.safe_load(open(NORMALIZER_CONFIG)))
    with pytest.raises(ValueError):
        condition_maps(SIM_PARAMS, normalizer, 64, 64, kind="thermal")


def test_thermal_layer_survives_downsampling_to_64():
    _, _, thermal = raw_maps(64)
    assert torch.allclose(thermal[0, column_of(0.0, 64)], torch.tensor(0.4), atol=0.02)
    assert thermal[0, column_of(-7.5, 64)].abs() < 1e-5  # outside the heater
    assert thermal[1:].abs().max() < 1e-5  # the layer is thinner than one 64x64 cell
