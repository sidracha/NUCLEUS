import pathlib

import torch
import yaml

from nucleus.data.normalize import NonDimensionalTempNormalizer, get_normalizer

CONFIG_PATH = pathlib.Path(__file__).parents[2] / "config" / "normalizer_cfg" / "op250_64_nondim.yaml"


def load_normalizer():
    with open(CONFIG_PATH) as handle:
        return get_normalizer(yaml.safe_load(handle))


def test_config_builds_nondim_normalizer():
    assert isinstance(load_normalizer(), NonDimensionalTempNormalizer)


def test_roundtrip_recovers_fields():
    normalizer = load_normalizer()
    fields = torch.randn(2, 4, 8, 8, 4)
    restored = normalizer.unnormalize(normalizer.normalize(fields, 41.0), 41.0)
    torch.testing.assert_close(restored, fields)


def test_temperature_ignores_dimensional_bulk_temp():
    normalizer = load_normalizer()
    fields = torch.rand(1, 4, 8, 8, 4)
    cold = normalizer.normalize(fields, -196.0)
    hot = normalizer.normalize(fields, 58.0)
    torch.testing.assert_close(cold[..., 1], hot[..., 1])
