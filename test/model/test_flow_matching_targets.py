from types import SimpleNamespace

import torch

from nucleus.models.flow_matching_module import FlowMatchingModule

NUM_INPUT_FRAMES = 4
FIELD_SHAPE = (2, 4, 8, 8, 4)


def make_flow_sample():
    torch.manual_seed(0)
    context = torch.randn(FIELD_SHAPE)
    delta = torch.randn(FIELD_SHAPE)
    noise = torch.randn(FIELD_SHAPE)
    flow_time = torch.tensor([[0.2], [0.8]])
    weight = flow_time[:, :, None, None, None]
    zt = (1.0 - weight) * noise + weight * delta
    model_input = torch.cat([context, zt], dim=1)
    return context, model_input, flow_time, delta, delta - noise


def build_module(prediction_target: str, network_output: torch.Tensor, flow_sample) -> FlowMatchingModule:
    context, model_input, flow_time, _, v_target = flow_sample
    module = FlowMatchingModule.__new__(FlowMatchingModule)
    torch.nn.Module.__init__(module)
    module.prediction_target = prediction_target
    module.model = SimpleNamespace(step=lambda *args, **kwargs: (network_output, []))
    module._sample_flow = lambda batch, flow_time_power=1.0: (model_input, flow_time, v_target)
    return module


def fake_batch(context):
    return SimpleNamespace(input=context, sim_params_tensor=torch.zeros(context.shape[0], 13))


def test_endpoint_target_recovers_true_velocity():
    flow_sample = make_flow_sample()
    context, _, _, delta, v_target = flow_sample
    prediction = build_module("endpoint", delta, flow_sample)._predict(fake_batch(context))
    torch.testing.assert_close(prediction.delta, delta)
    torch.testing.assert_close(prediction.v_pred, v_target)


def test_velocity_target_recovers_true_endpoint():
    flow_sample = make_flow_sample()
    context, _, _, delta, v_target = flow_sample
    prediction = build_module("velocity", v_target, flow_sample)._predict(fake_batch(context))
    torch.testing.assert_close(prediction.delta_pred, delta)


def test_flow_time_power_biases_samples_toward_small_t():
    torch.manual_seed(0)
    module = FlowMatchingModule.__new__(FlowMatchingModule)
    module.residual = True
    batch = SimpleNamespace(input=torch.zeros(20000, 1, 1, 1, 1), target=torch.zeros(20000, 1, 1, 1, 1))
    _, uniform_time, _ = module._sample_flow(batch, flow_time_power=1.0)
    _, biased_time, _ = module._sample_flow(batch, flow_time_power=2.0)
    assert abs((uniform_time < 0.25).float().mean().item() - 0.25) < 0.02
    assert abs((biased_time < 0.25).float().mean().item() - 0.5) < 0.02


def flow_endpoint_from_sample(residual: bool, context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    module = FlowMatchingModule.__new__(FlowMatchingModule)
    module.residual = residual
    model_input, flow_time, v_target = module._sample_flow(SimpleNamespace(input=context, target=target))
    zt = model_input[:, context.shape[1]:]
    return zt + (1.0 - flow_time)[:, :, None, None, None] * v_target


def test_full_field_flow_ends_at_clean_target():
    torch.manual_seed(0)
    context, target = torch.randn(FIELD_SHAPE), torch.randn(FIELD_SHAPE)
    torch.testing.assert_close(flow_endpoint_from_sample(False, context, target), target)
    torch.testing.assert_close(flow_endpoint_from_sample(True, context, target), target - context[:, -1:])


def test_context_noise_changes_only_the_context():
    from nucleus.noise import LogUniformNoise
    torch.manual_seed(0)
    module = FlowMatchingModule.__new__(FlowMatchingModule)
    module.context_augmentations = [LogUniformNoise(0.001, 5e-2, skip_prob=0.0)]
    context, target = torch.randn(FIELD_SHAPE), torch.randn(FIELD_SHAPE)
    batch = module._augment_context(SimpleNamespace(input=context.clone(), target=target.clone()))
    assert not torch.allclose(batch.input, context)
    assert (batch.input - context).abs().max() < 0.5
    torch.testing.assert_close(batch.target, target)


def test_trained_pair_matches_prediction_target():
    flow_sample = make_flow_sample()
    context, _, _, delta, v_target = flow_sample
    endpoint_module = build_module("endpoint", delta, flow_sample)
    trained_pred, trained_target = endpoint_module._trained_pair(endpoint_module._predict(fake_batch(context)))
    torch.testing.assert_close(trained_pred, delta)
    torch.testing.assert_close(trained_target, delta)
