import pytest
import torch

from nucleus.models import get_model

NUM_FIELDS = 4
NUM_CONTEXT_FRAMES = 4
SMALL_MODEL = dict(patch_size=4, embed_dim=64, moe_intermediate_dim=64, processor_blocks=1,
                   num_heads=4, num_experts=2, topk=1, patching="Linear", num_condition_channels=4)


def build_additive_models():
    nucleus = get_model("nucleus2_moe", condition_injection="additive", **SMALL_MODEL)
    flow = get_model("nucleus2_moe_fm_concat", condition_injection="additive",
                     num_context_frames=NUM_CONTEXT_FRAMES, **SMALL_MODEL)
    return nucleus, flow


def test_additive_injection_keeps_condition_out_of_the_field_embedding():
    nucleus, flow = build_additive_models()
    assert nucleus.embed.linear.in_features == NUM_FIELDS * 16
    assert flow.embed.linear.in_features == NUM_FIELDS * (1 + NUM_CONTEXT_FRAMES) * 16


@pytest.mark.parametrize("model_index", [0, 1])
def test_additive_condition_starts_as_a_no_op(model_index):
    model = build_additive_models()[model_index]
    frames = torch.randn(2, 3, 8, 8, model.embed.linear.in_features // 16)
    condition = torch.randn(2, 8, 8, 4)
    torch.testing.assert_close(model._encode(frames, condition), model._encode(frames, None))


def test_additive_condition_is_learnable_and_shared_across_frames():
    nucleus, _ = build_additive_models()
    frames = torch.randn(2, 3, 8, 8, NUM_FIELDS)
    condition = torch.randn(2, 8, 8, 4)
    nucleus._encode(frames, condition).sum().backward()
    assert nucleus.condition_encoder.project.weight.grad.abs().sum() > 0

    torch.nn.init.normal_(nucleus.condition_encoder.project.weight)
    with torch.no_grad():
        offset = nucleus._encode(frames, condition) - nucleus._encode(frames, None)
    for frame in range(1, 3):
        torch.testing.assert_close(offset[:, frame], offset[:, 0])
