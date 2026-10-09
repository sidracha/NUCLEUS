import torch

from nucleus.models import get_model
from nucleus.models.nucleus2_moe_fm import Nucleus2MoEFlowMatchingConcat

NUM_CONTEXT_FRAMES = 4
NUM_FIELDS = 4


def build_small_concat_model():
    return get_model(
        "nucleus2_moe_fm_concat",
        patch_size=4, embed_dim=64, moe_intermediate_dim=64, processor_blocks=1,
        num_heads=4, num_experts=2, topk=1, patching="Linear",
        num_context_frames=NUM_CONTEXT_FRAMES,
    )


def test_embedding_takes_noisy_plus_context_channels():
    model = build_small_concat_model()
    expected_inputs = NUM_FIELDS * (1 + NUM_CONTEXT_FRAMES) * 4 * 4
    assert model.embed.linear.in_features == expected_inputs


def test_network_frames_pair_each_noisy_frame_with_full_history():
    model = build_small_concat_model()
    context = torch.randn(2, NUM_CONTEXT_FRAMES, 8, 8, NUM_FIELDS)
    noisy_frames = torch.randn(2, 3, 8, 8, NUM_FIELDS)
    frames, num_dropped = model._network_frames(torch.cat([context, noisy_frames], dim=1), NUM_CONTEXT_FRAMES, None)

    assert num_dropped == 0
    assert frames.shape == (2, 3, 8, 8, NUM_FIELDS * (1 + NUM_CONTEXT_FRAMES))
    torch.testing.assert_close(frames[..., :NUM_FIELDS], noisy_frames)
    for output_frame in range(3):
        for context_frame in range(NUM_CONTEXT_FRAMES):
            channels = slice(NUM_FIELDS * (1 + context_frame), NUM_FIELDS * (2 + context_frame))
            torch.testing.assert_close(frames[:, output_frame, ..., channels], context[:, context_frame])


def test_condition_channels_widen_embeddings_and_reach_every_output_token():
    concat_model = get_model(
        "nucleus2_moe_fm_concat",
        patch_size=4, embed_dim=64, moe_intermediate_dim=64, processor_blocks=1,
        num_heads=4, num_experts=2, topk=1, patching="Linear",
        num_context_frames=NUM_CONTEXT_FRAMES, num_condition_channels=4,
    )
    nucleus = get_model(
        "nucleus2_moe", patch_size=4, embed_dim=64, moe_intermediate_dim=64, processor_blocks=1,
        num_heads=4, num_experts=2, topk=1, patching="Linear", num_condition_channels=4,
    )
    assert concat_model.embed.linear.in_features == (NUM_FIELDS * (1 + NUM_CONTEXT_FRAMES) + 4) * 16
    assert nucleus.embed.linear.in_features == (NUM_FIELDS + 4) * 16

    condition = torch.randn(2, 8, 8, 4)
    frames, _ = concat_model._network_frames(torch.randn(2, NUM_CONTEXT_FRAMES + 3, 8, 8, NUM_FIELDS),
                                             NUM_CONTEXT_FRAMES, condition)
    for output_frame in range(3):
        torch.testing.assert_close(frames[:, output_frame, ..., -4:], condition)


def test_checkpoint_state_keeps_context_frame_count():
    model = build_small_concat_model()
    restored = build_small_concat_model()
    restored.config.num_context_frames = 99
    restored.load_state_dict(model.state_dict())
    assert isinstance(restored, Nucleus2MoEFlowMatchingConcat)
    assert restored.config.num_context_frames == NUM_CONTEXT_FRAMES
