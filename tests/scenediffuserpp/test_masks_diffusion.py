import pytest
import torch

from topoworld.scenediffuserpp.diffusion import cosine_alpha_sigma
from topoworld.scenediffuserpp.diffusion import forward_noise
from topoworld.scenediffuserpp.diffusion import recover_clean
from topoworld.scenediffuserpp.diffusion import sparse_v_loss
from topoworld.scenediffuserpp.diffusion import velocity_target
from topoworld.scenediffuserpp.masks import behavior_prediction_mask
from topoworld.scenediffuserpp.masks import mixed_multitensor_training_masks
from topoworld.scenediffuserpp.masks import random_control_mask
from topoworld.scenediffuserpp.masks import scene_generation_mask


def test_cosine_schedule_has_clean_and_noise_endpoints():
    alpha, sigma = cosine_alpha_sigma(torch.tensor([0.0, 1.0]))

    torch.testing.assert_close(alpha, torch.tensor([1.0, 0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(sigma, torch.tensor([0.0, 1.0]), atol=1e-6, rtol=0)


def test_cosine_schedule_preserves_unit_circle_in_float64():
    times = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    alpha, sigma = cosine_alpha_sigma(times)

    torch.testing.assert_close(
        alpha.square() + sigma.square(),
        torch.ones_like(times),
        atol=1e-12,
        rtol=0,
    )


def test_velocity_parameterization_exactly_recovers_clean_tensor():
    clean = torch.tensor(
        [[[[0.2, -0.5], [0.7, 0.1]]], [[[0.8, -0.3], [-0.2, 0.9]]]],
        dtype=torch.float64,
    )
    noise = torch.flip(clean, dims=(-1,))
    times = torch.tensor([0.2, 0.8], dtype=torch.float64)

    noisy = forward_noise(clean, noise, times)
    velocity = velocity_target(clean, noise, times)

    torch.testing.assert_close(recover_clean(noisy, velocity, times), clean, atol=1e-12, rtol=0)


def test_bp_mask_keeps_exactly_first_eleven_frames():
    mask = behavior_prediction_mask(entities=3, timesteps=91, channels=12, history=11)

    assert mask.dtype == torch.bool
    assert mask[:, :11].all()
    assert not mask[:, 11:].any()


def test_bp_mask_inpaints_history_validity_channel():
    mask = behavior_prediction_mask(entities=2, timesteps=6, channels=4, history=3)

    assert mask[:, :3, -1].all()
    assert not mask[:, 3:, -1].any()


def test_scene_generation_mask_uses_only_explicit_generator():
    first = scene_generation_mask(
        entities=8,
        timesteps=91,
        channels=12,
        context_entities=3,
        generator=torch.Generator().manual_seed(17),
    )
    second = scene_generation_mask(
        entities=8,
        timesteps=91,
        channels=12,
        context_entities=3,
        generator=torch.Generator().manual_seed(17),
    )

    torch.testing.assert_close(first, second)
    assert first[:, 0, 0].sum().item() == 3
    assert torch.equal(first[:, 0, 0], first[:, -1, -1])


def test_random_control_mask_never_reveals_outside_base_context():
    base = behavior_prediction_mask(entities=4, timesteps=20, channels=6, history=5)
    controlled = random_control_mask(
        base,
        keep_probability=0.5,
        generator=torch.Generator().manual_seed(9),
    )

    assert not (controlled & ~base).any()


def test_mixed_masks_use_behavior_prediction_when_scene_probability_is_zero():
    agent_validity = torch.tensor(
        [
            [
                [True, True, True, True],
                [True, True, False, False],
                [False, False, False, False],
            ]
        ]
    )
    light_validity = torch.ones(1, 2, 4, dtype=torch.bool)

    result = mixed_multitensor_training_masks(
        agent_validity=agent_validity,
        light_validity=light_validity,
        agent_channels=3,
        light_channels=2,
        history=2,
        scene_generation_probability=0.0,
        control_feature_probability=1.0,
        generator=torch.Generator().manual_seed(5),
    )

    assert result.task_is_scene_generation.tolist() == [False]
    assert not result.agent_mask[:, :, 2:].any()
    assert not result.light_mask[:, :, 2:].any()
    assert not result.agent_mask[:, 2].any()


def test_mixed_masks_scene_generation_selects_only_valid_entities():
    agent_validity = torch.tensor(
        [
            [
                [True, True, True, True],
                [True, False, False, False],
                [False, False, False, False],
            ]
        ]
    )
    light_validity = torch.tensor(
        [[[True, True, True, True], [False, False, False, False]]], dtype=torch.bool
    )

    result = mixed_multitensor_training_masks(
        agent_validity=agent_validity,
        light_validity=light_validity,
        agent_channels=3,
        light_channels=2,
        history=2,
        scene_generation_probability=1.0,
        control_feature_probability=1.0,
        generator=torch.Generator().manual_seed(17),
    )

    assert result.task_is_scene_generation.tolist() == [True]
    assert not result.agent_mask[:, 2].any()
    assert not result.light_mask[:, 1].any()
    for mask in (result.agent_mask, result.light_mask):
        entity_is_known = mask.any(dim=(2, 3))
        for entity in range(mask.shape[1]):
            if entity_is_known[0, entity]:
                assert mask[0, entity].any()


def test_mixed_masks_random_control_is_factorized_and_reproducible():
    validity = torch.ones(2, 4, 6, dtype=torch.bool)
    kwargs = dict(
        agent_validity=validity,
        light_validity=validity[:, :2],
        agent_channels=5,
        light_channels=3,
        history=3,
        scene_generation_probability=0.5,
        control_feature_probability=0.5,
    )

    first = mixed_multitensor_training_masks(
        **kwargs, generator=torch.Generator().manual_seed(29)
    )
    second = mixed_multitensor_training_masks(
        **kwargs, generator=torch.Generator().manual_seed(29)
    )

    torch.testing.assert_close(first.agent_mask, second.agent_mask)
    torch.testing.assert_close(first.light_mask, second.light_mask)
    torch.testing.assert_close(
        first.task_is_scene_generation, second.task_is_scene_generation
    )
    for mask in (first.agent_mask, first.light_mask):
        for batch in range(mask.shape[0]):
            active = mask[batch]
            entity_factor = active.any(dim=(1, 2))
            time_factor = active.any(dim=(0, 2))
            channel_factor = active.any(dim=(0, 1))
            reconstructed = (
                entity_factor[:, None, None]
                & time_factor[None, :, None]
                & channel_factor[None, None, :]
            )
            assert not (active & ~reconstructed).any()


def test_mixed_masks_sample_both_tasks_at_the_configured_half_probability():
    batch_size = 8192
    validity = torch.ones(batch_size, 2, 4, dtype=torch.bool)

    result = mixed_multitensor_training_masks(
        agent_validity=validity,
        light_validity=validity[:, :1],
        agent_channels=3,
        light_channels=2,
        history=2,
        scene_generation_probability=0.5,
        control_feature_probability=0.5,
        generator=torch.Generator().manual_seed(20260720),
    )

    scene_fraction = result.task_is_scene_generation.float().mean().item()
    assert 0.48 <= scene_fraction <= 0.52
    assert result.task_is_scene_generation.any()
    assert (~result.task_is_scene_generation).any()


def test_invalid_steps_supervise_only_validity_channel():
    prediction = torch.ones(1, 1, 1, 4)
    target = torch.zeros_like(prediction)
    validity = torch.tensor([[[-1.0]]])

    loss, parts = sparse_v_loss(prediction, target, validity, inpaint_mask=None)

    assert loss.item() == 1.0
    assert parts["value_count"] == 0
    assert parts["validity_count"] == 1


def test_sparse_loss_keeps_inpainted_entries_under_published_sparse_weighting():
    prediction = torch.ones(1, 1, 2, 3)
    target = torch.zeros_like(prediction)
    validity = torch.ones(1, 1, 2)
    inpaint = torch.zeros_like(prediction, dtype=torch.bool)
    inpaint[..., 0, :] = True

    loss, parts = sparse_v_loss(prediction, target, validity, inpaint_mask=inpaint)

    assert loss.item() == 1.0
    assert parts["value_count"] == 4
    assert parts["validity_count"] == 2


def test_sparse_loss_can_upweight_validity_transitions():
    prediction = torch.zeros(1, 1, 3, 2)
    target = torch.zeros_like(prediction)
    prediction[..., -1] = torch.tensor([[[0.0, 1.0, 0.0]]])
    validity = torch.tensor([[[1.0, -1.0, -1.0]]])

    baseline_loss, _ = sparse_v_loss(
        prediction,
        target,
        validity,
        inpaint_mask=None,
    )
    weighted_loss, parts = sparse_v_loss(
        prediction,
        target,
        validity,
        inpaint_mask=None,
        validity_transition_weight=5.0,
    )

    assert parts["validity_transition_count"] == 1
    assert parts["validity_weight_sum"] == 7.0
    assert weighted_loss > baseline_loss
    torch.testing.assert_close(parts["validity_loss"], torch.tensor(5.0 / 7.0))


def test_sparse_loss_rejects_invalid_validity_transition_weight():
    prediction = torch.zeros(1, 1, 1, 2)
    target = torch.zeros_like(prediction)
    validity = torch.ones(1, 1, 1)

    with pytest.raises(ValueError, match="validity_transition_weight"):
        sparse_v_loss(
            prediction,
            target,
            validity,
            inpaint_mask=None,
            validity_transition_weight=0.5,
        )
