import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from inf_fm import integrate_flow
from nucleus.data.normalize import get_normalizer
from nucleus.models import get_model
from nucleus.models.flow_matching_module import FlowMatchingModule
from nucleus.models.modules import default_input_augmentations
from nucleus.noise import LogUniformNoise, sdf_context_corruptions

FIELDS = ("dfun", "temperature", "velx", "vely")
WINDOW = 4
DEVICE = "cuda"
NUCLEUS_CONFIG = Path("config/model_cfg/nucleus2/nucleus2_experiment.yaml")
# the small NUCLEUS control used in treasured-bird-24, sized like the flow-matching model
NUCLEUS_SIZE = dict(embed_dim=256, moe_intermediate_dim=1024, processor_blocks=6, num_heads=4, num_experts=4, topk=2)


@dataclass
class Forecaster:
    model: nn.Module
    training_loss: Callable  # batch -> (loss to optimize, forecast MSE for logging)
    forecast: Callable  # (contexts, num_flow_steps, generator) -> next window for each context
    step_options: tuple  # Euler step counts to evaluate; (None,) for a single forward pass
    num_seeds: int


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("flow", "nucleus"), default="flow")
    parser.add_argument("--data", type=Path,
                        default=Path.home() / "BubbleML_2_downsample64/PoolBoiling-Subcooled-OP250-2D/Twall_85.hdf5")
    parser.add_argument("--normalizer", type=Path, default=Path("config/normalizer_cfg/op250_64_nondim.yaml"))
    parser.add_argument("--model-config", type=Path,
                        default=Path("config/model_cfg/flow_matching/nucleus2_moe_fm_concat_small.yaml"))
    parser.add_argument("--start-frame", type=int, default=1600)
    parser.add_argument("--num-windows", type=int, default=4, help="consecutive windows for the rollout phase")
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--prediction-target", choices=("velocity", "endpoint"), default="endpoint")
    parser.add_argument("--full-field", action="store_true", help="flow toward the target frame, not the residual")
    parser.add_argument("--flow-time-power", type=float, default=1.0)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--context-noise", action="store_true", help="flow only: NUCLEUS's white input noise on the context")
    parser.add_argument("--sdf-corruptions", action="store_true", help="both models: SDF-aware context corruptions")
    return parser.parse_args()


def context_corruptions(args, normalizer, include_white_noise):
    constants = normalizer.constants
    white_noise = [aug for aug in default_input_augmentations() if isinstance(aug, LogUniformNoise)]
    return (white_noise if include_white_noise else []) + (
        sdf_context_corruptions(constants.sdf_mean, constants.sdf_std, constants.max_domain_size)
        if args.sdf_corruptions else [])


def load_windows(args, normalizer, num_windows):
    last_frame = args.start_frame + WINDOW * (num_windows + 1)
    with h5py.File(args.data, "r") as handle:
        raw = torch.from_numpy(np.stack([handle[field][args.start_frame:last_frame] for field in FIELDS], axis=-1))
    frames = normalizer.normalize(raw[None], 0.0)[0].to(DEVICE)
    contexts = torch.stack([frames[WINDOW * k:WINDOW * (k + 1)] for k in range(num_windows)])
    targets = torch.stack([frames[WINDOW * (k + 1):WINDOW * (k + 2)] for k in range(num_windows)])
    return contexts, targets


def build_flow_forecaster(args, sim_params_for, normalizer):
    model_cfg = yaml.safe_load(open(args.model_config))
    model = get_model(model_cfg["name"], **dict(model_cfg["params"], num_context_frames=WINDOW)).to(DEVICE)
    # reuse the training module's own flow sampling, context corruption and loss, without Lightning's setup
    objective = FlowMatchingModule.__new__(FlowMatchingModule)
    nn.Module.__init__(objective)
    objective.model, objective.prediction_target = model, args.prediction_target
    objective.residual, objective.flow_time_power = not args.full_field, args.flow_time_power
    objective.context_augmentations = context_corruptions(args, normalizer, include_white_noise=args.context_noise)

    def training_loss(batch):
        prediction = objective._predict(objective._augment_context(batch), args.flow_time_power)
        return F.mse_loss(*objective._trained_pair(prediction)), F.mse_loss(prediction.delta_pred, prediction.delta).detach()

    def forecast(contexts, num_steps, generator):
        return integrate_flow(model, contexts, sim_params_for(model, contexts.shape[0]), WINDOW, num_steps,
                              objective.residual, objective.prediction_target, generator)

    return Forecaster(model, training_loss, forecast, (20, 50), args.num_seeds)


def build_nucleus_forecaster(args, sim_params_for, normalizer):
    model_cfg = yaml.safe_load(open(NUCLEUS_CONFIG))
    model = get_model(model_cfg["name"], **dict(model_cfg["params"], **NUCLEUS_SIZE)).to(DEVICE)
    augmentations = default_input_augmentations() + context_corruptions(args, normalizer, include_white_noise=False)

    def training_loss(batch):
        with torch.no_grad():
            context = batch.input
            for augmentation in augmentations:
                context = augmentation(context)
        prediction, _ = model.step(context, batch.sim_params_tensor)
        return F.l1_loss(prediction, batch.target), F.mse_loss(prediction, batch.target).detach()

    @torch.inference_mode()
    def forecast(contexts, num_steps, generator):
        return model.step(contexts, sim_params_for(model, contexts.shape[0]))[0]

    return Forecaster(model, training_loss, forecast, (None,), 1)


def train(forecaster, contexts, targets, sim_params, args):
    optimizer = torch.optim.AdamW(forecaster.model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_steps)
    window_index = torch.arange(args.batch_size, device=DEVICE) % contexts.shape[0]
    log_interval = max(1, args.train_steps // 8)
    interval_loss, interval_error = 0.0, 0.0
    forecaster.model.train()
    for step in range(1, args.train_steps + 1):
        batch = SimpleNamespace(input=contexts[window_index], target=targets[window_index], sim_params_tensor=sim_params)
        loss, forecast_error = forecaster.training_loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        interval_loss, interval_error = interval_loss + loss.detach(), interval_error + forecast_error
        if step % log_interval == 0:
            print(f"  steps {step - log_interval + 1:>5}-{step:>5}: mean training loss {interval_loss.item() / log_interval:.5f}"
                  f"  mean forecast MSE {interval_error.item() / log_interval:.5f}")
            interval_loss, interval_error = 0.0, 0.0
    forecaster.model.eval()


def describe(samples, targets):
    per_field = ((samples - targets[:, None]) ** 2).mean(dim=(0, 1, 2, 3, 4)).cpu().numpy()
    spread = samples.var(dim=1).mean().item() if samples.shape[1] > 1 else 0.0
    fields = "  ".join(f"{field}={error:.5f}" for field, error in zip(FIELDS, per_field))
    return f"error {per_field.mean():.5f} ({fields})  spread between seeds {spread:.6f}"


def step_label(num_steps):
    return "one forward pass" if num_steps is None else f"{num_steps} Euler steps"


def evaluate_true_context(forecaster, contexts, targets):
    persistence = F.mse_loss(contexts[:, -1:].expand_as(targets), targets).item()
    print(f"  copy-last-frame error {persistence:.5f}")
    for num_steps in forecaster.step_options:
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        repeated = contexts.repeat_interleave(forecaster.num_seeds, dim=0)
        samples = forecaster.forecast(repeated, num_steps, generator).unflatten(0, (contexts.shape[0], forecaster.num_seeds))
        print(f"  {step_label(num_steps)}, {forecaster.num_seeds} seed(s): {describe(samples, targets)}")


def evaluate_rollout(forecaster, contexts, targets):
    num_steps = forecaster.step_options[-1]
    print(f" autoregressive rollout from window 1's true context ({step_label(num_steps)}):")
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    context = contexts[:1].repeat_interleave(forecaster.num_seeds, dim=0)
    for window in range(contexts.shape[0]):
        context = forecaster.forecast(context, num_steps, generator)
        print(f"  window {window + 1}: {describe(context[None], targets[window:window + 1])}")


def run_phase(label, num_windows, args, normalizer, sim_params_dict):
    print(f"\n=== {label}: training on {num_windows} window(s) from frame {args.start_frame} ===")
    torch.manual_seed(0)
    contexts, targets = load_windows(args, normalizer, num_windows)
    sim_params_for = lambda model, batch_size: model._normalized_sim_params(sim_params_dict, normalizer, DEVICE, batch_size)
    build = build_flow_forecaster if args.model == "flow" else build_nucleus_forecaster
    forecaster = build(args, sim_params_for, normalizer)
    train(forecaster, contexts, targets, sim_params_for(forecaster.model, args.batch_size), args)
    print(" forecasts from the true context of each trained window:")
    evaluate_true_context(forecaster, contexts, targets)
    if num_windows > 1:
        evaluate_rollout(forecaster, contexts, targets)


def main():
    args = parse_args()
    normalizer = get_normalizer(yaml.safe_load(open(args.normalizer)))
    sim_params_dict = json.load(open(str(args.data).replace(".hdf5", ".json")))
    if args.model == "flow":
        mode = f"flow matching, {'full-field' if args.full_field else 'residual'} / {args.prediction_target} / t power {args.flow_time_power}"
    else:
        mode = "NUCLEUS regression (L1, input noise + field dropout)"
    corruptions = [name for name, enabled in (("white context noise", args.context_noise and args.model == "flow"),
                                               ("SDF corruptions", args.sdf_corruptions)) if enabled]
    print(f"rollout sanity check: {mode}; extra context corruptions: {', '.join(corruptions) or 'none'}")
    run_phase("single pair", 1, args, normalizer, sim_params_dict)
    run_phase("short sequence", args.num_windows, args, normalizer, sim_params_dict)


if __name__ == "__main__":
    main()
