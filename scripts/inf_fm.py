import json
import pathlib
from types import SimpleNamespace

import h5py
import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from inf import save_trajectory_as_hdf5
from nucleus.data.conditioning import condition_map_kind, condition_maps
from nucleus.data.normalize import get_normalizer
from nucleus.models import load_model_from_checkpoint
from nucleus.plot.plotting import plot_rollout, plot_rollout_stability
from nucleus.utils.set_fp32_precision import set_fp32_precision
from nucleus.utils.sdf_reinit import sdf_reinit_sussman

FIELDS = ("dfun", "temperature", "velx", "vely")


def load_test_case(test_file_path: str) -> tuple[torch.Tensor, dict]:
    with h5py.File(test_file_path, "r") as handle:
        trajectory = torch.stack([torch.from_numpy(handle[field][:]) for field in FIELDS], dim=-1)
    with open(test_file_path.replace(".hdf5", ".json"), "r") as handle:
        sim_params_dict = json.load(handle)
    return trajectory, sim_params_dict


@torch.inference_mode()
def integrate_flow(
    model,
    context: torch.Tensor,
    sim_params: torch.Tensor,
    num_output_frames: int,
    num_flow_steps: int,
    residual: bool,
    prediction_target: str,
    generator: torch.Generator,
    noise_std: float | list = 1.0,
    condition: torch.Tensor | None = None,
    solver: str = "rk4",
) -> torch.Tensor:
    batch_size, num_input_frames, height, width, channels = context.shape
    raw = torch.randn(
        batch_size, num_output_frames, height, width, channels,
        device=context.device, generator=generator,
    )
    if isinstance(noise_std, (list, tuple)):
        std = torch.tensor(noise_std, device=context.device, dtype=raw.dtype)[None, None, None, None, :]
    else:
        std = noise_std
    state = std * raw
    step_size = 1.0 / num_flow_steps

    def velocity_at(s: torch.Tensor, t_scalar: float) -> torch.Tensor:
        flow_time = torch.full((batch_size, 1), t_scalar, device=context.device)
        model_input = torch.cat([context, s], dim=1)
        v, _ = model.step(model_input, sim_params, flow_time, T_in=num_input_frames, condition=condition)
        if prediction_target == "endpoint":
            remaining = max(1.0 - t_scalar, 1e-3)
            return (v - s) / remaining
        return v

    for step_index in range(num_flow_steps):
        t = step_index * step_size
        if solver == "rk4":
            k1 = velocity_at(state, t)
            k2 = velocity_at(state + 0.5 * step_size * k1, t + 0.5 * step_size)
            k3 = velocity_at(state + 0.5 * step_size * k2, t + 0.5 * step_size)
            k4 = velocity_at(state + step_size * k3, t + step_size)
            state = state + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            state = state + step_size * velocity_at(state, t)

    if residual:
        return context[:, -1:] + state
    return state


def noise_std_from_cfg(model_cfg: DictConfig) -> float | list:
    noise_std = model_cfg.get("noise_std", 1.0)
    # hydra hands lists over as ListConfig, which integrate_flow's list check doesn't recognise
    if isinstance(noise_std, ListConfig):
        return OmegaConf.to_container(noise_std)
    return noise_std


def rollout_condition(model, sim_params_dict: dict, normalizer, trajectory_shape, device):
    map_kind = condition_map_kind(model.config)
    if map_kind is None:
        return None
    _, height, width, _ = trajectory_shape
    return condition_maps(sim_params_dict, normalizer, height, width, map_kind)[None].to(device)


def rollout(model, ground_truth: torch.Tensor, sim_params_dict: dict, normalizer, cfg: DictConfig) -> torch.Tensor:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    bulk_temp = sim_params_dict["bulk_temp"]
    sim_params = model._normalized_sim_params(sim_params_dict, normalizer, device, batch_size=1)

    condition = rollout_condition(model, sim_params_dict, normalizer, ground_truth.shape, device)

    _, height, width, _ = ground_truth.shape
    dx = (sim_params_dict["x_max"] - sim_params_dict["x_min"]) / width

    initial_window = slice(cfg.start_time, cfg.start_time + cfg.history_time_window)
    trajectory = ground_truth[initial_window][None].to(device)
    while trajectory.shape[1] < cfg.trajectory_steps:
        context = normalizer.normalize(trajectory[:, -cfg.history_time_window:], bulk_temp)
        normalized_forecast = integrate_flow(
            model, context, sim_params,
            cfg.future_time_window, cfg.num_flow_steps, cfg.model_cfg.get("residual", True),
            cfg.model_cfg.get("prediction_target", "velocity"), generator,
            noise_std=noise_std_from_cfg(cfg.model_cfg), condition=condition,
            solver=cfg.get("solver", "rk4"),
        )
        forecast = normalizer.unnormalize(normalized_forecast, bulk_temp)
        # reinitialize SDF in physical space so ||∇φ||≈1 is maintained across windows
        forecast[..., 0] = sdf_reinit_sussman(forecast[..., 0], dx=dx, n_iter=5)
        trajectory = torch.cat([trajectory, forecast], dim=1)

    return trajectory[0, :cfg.trajectory_steps].cpu()


def plot_case(case_root: pathlib.Path, sim_params_dict: dict, prediction, target, plot_every: int):
    plot_dir = case_root / "plots"
    plot_dir.mkdir(exist_ok=True)
    # BubbleML_2 temperature is non-dimensional (0 = bulk, 1 = wall), but plot_temp takes
    # its color range from the dimensional bulk/wall temps in the sim params
    nondim_temp_range = {**sim_params_dict, "bulk_temp": 0.0, "heater": {**sim_params_dict["heater"], "wallTemp": 1.0}}
    # plot_rollout only reads .fluid_params and .targets from its results argument
    plot_inputs = SimpleNamespace(fluid_params=nondim_temp_range, targets=target)
    plot_rollout(str(plot_dir), prediction, plot_inputs, step_size=plot_every, include_ground_truth=True)
    plot_rollout_stability(str(plot_dir), prediction, target)


def save_case(save_root: pathlib.Path, cfg: DictConfig, sim_params_dict: dict, prediction, target):
    case_name = f"{sim_params_dict['setup']}_{sim_params_dict['liquid']}_{sim_params_dict['heater']['wallTemp']}"
    case_root = save_root / case_name
    case_root.mkdir(parents=True, exist_ok=True)
    if cfg.plot:
        plot_case(case_root, sim_params_dict, prediction, target, cfg.plot_every)
    OmegaConf.save(config=cfg.model_cfg, f=case_root / "model_cfg.yaml")
    save_trajectory_as_hdf5(case_root / "pred_trajectory.hdf5", prediction)
    save_trajectory_as_hdf5(case_root / "gt_trajectory.hdf5", target)
    with open(case_root / "sim_params.json", "w") as handle:
        json.dump(sim_params_dict, handle)
    mse = torch.mean((prediction - target) ** 2).item()
    print(f"{case_name}: rollout mse over {len(target)} frames = {mse:.4e}")


@hydra.main(version_base=None, config_path="../config", config_name="inference_fm")
def main(cfg: DictConfig):
    set_fp32_precision()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_checkpoint(
        cfg.checkpoint_path,
        map_location=device,
        model_cfg=OmegaConf.to_container(cfg.model_cfg, resolve=True),
    ).to(device).eval()
    normalizer = get_normalizer(OmegaConf.to_container(cfg.normalizer_cfg, resolve=True))

    mode = "residual" if cfg.model_cfg.get("residual", True) else "full"
    save_root = pathlib.Path(cfg.checkpoint_path).parent / f"rollouts_fm_{mode}_seed{cfg.seed}"
    save_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, f=save_root / "config.yaml")

    for test_file_path in cfg.data_cfg.test_paths:
        ground_truth, sim_params_dict = load_test_case(test_file_path)
        prediction = rollout(model, ground_truth, sim_params_dict, normalizer, cfg)
        target = ground_truth[cfg.start_time: cfg.start_time + cfg.trajectory_steps]
        save_case(save_root, cfg, sim_params_dict, prediction, target)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
