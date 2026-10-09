from typing import Optional

import torch
import torch.nn.functional as F

from nucleus.data.normalize import Normalizer

NUM_CONDITION_CHANNELS = 4
CONDITION_MAP_KINDS = ("initial_state", "geometry")
# measured from frame 0 of the BubbleML_2 OP-250 simulations: temperature = 1 - y / 0.2 over
# the heater (non-dimensional, so wall and bulk temperatures are implicit)
THERMAL_BOUNDARY_LAYER_THICKNESS = 0.2


def cell_centers(lower: float, upper: float, num_cells: int) -> torch.Tensor:
    spacing = (upper - lower) / num_cells
    return lower + (torch.arange(num_cells, dtype=torch.float64) + 0.5) * spacing


def seed_distance_field(sim_params_dict: dict, x_centers: torch.Tensor, y_centers: torch.Tensor) -> torch.Tensor:
    # the simulator's seed level set max(phi, r - dist) as a plain distance field (no -1e13
    # background); this equals frame 0 of the simulations exactly and stays visible on coarse grids
    sites_x = torch.tensor(sim_params_dict["nuc_sites_x"], dtype=torch.float64)
    radii = torch.tensor(sim_params_dict["nuc_seed_radii"], dtype=torch.float64)
    # seeds are lifted so they meet the wall at the contact angle: center at r * cos(angle) above
    # the site, which reproduces frame 0 of the simulations
    contact_angle = torch.deg2rad(torch.tensor(sim_params_dict["heater"]["rcdAngle"], dtype=torch.float64))
    sites_y = torch.tensor(sim_params_dict["nuc_sites_y"], dtype=torch.float64) + radii * torch.cos(contact_angle)
    distance = torch.sqrt((y_centers[None, :, None] - sites_y[:, None, None]) ** 2
                          + (x_centers[None, None, :] - sites_x[:, None, None]) ** 2)
    return (radii[:, None, None] - distance).amax(dim=0)  # (H, W)


def initial_thermal_layer(sim_params_dict: dict, height: int, width: int, fine_factor: int = 8) -> torch.Tensor:
    # linear layer from 1 at the heater to 0 at the layer thickness, built on a finer grid and
    # averaged down like the data, since at 64x64 the layer is thinner than one cell
    heater = sim_params_dict["heater"]
    if heater["yMax"] > sim_params_dict["y_min"] + 1e-6:
        raise ValueError("conditioning maps assume the heater is on the bottom wall")
    fine_x = cell_centers(sim_params_dict["x_min"], sim_params_dict["x_max"], width * fine_factor)
    fine_y = cell_centers(sim_params_dict["y_min"], sim_params_dict["y_max"], height * fine_factor)
    height_above_wall = fine_y[:, None] - sim_params_dict["y_min"]
    layer = (1.0 - height_above_wall / THERMAL_BOUNDARY_LAYER_THICKNESS).clamp(min=0.0)
    on_heater = (fine_x[None, :] >= heater["xMin"]) & (fine_x[None, :] <= heater["xMax"])
    fine_layer = (layer * on_heater).to(torch.float32)
    return F.avg_pool2d(fine_layer[None, None], fine_factor)[0, 0]


def heater_mask(sim_params_dict: dict, height: int, width: int, fine_factor: int = 8) -> torch.Tensor:
    # fraction of each bottom-row cell covered by the heater, so partly covered edge cells stay partial
    heater = sim_params_dict["heater"]
    fine_x = cell_centers(sim_params_dict["x_min"], sim_params_dict["x_max"], width * fine_factor)
    on_heater = ((fine_x >= heater["xMin"]) & (fine_x <= heater["xMax"])).float()
    mask = torch.zeros(height, width)
    mask[0] = on_heater.reshape(width, fine_factor).mean(dim=1)
    return mask


def condition_maps(sim_params_dict: dict, normalizer: Normalizer, height: int, width: int,
                   kind: str = "initial_state") -> torch.Tensor:
    """Static per-trajectory maps (H, W, 4): x, y, seed SDF, and the initial thermal layer
    ("initial_state") or the heater mask ("geometry")."""
    if kind not in CONDITION_MAP_KINDS:
        raise ValueError(f"unknown condition map kind {kind!r}, expected one of {CONDITION_MAP_KINDS}")
    x_centers = cell_centers(sim_params_dict["x_min"], sim_params_dict["x_max"], width)
    y_centers = cell_centers(sim_params_dict["y_min"], sim_params_dict["y_max"], height)
    x_unit = (2 * (x_centers - sim_params_dict["x_min"]) / (sim_params_dict["x_max"] - sim_params_dict["x_min"]) - 1).float()
    y_unit = (2 * (y_centers - sim_params_dict["y_min"]) / (sim_params_dict["y_max"] - sim_params_dict["y_min"]) - 1).float()
    seeds = normalizer.normalize_sdf(seed_distance_field(sim_params_dict, x_centers, y_centers).float())
    if kind == "geometry":
        heater_channel = heater_mask(sim_params_dict, height, width)
    else:
        # BubbleML temperature is non-dimensional, so no bulk temperature is subtracted here
        heater_channel = normalizer.normalize_temp(initial_thermal_layer(sim_params_dict, height, width), 0.0)
    return torch.stack([
        x_unit[None, :].expand(height, width),
        y_unit[:, None].expand(height, width),
        seeds,
        heater_channel,
    ], dim=-1)


def condition_map_kind(model_config) -> Optional[str]:
    """Which maps a model was trained with, or None for an unconditioned model."""
    if getattr(model_config, "num_condition_channels", 0) == 0:
        return None
    # checkpoints from before the option existed were trained on the initial state
    return getattr(model_config, "condition_maps", "initial_state")
