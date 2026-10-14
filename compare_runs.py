"""
Compare rollout metrics between two inference runs.
  run A: effortless-lion-61   (nucleus2 geom medium)
  run B: earthy-microwave-68  (nucleus2 fm concat medium)

Metrics per case (matching previous compare tables):
  - vapor fraction MAE         (time-averaged |pred_vf - gt_vf|)
  - mean bubble count          (time-averaged pred bubble count)
  - rollout error at frame 199 (MAE between pred/gt at final frame, avg over fields)
  - SDF roughness at frame 199 (mean ||∇φ|| at final frame; truth ≈ 1)
  - liquid temperature at heater MAE
  - mean liquid temperature MAE
  - eikonal                    (mean ||∇φ|| - 1 over full rollout)
"""
import h5py
import json
import numpy as np
import pathlib
from scipy.ndimage import label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

RUN_A = pathlib.Path.home() / "nucleus2_fm_logs/effortless-lion-61/checkpoints/rollouts"
RUN_B = pathlib.Path.home() / "nucleus2_fm_logs/earthy-microwave-68/checkpoints/rollouts_fm_residual_seed0"

FIELDS = ["dfun", "temperature", "velx", "vely"]
# SDF/temperature are stored in physical units; grid is 128×128 cells on a 16×16 domain
DX_PHYS = 16.0 / 128  # 0.125 — physical spacing for gradient-based metrics (eikonal)

def load(case_dir):
    with h5py.File(case_dir / "pred_trajectory.hdf5", "r") as f:
        pred = {k: f[k][:].squeeze(-1) for k in FIELDS}  # (T, H, W)
    with h5py.File(case_dir / "gt_trajectory.hdf5", "r") as f:
        gt = {k: f[k][:].squeeze(-1) for k in FIELDS}
    with open(case_dir / "sim_params.json") as f:
        sim_params = json.load(f)
    return pred, gt, sim_params

# ── physical metrics ─────────────────────────────────────────────────────────

def vapor_fraction(sdf):
    """(T,) fraction of cells in vapor phase."""
    T, H, W = sdf.shape
    return (sdf >= 0).sum(axis=(-2, -1)) / (H * W)

def eikonal_mag(sdf):
    """(T,) mean ||∇φ|| per frame (truth = 1 for a proper SDF)."""
    grad_y, grad_x = np.gradient(sdf, DX_PHYS, axis=(-2, -1))
    return np.sqrt(grad_x**2 + grad_y**2).mean(axis=(-2, -1))

def mean_liquid_temperature(temperature, sdf):
    """(T,) mean temperature in liquid cells."""
    mask = sdf < 0  # liquid
    return np.where(mask.sum(axis=(-2,-1)) > 0,
                    (temperature * mask).sum(axis=(-2,-1)) / mask.sum(axis=(-2,-1)),
                    np.nan)

def liquid_temp_at_heater(temperature, sdf, sim_params):
    """(T,) mean liquid temperature at bottom row within heater x-range."""
    H, W = sdf.shape[-2], sdf.shape[-1]
    x_min_phys = sim_params["x_min"]
    x_max_phys = sim_params["x_max"]
    heater_xmin = sim_params["heater"]["xMin"]
    heater_xmax = sim_params["heater"]["xMax"]
    xs = np.linspace(x_min_phys, x_max_phys, W, endpoint=False) + (x_max_phys - x_min_phys) / (2 * W)
    heater_mask = (xs >= heater_xmin) & (xs <= heater_xmax)
    bottom_liquid = sdf[:, 0, :] < 0  # (T, W) — bottom row
    combined = heater_mask[None, :] & bottom_liquid
    count = combined.sum(axis=-1)
    return np.where(count > 0,
                    (temperature[:, 0, :] * combined).sum(axis=-1) / count,
                    np.nan)

def count_bubbles_at_frame(sdf_frame):
    """Number of distinct vapor regions in one (H, W) SDF frame."""
    vapor = sdf_frame > 0
    if not vapor.any():
        return 0
    coords = peak_local_max(sdf_frame, labels=vapor, min_distance=2)
    mask = np.zeros_like(sdf_frame, dtype=bool)
    if coords.size:
        mask[tuple(coords.T)] = True
    markers, _ = label(mask)
    labels = watershed(-sdf_frame, markers, mask=vapor)
    return int(labels.max())

def mean_bubble_count(sdf):
    """Mean bubble count over all frames."""
    return np.mean([count_bubbles_at_frame(sdf[t]) for t in range(len(sdf))])

def field_mae_at_frame(pred, gt, t=-1):
    """Mean absolute error across all 4 fields at a single frame."""
    errs = [np.abs(pred[k][t] - gt[k][t]).mean() for k in FIELDS]
    return float(np.mean(errs))

def field_mae_rollout(pred, gt):
    """Mean absolute error across all 4 fields averaged over all timesteps."""
    errs = [np.abs(pred[k] - gt[k]).mean() for k in FIELDS]
    return float(np.mean(errs))

# ── per-case metric dict ─────────────────────────────────────────────────────

def compute_metrics(pred, gt, sim_params):
    # vapor fraction MAE (time-averaged)
    vf_pred = vapor_fraction(pred["dfun"])
    vf_gt   = vapor_fraction(gt["dfun"])
    vf_mae  = float(np.abs(vf_pred - vf_gt).mean())

    # mean bubble count (pred)
    bcount = mean_bubble_count(pred["dfun"])

    # rollout error at final frame and over full trajectory
    rollout_err = field_mae_at_frame(pred, gt, t=-1)
    rollout_mae = field_mae_rollout(pred, gt)

    # SDF roughness at final frame (eikonal; truth ≈ 1)
    sdf_roughness = float(eikonal_mag(pred["dfun"][[-1]]).mean())

    # liquid temperature metrics
    lt_pred = mean_liquid_temperature(pred["temperature"], pred["dfun"])
    lt_gt   = mean_liquid_temperature(gt["temperature"],  gt["dfun"])
    lt_mae  = float(np.nanmean(np.abs(lt_pred - lt_gt)))

    lth_pred = liquid_temp_at_heater(pred["temperature"], pred["dfun"], sim_params)
    lth_gt   = liquid_temp_at_heater(gt["temperature"],  gt["dfun"],  sim_params)
    lth_mae  = float(np.nanmean(np.abs(lth_pred - lth_gt)))

    # eikonal (mean over full rollout; truth = 1)
    eik = float(eikonal_mag(pred["dfun"]).mean())

    return dict(
        vf_mae=vf_mae,
        bubble_count=bcount,
        rollout_mae=rollout_mae,
        rollout_err=rollout_err,
        sdf_roughness=sdf_roughness,
        lth_mae=lth_mae,
        lt_mae=lt_mae,
        eikonal=eik,
    )

# ── run ──────────────────────────────────────────────────────────────────────

cases = sorted(p.name for p in RUN_A.iterdir() if p.is_dir())

results = {}
for case in cases:
    print(f"  {case}...", flush=True)
    pred_a, gt_a, sp_a = load(RUN_A / case)
    pred_b, gt_b, sp_b = load(RUN_B / case)
    results[case] = (compute_metrics(pred_a, gt_a, sp_a),
                     compute_metrics(pred_b, gt_b, sp_b),
                     gt_a)   # store gt for reference

METRIC_KEYS = ["vf_mae", "bubble_count", "rollout_mae", "rollout_err", "sdf_roughness", "lth_mae", "lt_mae", "eikonal"]
METRIC_LABELS = [
    "vapor fraction MAE",
    "mean bubble count",
    "rollout MAE (all frames)",
    "rollout error @ frame 199",
    "SDF roughness @ frame 199",
    "liquid temp at heater MAE",
    "mean liquid temp MAE",
    "eikonal",
]

# ── print table ──────────────────────────────────────────────────────────────
print()
print(f"  run A : effortless-lion-61   (nucleus2 geom medium)")
print(f"  run B : earthy-microwave-68  (nucleus2 fm concat medium)")
print()

C = 14
header = f"{'metric':<28}" + "".join(f"{'A':>{C}}{'B':>{C}}{'Δ(B-A)':>{C}}" for _ in METRIC_KEYS[:1])
# Print one metric-row per case, then summary

# Per-case table
for case, (ma, mb, gt) in results.items():
    print(f"── {case} ──")
    print(f"  {'metric':<30}{'A':>{C}}{'B':>{C}}{'Δ(B-A)':>{C}}")
    print(f"  {'-'*60}")
    for key, label in zip(METRIC_KEYS, METRIC_LABELS):
        va, vb = ma[key], mb[key]
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<30}{va:>{C}.4f}{vb:>{C}.4f}{sign+f'{delta:.4f}':>{C}}")
    print()

# Summary: mean across cases
print("═" * (30 + 3 * C + 2))
print(f"  MEAN ACROSS {len(cases)} CASES")
print(f"  {'metric':<30}{'A':>{C}}{'B':>{C}}{'Δ(B-A)':>{C}}")
print(f"  {'-'*60}")
for key, label in zip(METRIC_KEYS, METRIC_LABELS):
    vals_a = [results[c][0][key] for c in cases]
    vals_b = [results[c][1][key] for c in cases]
    va, vb = np.mean(vals_a), np.mean(vals_b)
    delta = vb - va
    sign = "+" if delta >= 0 else ""
    print(f"  {label:<30}{va:>{C}.4f}{vb:>{C}.4f}{sign+f'{delta:.4f}':>{C}}")
print()
