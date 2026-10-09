"""
Bicubic-downsample every HDF5 field in a BubbleML_2-style directory tree and
mirror the layout under a new root. Non-HDF5 files (e.g., JSON sim-params) are
copied verbatim. Coordinate arrays (x_centers/x_faces/y_centers/y_faces) are
regenerated for the new grid using the physical bounds from the companion JSON.

Example:
    python scripts/downsample_dataset.py \
        --src ~/BubbleML_2 \
        --dst ~/BubbleML_2_downsample64 \
        --size 64
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


FIELD_RANK = 3  # (T, H, W) — datasets with this rank get bicubic-resized
# Only these fields are written to the output file (plus regenerated coords).
# Other 3D fields in the source (massflux, normx, normy, pressure) are skipped.
DOWNSAMPLE_FIELDS = {"dfun", "temperature", "velx", "vely"}


def bicubic_resize(x_np: np.ndarray, out_h: int, out_w: int, device: str) -> np.ndarray:
    x = torch.from_numpy(x_np).unsqueeze(1).to(device)  # (T, 1, H, W)
    y = F.interpolate(x, size=(out_h, out_w), mode="bicubic",
                      align_corners=False, antialias=True)
    return y.squeeze(1).cpu().numpy().astype(np.float32)


def regen_coords(json_path: Path, size: int):
    with open(json_path) as f:
        meta = json.load(f)
    x_min, x_max = float(meta["x_min"]), float(meta["x_max"])
    y_min, y_max = float(meta["y_min"]), float(meta["y_max"])

    x_faces = np.linspace(x_min, x_max, size + 1, dtype=np.float32)
    y_faces = np.linspace(y_min, y_max, size + 1, dtype=np.float32)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    return {
        "x_faces": x_faces,
        "y_faces": y_faces,
        "x_centers": x_centers,
        "y_centers": y_centers,
    }


def downsample_hdf5(src: Path, dst: Path, size: int, device: str, chunk_t: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    json_src = src.with_suffix(".json")
    coords = regen_coords(json_src, size) if json_src.exists() else None

    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        for name, item in fin.attrs.items():
            fout.attrs[name] = item

        for key in fin.keys():
            d = fin[key]

            if coords is not None and key in coords:
                fout.create_dataset(key, data=coords[key])
                continue

            if d.ndim == FIELD_RANK:
                if key not in DOWNSAMPLE_FIELDS:
                    continue
                T = d.shape[0]
                out = fout.create_dataset(
                    key, shape=(T, size, size), dtype=np.float32,
                    chunks=(min(chunk_t, T), size, size), compression=None,
                )
                for i in range(0, T, chunk_t):
                    j = min(i + chunk_t, T)
                    arr = d[i:j].astype(np.float32, copy=False)
                    out[i:j] = bicubic_resize(arr, size, size, device)
            else:
                fout.create_dataset(key, data=d[...])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--chunk-t", type=int, default=256,
                   help="time-steps per bicubic call; lower if OOM")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    src_root = args.src.expanduser().resolve()
    dst_root = args.dst.expanduser().resolve()

    hdf5s = sorted(src_root.rglob("*.hdf5"))
    others = [p for p in src_root.rglob("*") if p.is_file() and p.suffix != ".hdf5"]

    print(f"src: {src_root}")
    print(f"dst: {dst_root}")
    print(f"size: {args.size}x{args.size}  device: {args.device}  chunk_t: {args.chunk_t}")
    print(f"hdf5 files: {len(hdf5s)}  other files: {len(others)}")

    for src in tqdm(hdf5s, desc="hdf5"):
        rel = src.relative_to(src_root)
        downsample_hdf5(src, dst_root / rel, args.size, args.device, args.chunk_t)

    for src in tqdm(others, desc="copy"):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
