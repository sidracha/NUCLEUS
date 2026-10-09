"""
Patch BubbleML_2 JSON sim-params files with fields that are expected by the
nucleus training code but missing from the public dataset.

Known issue: BubbleML_2 ships without `sat_temp` (saturation temperature °C),
which both Nucleus1 and Nucleus2 list in `expected_fluid_params`. We fill it
in from the `liquid` field using a hard-coded physical lookup.

Any file that still has missing fields after patching is reported — add
handling here or in the dataset JSON upstream.

Default scope matches the expectations of
`src/nucleus/models/nucleus2_moe.py:137-142` (which the FM model inherits).

Usage:
    python scripts/patch_sim_params.py --root ~/BubbleML_2_downsample64
    python scripts/patch_sim_params.py --root ~/BubbleML_2_downsample64 --dry-run
"""

import argparse
import json
import shutil
from pathlib import Path

# Saturation temperature (°C at 1 atm) for each BubbleML fluid.
SAT_TEMP_BY_LIQUID = {
    "op250": 58.0,
    "fc72":  56.0,
    "r515b": 37.0,   # unverified from nucleus configs; standard refrigerant table
    "ln2":  -196.0,
}

# Fields the training code looks up per sample. Must match
# src/nucleus/models/nucleus2_moe.py:137-142 (FM inherits from Nucleus2MoE).
EXPECTED_FLUID  = {"inv_reynolds", "cpgas", "mugas", "rhogas", "thcogas",
                   "stefan", "prandtl", "bulk_temp", "sat_temp"}
EXPECTED_HEATER = {"wallTemp", "xMin", "xMax"}
EXPECTED_GLOBAL = {"gravy"}


def is_sim_params_json(d: dict) -> bool:
    """Heuristic: a BubbleML sim-params JSON has `liquid` and `heater`."""
    return isinstance(d, dict) and "liquid" in d and "heater" in d


def patch_file(path: Path, dry_run: bool, backup: bool) -> tuple[bool, list[str]]:
    """
    Returns (modified, still_missing). `still_missing` lists any fields the
    patcher could not fill (should be empty on success).
    """
    with open(path) as f:
        d = json.load(f)

    if not is_sim_params_json(d):
        return False, []

    changed = False
    if "sat_temp" not in d:
        liquid = str(d.get("liquid", "")).lower()
        if liquid in SAT_TEMP_BY_LIQUID:
            d["sat_temp"] = SAT_TEMP_BY_LIQUID[liquid]
            changed = True

    heater = d.get("heater", {})
    missing = sorted(
        [f"fluid.{k}"  for k in EXPECTED_FLUID  - d.keys()]
      + [f"heater.{k}" for k in EXPECTED_HEATER - heater.keys()]
      + [f"global.{k}" for k in EXPECTED_GLOBAL - d.keys()]
    )

    if changed and not dry_run:
        if backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    return changed, missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="directory to scan recursively for sim-params JSONs")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would change without writing")
    p.add_argument("--backup", action="store_true",
                   help="write a .bak copy of each file before patching")
    args = p.parse_args()

    root = args.root.expanduser().resolve()
    jsons = sorted(root.rglob("*.json"))

    print(f"scanning {len(jsons)} JSON files under {root}")
    print(f"mode: {'DRY-RUN' if args.dry_run else 'WRITE'}  backup: {args.backup}\n")

    n_patched = n_ok = n_skip = 0
    unresolved = []
    for path in jsons:
        try:
            modified, missing = patch_file(path, args.dry_run, args.backup)
        except json.JSONDecodeError:
            print(f"  skip (not JSON):   {path.relative_to(root)}")
            n_skip += 1
            continue

        rel = path.relative_to(root)
        if missing:
            print(f"  UNRESOLVED {rel}: still missing {missing}")
            unresolved.append((rel, missing))
        elif modified:
            print(f"  patched    {rel}")
            n_patched += 1
        else:
            n_ok += 1

    print(f"\ndone — patched {n_patched}, already-ok {n_ok}, "
          f"skipped {n_skip}, unresolved {len(unresolved)}")
    if unresolved:
        print("\nNOTE: files with unresolved fields won't train. Either:")
        print(" - add the fluid to SAT_TEMP_BY_LIQUID at the top of this script, or")
        print(" - hand-patch the file to include the listed fields.")


if __name__ == "__main__":
    main()
