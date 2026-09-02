#!/usr/bin/env python3
"""Checks for where the sampler's tile geometry comes from.

The tiles in a bundle were cropped and encoded against the ENCODER's grid and
overlap, so those are the numbers the sampler must visit. Reading them from the
bundle removes a pair of widgets that otherwise had to be kept in step by hand,
and with them a run that dies seconds in over a settings mismatch.

Run with ComfyUI's interpreter:
    ComfyUI/.venv/bin/python tests/test_krea2_sampler_tile_geometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/media/aikenyon/NVME_2/ubuntu_comfy/ComfyUI")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from krea2_tile_planning import plan_latent_tiles  # noqa: E402
from krea2_tiled_diffusion_node import (  # noqa: E402
    conditioning_per_tile_matching_plan,
    tile_geometry_to_sample_with,
)

failures: list[str] = []


def check(description: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}: {description}{(' ' + detail) if detail else ''}")
    if not passed:
        failures.append(description)


def bundle(tile_grid: str, tile_overlap: int, tile_count: int) -> dict:
    return {"tile_grid": tile_grid, "tile_overlap": tile_overlap,
            "conditioning_per_tile": [f"tile {index}" for index in range(tile_count)]}


def main() -> int:
    check("with no bundle the node's own geometry is used",
          tile_geometry_to_sample_with(None, "3x3", 512) == ("3x3", 512),
          f"got {tile_geometry_to_sample_with(None, '3x3', 512)}")

    check("a bundle's geometry wins over the node's widgets",
          tile_geometry_to_sample_with(bundle("2x2", 128, 4), "2x2", 32)
          == ("2x2", 128),
          f"got {tile_geometry_to_sample_with(bundle('2x2', 128, 4), '2x2', 32)}")

    check("a bundle's GRID wins too, not just the overlap",
          tile_geometry_to_sample_with(bundle("3x3", 512, 9), "2x2", 256)
          == ("3x3", 512),
          f"got {tile_geometry_to_sample_with(bundle('3x3', 512, 9), '2x2', 256)}")

    check("matching values pass through unchanged",
          tile_geometry_to_sample_with(bundle("2x2", 256, 4), "2x2", 256)
          == ("2x2", 256))

    check("a bundle missing its geometry falls back to the node's",
          tile_geometry_to_sample_with({"conditioning_per_tile": []}, "2x3", 64)
          == ("2x3", 64),
          f"got {tile_geometry_to_sample_with({'conditioning_per_tile': []}, '2x3', 64)}")

    # Once the geometry comes from the bundle, a count that still disagrees means
    # the encoder measured a different image than this latent came from.
    plan = plan_latent_tiles(latent_width=128, latent_height=128,
                             tile_grid="2x2", tile_overlap_pixels=256)
    check("a bundle holding one conditioning per tile is accepted",
          conditioning_per_tile_matching_plan(bundle("2x2", 256, 4), plan)
          == ["tile 0", "tile 1", "tile 2", "tile 3"])
    check("no bundle yields no per-tile conditioning",
          conditioning_per_tile_matching_plan(None, plan) is None)
    check("a bundle whose tile count disagrees with the plan is refused",
          refuses_a_bundle_of_the_wrong_size(plan))

    if failures:
        print(f"\n{len(failures)} failing checks:")
        for failed in failures:
            print(f"  - {failed}")
        return 1
    print("\nall krea2 sampler tile geometry checks passed")
    return 0


def refuses_a_bundle_of_the_wrong_size(plan) -> bool:
    try:
        conditioning_per_tile_matching_plan(bundle("2x2", 256, 9), plan)
    except ValueError as refusal:
        return "9 conditionings" in str(refusal) and "4 tiles" in str(refusal)
    return False


if __name__ == "__main__":
    raise SystemExit(main())
