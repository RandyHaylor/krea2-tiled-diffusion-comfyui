#!/usr/bin/env python3
"""Checks for tile geometry and the fusion taper.

Expected values come from the working implementation: pixel-space sizing from
scripts/tiled_diffusion.py in the krea runtime repo, start positions and the taper
from stable-diffusion.cpp:3044-3066, and one case from a real GPU log line.

Run with plain `python3 tests/test_krea2_tile_planning.py`, no pytest.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from krea2_tile_planning import (  # noqa: E402
    LATENT_SCALE,
    TILE_GRIDS,
    actual_overlap_between_tiles,
    parse_tile_grid,
    plan_latent_tiles,
    raised_cosine_taper,
    tile_size_covering_length,
    tile_start_positions_covering_length,
)

failures: list[str] = []


def check(description: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}: {description}{(' ' + detail) if detail else ''}")
    if not passed:
        failures.append(description)


def main() -> int:
    check("the latent scale matches the VAE's factor of eight", LATENT_SCALE == 8)

    check("a grid parses to columns then rows", parse_tile_grid("3x2") == (3, 2))
    check("a malformed grid falls back to a single tile",
          parse_tile_grid("nonsense") == (1, 1))
    check("a grid below one is clamped up", parse_tile_grid("0x0") == (1, 1))

    # scripts/tiled_diffusion.py's own suite asserts 688x976 here.
    check("tile size matches the working app at 2x2 and 128px overlap",
          (tile_size_covering_length(1248, 2, 128),
           tile_size_covering_length(1824, 2, 128)) == (688, 976),
          f"got {tile_size_covering_length(1248, 2, 128)}x"
          f"{tile_size_covering_length(1824, 2, 128)}")

    check("tile sizes round UP to a whole latent cell",
          tile_size_covering_length(1000, 3, 100) % LATENT_SCALE == 0,
          f"got {tile_size_covering_length(1000, 3, 100)}")
    check("a requested overlap larger than the axis allows is clamped",
          tile_size_covering_length(1824, 3, 4096)
          == tile_size_covering_length(1824, 3, 1824 // 3),
          "the clamp is length // count")
    check("one tile covers the whole axis",
          tile_size_covering_length(1024, 1, 128) == 1024)

    check("the resulting overlap is reported, not the requested one",
          actual_overlap_between_tiles(1248, 696, 3) == 420,
          f"got {actual_overlap_between_tiles(1248, 696, 3)}")
    check("a single tile shares nothing",
          actual_overlap_between_tiles(1248, 1248, 1) == 0)

    # stable-diffusion.cpp:3044-3057
    starts = tile_start_positions_covering_length(156, 87, 52)
    check("start positions begin at zero and end flush with the canvas",
          starts[0] == 0 and starts[-1] == 156 - 87, f"got {starts}")
    check("start positions are spread, so gaps differ by at most one cell",
          max(starts[i + 1] - starts[i] for i in range(len(starts) - 1))
          - min(starts[i + 1] - starts[i] for i in range(len(starts) - 1)) <= 1,
          f"got {starts}")
    check("a tile at least as large as the axis needs one position",
          tile_start_positions_covering_length(100, 100, 16) == [0])
    check("an exact doubling needs three tiles, not two",
          len(tile_start_positions_covering_length(256, 128, 16)) == 3,
          f"got {tile_start_positions_covering_length(256, 128, 16)}")

    # stable-diffusion.cpp:3063-3066
    check("the taper is a raised cosine with the runtime's floor",
          all(abs(raised_cosine_taper(position, 8)
                  - (0.0001 + 0.5 * (1 - math.cos(2 * math.pi * (position + 0.5) / 8)))) < 1e-12
              for position in range(8)))
    check("the taper never reaches zero, so an edge-only pixel can still divide",
          min(raised_cosine_taper(position, 16) for position in range(16)) > 0.0,
          f"min {min(raised_cosine_taper(p, 16) for p in range(16))}")
    check("the taper peaks in the middle of the tile",
          raised_cosine_taper(7, 16) > raised_cosine_taper(0, 16)
          and raised_cosine_taper(8, 16) > raised_cosine_taper(15, 16))

    # GROUND TRUTH from a real run. The server log for a 1248x1824 hires pass at a
    # 3x3 grid with 512px overlap reported:
    #   "tiled diffusion: latent tile 87x119, overlap 52"
    #   "tiled diffusion: 9 tiles per step (3 x 3) over a 156x228 latent"
    plan = plan_latent_tiles(latent_width=156, latent_height=228,
                             tile_grid="3x3", tile_overlap_pixels=512)
    check("the plan reproduces the latent tile size from the real run",
          (plan.tile_width, plan.tile_height) == (87, 119),
          f"got {plan.tile_width}x{plan.tile_height}")
    check("the plan reproduces the overlap from the real run",
          plan.overlap == 52, f"got {plan.overlap}")
    check("the plan reproduces the tile count and grid from the real run",
          (len(plan.tiles), plan.columns, plan.rows) == (9, 3, 3),
          f"got {len(plan.tiles)} tiles, {plan.columns} x {plan.rows}")

    check("tiles are planned rows outer, columns inner",
          [tile.row_start for tile in plan.tiles]
          == sorted(tile.row_start for tile in plan.tiles),
          f"got {[(t.column_start, t.row_start) for t in plan.tiles]}")
    check("every tile lies inside the canvas",
          all(tile.column_start + tile.width <= 156
              and tile.row_start + tile.height <= 228 for tile in plan.tiles))
    check("the tiles together cover every column and row of the canvas",
          covers_every_position(plan.tiles, 156, 228))

    # The 2x2 default at 256px overlap, the recipe the node ships with.
    default_plan = plan_latent_tiles(latent_width=156, latent_height=228,
                                     tile_grid="2x2", tile_overlap_pixels=256)
    check("the default 2x2 recipe plans four tiles",
          len(default_plan.tiles) == 4,
          f"got {len(default_plan.tiles)}, {default_plan.columns} x {default_plan.rows}")
    check("the default recipe covers the canvas",
          covers_every_position(default_plan.tiles, 156, 228))

    single = plan_latent_tiles(latent_width=64, latent_height=64,
                               tile_grid="1x1", tile_overlap_pixels=128)
    check("a 1x1 grid plans a single tile covering everything",
          len(single.tiles) == 1
          and single.tiles[0].width == 64 and single.tiles[0].height == 64,
          f"got {single.tiles}")

    # The grid is a promise: a 2x1 must be two tiles whatever overlap is asked
    # for. An axis holding a single tile has no overlap to report, which must not
    # be read as "the requested overlap was unachievable" for the OTHER axis.
    check("the plan always has exactly the requested columns and rows",
          *plan_always_matches_the_requested_grid())

    if failures:
        print(f"\n{len(failures)} failing checks:")
        for failed in failures:
            print(f"  - {failed}")
        return 1
    print("\nall krea2 tile planning checks passed")
    return 0


def plan_always_matches_the_requested_grid() -> tuple[bool, str]:
    """Every supported grid, over awkward canvases and the whole overlap range."""
    canvas_latent_sizes = [(128, 128), (104, 152), (156, 228), (97, 131), (64, 200),
                           (32, 32), (16, 300), (255, 255), (40, 41)]
    overlaps_pixels = [8, 32, 64, 128, 256, 512, 1024, 2048]

    for latent_width, latent_height in canvas_latent_sizes:
        for tile_grid in TILE_GRIDS:
            for overlap_pixels in overlaps_pixels:
                requested_columns, requested_rows = parse_tile_grid(tile_grid)
                plan = plan_latent_tiles(latent_width=latent_width,
                                         latent_height=latent_height,
                                         tile_grid=tile_grid,
                                         tile_overlap_pixels=overlap_pixels)
                if (plan.columns, plan.rows) != (requested_columns, requested_rows):
                    return False, (f"{latent_width}x{latent_height} latent, grid "
                                   f"{tile_grid}, overlap {overlap_pixels} planned "
                                   f"{plan.columns}x{plan.rows} "
                                   f"({len(plan.tiles)} tiles)")
    return True, ""


def covers_every_position(tiles, width: int, height: int) -> bool:
    covered_columns = set()
    covered_rows = set()
    for tile in tiles:
        covered_columns.update(range(tile.column_start, tile.column_start + tile.width))
        covered_rows.update(range(tile.row_start, tile.row_start + tile.height))
    return covered_columns == set(range(width)) and covered_rows == set(range(height))


if __name__ == "__main__":
    raise SystemExit(main())
