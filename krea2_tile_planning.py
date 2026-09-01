"""Tile geometry and the fusion taper, ported from the working implementation.

Two halves of one contract are reproduced here.

The app half decides the geometry in PIXELS and hands the runtime three numbers in
latent units: tile width, tile height and a single overlap. See
`tiled_diffusion_sample_args` in scripts/tiled_diffusion.py of the krea runtime
repo. Sizes round UP to a whole latent cell, and the overlap sent is the one that
ACTUALLY results from that rounding, not the one requested.

The runtime half turns those into start positions and a fusion weight. See
stable-diffusion.cpp:3044-3066.

Both halves matter. Deriving the starts from the requested overlap instead of the
resulting one changes the tile count: a 3x3 grid stops being nine tiles.

Pure Python and math only, so it runs without torch or ComfyUI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

LATENT_SCALE = 8

# stable-diffusion.cpp:3065. Without this floor the outermost row and column of a
# tile weigh exactly zero, and a canvas pixel covered only by tile edges would
# have nothing to divide by.
TAPER_FLOOR = 0.0001


@dataclass(frozen=True)
class LatentTile:
    """One tile's place on the latent canvas, in latent cells."""
    column_start: int
    row_start: int
    width: int
    height: int


@dataclass(frozen=True)
class LatentTilePlan:
    """Everything the denoise loop needs, and what the geometry actually became."""
    tiles: list[LatentTile]
    tile_width: int
    tile_height: int
    overlap: int
    columns: int
    rows: int


def parse_tile_grid(tile_grid: str) -> tuple[int, int]:
    """The requested grid as (columns, rows), never below one."""
    columns, _, rows = str(tile_grid).partition("x")
    try:
        return max(1, int(columns)), max(1, int(rows))
    except ValueError:
        return 1, 1


def tile_size_covering_length(length: int, tile_count: int, overlap: int) -> int:
    """The tile edge covering one axis with this many tiles and this overlap.

    In PIXELS, rounded up to a whole latent cell, because covering the canvas
    matters more than landing on the requested overlap exactly.
    """
    length, tile_count = max(1, int(length)), max(1, int(tile_count))
    if tile_count == 1:
        return length
    overlap = max(0, min(int(overlap), length // tile_count))
    exact = (length + overlap * (tile_count - 1)) / tile_count
    rounded = int(-(-exact // LATENT_SCALE)) * LATENT_SCALE
    return min(length, max(LATENT_SCALE, rounded))


def actual_overlap_between_tiles(length: int, tile_size: int, tile_count: int) -> int:
    """How far neighbours really share once the tiles are spread across the axis.

    The requested overlap is a target: rounding the tile to a whole latent cell
    moves it, so this reports what the canvas ends up with.
    """
    if int(tile_count) < 2:
        return 0
    stride = (int(length) - int(tile_size)) / (int(tile_count) - 1)
    return max(0, int(round(int(tile_size) - stride)))


def tile_start_positions_covering_length(length: int, tile_size: int,
                                         overlap: int) -> list[int]:
    """Where each tile begins along one axis, mirroring stable-diffusion.cpp:3044.

    The count comes from the stride, but the positions are spread evenly across
    the length rather than marching at that stride, so the last tile is flush with
    the edge instead of short. Integer division means neighbouring gaps can differ
    by one cell.
    """
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    count = (length - tile_size + stride - 1) // stride + 1
    if count == 1:
        return [0]
    return [(index * (length - tile_size)) // (count - 1) for index in range(count)]


def raised_cosine_taper(position: int, tile_length: int) -> float:
    """One axis of the fusion weight, from stable-diffusion.cpp:3063-3066.

    A full Hann window across the tile, so a tile contributes least at its edges
    and neighbours sum smoothly. The pixel weight is the product along each axis.
    """
    centred = (position + 0.5) / tile_length
    return TAPER_FLOOR + 0.5 * (1.0 - math.cos(2.0 * math.pi * centred))


def plan_latent_tiles(latent_width: int, latent_height: int, tile_grid: str,
                      tile_overlap_pixels: int) -> LatentTilePlan:
    """Every tile the denoise loop will visit, rows outer and columns inner.

    Reproduces the app's pixel-space sizing and the runtime's latent-space
    spreading, in that order, so the plan matches what the working pipeline runs.
    Tiles are clipped at the canvas edge as the runtime does at
    stable-diffusion.cpp:3104.
    """
    columns, rows = parse_tile_grid(tile_grid)
    canvas_width_pixels = int(latent_width) * LATENT_SCALE
    canvas_height_pixels = int(latent_height) * LATENT_SCALE

    tile_width_pixels = tile_size_covering_length(
        canvas_width_pixels, columns, tile_overlap_pixels)
    tile_height_pixels = tile_size_covering_length(
        canvas_height_pixels, rows, tile_overlap_pixels)

    overlap_pixels = min(
        actual_overlap_between_tiles(canvas_width_pixels, tile_width_pixels, columns),
        actual_overlap_between_tiles(canvas_height_pixels, tile_height_pixels, rows))
    if overlap_pixels <= 0:
        overlap_pixels = int(tile_overlap_pixels)

    tile_width = tile_width_pixels // LATENT_SCALE
    tile_height = tile_height_pixels // LATENT_SCALE
    overlap = max(1, overlap_pixels // LATENT_SCALE)

    column_starts = tile_start_positions_covering_length(latent_width, tile_width, overlap)
    row_starts = tile_start_positions_covering_length(latent_height, tile_height, overlap)

    return LatentTilePlan(
        tiles=[LatentTile(column_start=column_start,
                          row_start=row_start,
                          width=min(tile_width, latent_width - column_start),
                          height=min(tile_height, latent_height - row_start))
               for row_start in row_starts
               for column_start in column_starts],
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        columns=len(column_starts),
        rows=len(row_starts))
