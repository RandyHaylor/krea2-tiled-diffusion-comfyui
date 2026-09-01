"""Denoising one step as overlapping tiles, fused before the step returns.

This is the mechanism the whole node exists for, ported from
stable-diffusion.cpp:3020-3172. Every tile is denoised by the real model for the
SAME step and the results are recombined immediately, so neighbours cannot drift
apart. Anything that finishes a tile before blending is a different algorithm.

ComfyUI calls the wrapper as
    wrapper(apply_model, {"input", "timestep", "c", "cond_or_uncond"})
and expects a tensor shaped like `input` (comfy/samplers.py:333). It is called per
cond batch, so `input` may hold cond and uncond stacked; the tiling is purely
spatial and works on the whole batch at once.
"""
from __future__ import annotations

import torch

from krea2_rope_tile_offset import TileRopeOffsetHolder
from krea2_tile_planning import LatentTilePlan, raised_cosine_taper


def build_tile_fusion_weight(tile_height: int, tile_width: int,
                             device, dtype) -> torch.Tensor:
    """The separable raised cosine weight for one tile, shaped (1, 1, h, w).

    The runtime multiplies a per-column taper by a per-row taper
    (stable-diffusion.cpp:3140), which is an outer product.
    """
    row_taper = torch.tensor(
        [raised_cosine_taper(row, tile_height) for row in range(tile_height)],
        device=device, dtype=dtype)
    column_taper = torch.tensor(
        [raised_cosine_taper(column, tile_width) for column in range(tile_width)],
        device=device, dtype=dtype)
    return torch.outer(row_taper, column_taper).reshape(1, 1, tile_height, tile_width)


def denoise_latent_as_fused_tiles(apply_model, latent: torch.Tensor,
                                  timestep, conditioning: dict,
                                  plan: LatentTilePlan,
                                  rope_offset_holder: TileRopeOffsetHolder | None
                                  ) -> torch.Tensor:
    """One denoise step, run per tile and fused under the raised cosine.

    `conditioning` is passed to every tile unchanged. Unlike the runtime, there is
    nothing here to crop per tile: ComfyUI composites the init latent and the
    denoise mask around apply_model rather than inside it, and Krea2's conditioning
    is text embeddings that are not tied to a region.
    """
    if len(plan.tiles) <= 1:
        return apply_model(latent, timestep, **conditioning)

    # Accumulate in float32 at least, matching the runtime's use of double for the
    # sums, so a bf16 model's tiles do not lose the overlap to rounding.
    accumulation_dtype = torch.float32 if latent.dtype != torch.float64 else torch.float64
    accumulated = torch.zeros(latent.shape, device=latent.device, dtype=accumulation_dtype)
    accumulated_weight = torch.zeros((1, 1, latent.shape[-2], latent.shape[-1]),
                                     device=latent.device, dtype=accumulation_dtype)

    for tile in plan.tiles:
        rows = slice(tile.row_start, tile.row_start + tile.height)
        columns = slice(tile.column_start, tile.column_start + tile.width)

        if rope_offset_holder is not None:
            rope_offset_holder.set_to_tile_origin(tile.row_start, tile.column_start)
        try:
            tile_prediction = apply_model(latent[:, :, rows, columns],
                                          timestep, **conditioning)
        finally:
            if rope_offset_holder is not None:
                rope_offset_holder.clear()

        weight = build_tile_fusion_weight(tile.height, tile.width,
                                          latent.device, accumulation_dtype)
        accumulated[:, :, rows, columns] += tile_prediction.to(accumulation_dtype) * weight
        accumulated_weight[:, :, rows, columns] += weight

    fused = accumulated / accumulated_weight.clamp_min(torch.finfo(accumulation_dtype).tiny)
    return fused.to(latent.dtype)


def build_tiled_denoise_wrapper(plan: LatentTilePlan,
                                rope_offset_holder: TileRopeOffsetHolder | None):
    """The callable ComfyUI installs with set_model_unet_function_wrapper."""
    def tiled_denoise_wrapper(apply_model, arguments: dict) -> torch.Tensor:
        return denoise_latent_as_fused_tiles(
            apply_model,
            arguments["input"],
            arguments["timestep"],
            arguments["c"],
            plan,
            rope_offset_holder)

    return tiled_denoise_wrapper
