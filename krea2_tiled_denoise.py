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
from krea2_tile_vision_substitution import (
    substitute_tile_conditioning_for_positive_rows,
)


def build_tile_fusion_weight(tile_height: int, tile_width: int, latent_rank: int,
                             device, dtype) -> torch.Tensor:
    """The separable raised cosine weight for one tile, shaped to broadcast.

    The runtime multiplies a per-column taper by a per-row taper
    (stable-diffusion.cpp:3140), which is an outer product. Leading singleton
    dimensions are added to match the latent's rank, so the same weight covers a
    four dimensional image latent and the five dimensional one Krea2 uses.
    """
    row_taper = torch.tensor(
        [raised_cosine_taper(row, tile_height) for row in range(tile_height)],
        device=device, dtype=dtype)
    column_taper = torch.tensor(
        [raised_cosine_taper(column, tile_width) for column in range(tile_width)],
        device=device, dtype=dtype)
    leading_singletons = (1,) * max(0, latent_rank - 2)
    return torch.outer(row_taper, column_taper).reshape(
        *leading_singletons, tile_height, tile_width)


def denoise_latent_as_fused_tiles(apply_model, latent: torch.Tensor,
                                  timestep, conditioning: dict,
                                  plan: LatentTilePlan,
                                  rope_offset_holder: TileRopeOffsetHolder | None,
                                  cond_or_uncond: list[int] | None = None,
                                  conditioning_per_tile: list | None = None
                                  ) -> torch.Tensor:
    """One denoise step, run per tile and fused under the raised cosine.

    Without `conditioning_per_tile` the same conditioning goes to every tile, which
    describes the whole canvas rather than the tile. With it, each tile receives a
    conditioning built from its own region, so a tile is told to draw the part of
    the picture it actually covers.

    ComfyUI composites the init latent and the denoise mask around apply_model
    rather than inside it, so there is nothing else here to crop per tile.
    """
    if len(plan.tiles) <= 1:
        return apply_model(latent, timestep, **conditioning)

    # Accumulate in float32 at least, matching the runtime's use of double for the
    # sums, so a bf16 model's tiles do not lose the overlap to rounding.
    accumulation_dtype = torch.float32 if latent.dtype != torch.float64 else torch.float64
    accumulated = torch.zeros(latent.shape, device=latent.device, dtype=accumulation_dtype)
    # Only the last two dimensions are tiled. Krea2's latents carry a frame axis as
    # well, so everything below indexes from the RIGHT rather than at fixed
    # positions, and the weight is shaped to broadcast over whatever leads.
    accumulated_weight = torch.zeros(
        (1,) * (latent.ndim - 2) + tuple(latent.shape[-2:]),
        device=latent.device, dtype=accumulation_dtype)

    for tile_index, tile in enumerate(plan.tiles):
        rows = slice(tile.row_start, tile.row_start + tile.height)
        columns = slice(tile.column_start, tile.column_start + tile.width)

        conditioning_for_this_tile = conditioning
        if conditioning_per_tile is not None and cond_or_uncond is not None:
            conditioning_for_this_tile = substitute_tile_conditioning_for_positive_rows(
                conditioning, cond_or_uncond, conditioning_per_tile[tile_index])

        if rope_offset_holder is not None:
            rope_offset_holder.set_to_tile_origin(tile.row_start, tile.column_start)
        try:
            tile_prediction = apply_model(latent[..., rows, columns],
                                          timestep, **conditioning_for_this_tile)
        finally:
            if rope_offset_holder is not None:
                rope_offset_holder.clear()

        weight = build_tile_fusion_weight(tile.height, tile.width, latent.ndim,
                                          latent.device, accumulation_dtype)
        accumulated[..., rows, columns] += tile_prediction.to(accumulation_dtype) * weight
        accumulated_weight[..., rows, columns] += weight

    fused = accumulated / accumulated_weight.clamp_min(torch.finfo(accumulation_dtype).tiny)
    return fused.to(latent.dtype)


def build_tiled_denoise_wrapper(plan: LatentTilePlan,
                                rope_offset_holder: TileRopeOffsetHolder | None,
                                conditioning_per_tile: list | None = None):
    """The callable ComfyUI installs with set_model_unet_function_wrapper."""
    def tiled_denoise_wrapper(apply_model, arguments: dict) -> torch.Tensor:
        return denoise_latent_as_fused_tiles(
            apply_model,
            arguments["input"],
            arguments["timestep"],
            arguments["c"],
            plan,
            rope_offset_holder,
            arguments.get("cond_or_uncond"),
            conditioning_per_tile)

    return tiled_denoise_wrapper
