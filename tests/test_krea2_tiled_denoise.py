#!/usr/bin/env python3
"""Checks for the per-step tile fusion and the per-tile RoPE offset.

apply_model is stubbed, so these run without a model or a GPU, but the tensor
arithmetic is real. Run with ComfyUI's interpreter:
    ComfyUI/.venv/bin/python tests/test_krea2_tiled_denoise.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from krea2_rope_tile_offset import (  # noqa: E402
    KREA2_PATCH_SIZE,
    TileRopeOffsetHolder,
    build_post_input_rope_offset_patch,
)
from krea2_tile_planning import plan_latent_tiles  # noqa: E402
from krea2_tiled_denoise import (  # noqa: E402
    build_tile_fusion_weight,
    denoise_latent_as_fused_tiles,
)

failures: list[str] = []


def record_the_conditioning_each_tile_receives(plan, conditioning_per_tile):
    """Run the fused loop with a stub model, returning what each tile was given."""
    received = []

    def record_and_return_zeros(tile, timestep, **conditioning):
        received.append(conditioning["c_crossattn"].clone())
        return torch.zeros_like(tile)

    latent = torch.zeros(1, 4, plan.tiles[0].height + plan.tiles[-1].row_start,
                         plan.tiles[0].width + plan.tiles[-1].column_start)
    denoise_latent_as_fused_tiles(
        record_and_return_zeros, latent, timestep=None,
        conditioning={"c_crossattn": torch.zeros(1, 6, 8)}, plan=plan,
        rope_offset_holder=None, cond_or_uncond=[0],
        conditioning_per_tile=conditioning_per_tile)
    return received


def every_tile_receives_its_own_conditioning(plan) -> bool:
    conditioning_per_tile = [[[torch.full((1, 6, 8), float(index)), {}]]
                             for index in range(len(plan.tiles))]
    received = record_the_conditioning_each_tile_receives(plan, conditioning_per_tile)
    if len(received) != len(plan.tiles):
        return False
    return all(float(embedding.max()) == float(index)
               for index, embedding in enumerate(received))


def every_tile_receives_the_same_conditioning(plan) -> bool:
    received = record_the_conditioning_each_tile_receives(plan, None)
    return all(torch.equal(embedding, received[0]) for embedding in received)


def check(description: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}: {description}{(' ' + detail) if detail else ''}")
    if not passed:
        failures.append(description)


def main() -> int:
    plan = plan_latent_tiles(latent_width=156, latent_height=228,
                             tile_grid="3x3", tile_overlap_pixels=512)
    latent = torch.randn(2, 16, 228, 156)

    # A model that returns its input unchanged. Whatever the weights are, a
    # correctly normalised fusion of identical predictions must rebuild the input.
    # This is what proves the raised cosine and the weight division agree.
    returns_input_unchanged = lambda tile, timestep, **conditioning: tile
    fused = denoise_latent_as_fused_tiles(
        returns_input_unchanged, latent, timestep=None, conditioning={},
        plan=plan, rope_offset_holder=None)
    check("fusing an identity model reconstructs the latent exactly",
          torch.allclose(fused, latent, atol=1e-5),
          f"max deviation {float((fused - latent).abs().max()):.3e}")
    check("the fused result keeps the input shape and dtype",
          fused.shape == latent.shape and fused.dtype == latent.dtype,
          f"got {tuple(fused.shape)} {fused.dtype}")

    # A model returning a constant must also come back as that constant, which
    # catches a weight accumulator that is off anywhere on the canvas.
    returns_constant = lambda tile, timestep, **conditioning: torch.full_like(tile, 3.5)
    constant_fused = denoise_latent_as_fused_tiles(
        returns_constant, latent, timestep=None, conditioning={},
        plan=plan, rope_offset_holder=None)
    check("a constant prediction fuses to that constant across the whole canvas",
          torch.allclose(constant_fused, torch.full_like(latent, 3.5), atol=1e-5),
          f"range {float(constant_fused.min()):.4f}..{float(constant_fused.max()):.4f}")

    tiles_seen = []
    record_tile_shapes = lambda tile, timestep, **conditioning: (
        tiles_seen.append(tuple(tile.shape)) or tile)
    denoise_latent_as_fused_tiles(record_tile_shapes, latent, timestep=None,
                                 conditioning={}, plan=plan, rope_offset_holder=None)
    check("the model is called once per planned tile",
          len(tiles_seen) == len(plan.tiles), f"got {len(tiles_seen)} calls")
    check("every tile keeps the full batch and channel count",
          all(shape[0] == 2 and shape[1] == 16 for shape in tiles_seen),
          f"got {set((s[0], s[1]) for s in tiles_seen)}")

    # Krea2's latent format is Wan21 and its forward branches on x.ndim == 5, so a
    # latent arrives as (batch, channels, frames, height, width). Slicing fixed
    # dimensions 2 and 3 would cut time and height instead of height and width.
    temporal_latent = torch.randn(2, 16, 1, 228, 156)
    temporal_fused = denoise_latent_as_fused_tiles(
        returns_input_unchanged, temporal_latent, timestep=None, conditioning={},
        plan=plan, rope_offset_holder=None)
    check("a five dimensional latent fuses back to itself",
          torch.allclose(temporal_fused, temporal_latent, atol=1e-5),
          f"max deviation {float((temporal_fused - temporal_latent).abs().max()):.3e}")
    check("a five dimensional latent keeps its shape",
          temporal_fused.shape == temporal_latent.shape,
          f"got {tuple(temporal_fused.shape)}")

    temporal_tiles_seen = []
    record_temporal_tiles = lambda tile, timestep, **conditioning: (
        temporal_tiles_seen.append(tuple(tile.shape)) or tile)
    denoise_latent_as_fused_tiles(record_temporal_tiles, temporal_latent,
                                 timestep=None, conditioning={}, plan=plan,
                                 rope_offset_holder=None)
    check("five dimensional tiles are cut on height and width, not time",
          all(shape[2] == 1 and shape[3] <= 228 and shape[4] <= 156
              for shape in temporal_tiles_seen),
          f"got {sorted(set(temporal_tiles_seen))}")
    check("five dimensional tiles match the planned tile sizes",
          {(shape[3], shape[4]) for shape in temporal_tiles_seen}
          == {(tile.height, tile.width) for tile in plan.tiles},
          f"got {sorted({(s[3], s[4]) for s in temporal_tiles_seen})}")

    single_tile_plan = plan_latent_tiles(latent_width=64, latent_height=64,
                                         tile_grid="1x1", tile_overlap_pixels=128)
    single_calls = []
    count_calls = lambda tile, timestep, **conditioning: (
        single_calls.append(tuple(tile.shape)) or tile)
    small_latent = torch.randn(1, 16, 64, 64)
    denoise_latent_as_fused_tiles(count_calls, small_latent, timestep=None,
                                 conditioning={}, plan=single_tile_plan,
                                 rope_offset_holder=None)
    check("a single tile skips the fusion and calls the model once, whole",
          single_calls == [(1, 16, 64, 64)], f"got {single_calls}")

    weight = build_tile_fusion_weight(8, 6, 4, torch.device("cpu"), torch.float32)
    check("the fusion weight is shaped for broadcasting over batch and channels",
          tuple(weight.shape) == (1, 1, 8, 6), f"got {tuple(weight.shape)}")
    temporal_weight = build_tile_fusion_weight(8, 6, 5, torch.device("cpu"),
                                               torch.float32)
    check("the fusion weight gains a leading axis for a five dimensional latent",
          tuple(temporal_weight.shape) == (1, 1, 1, 8, 6),
          f"got {tuple(temporal_weight.shape)}")
    check("the fusion weight is strictly positive everywhere",
          bool((weight > 0).all()), f"min {float(weight.min()):.6f}")
    check("the fusion weight is largest at the tile centre",
          float(weight[0, 0, 4, 3]) > float(weight[0, 0, 0, 0]))

    # stable-diffusion.cpp:3126 divides the latent start by the patch size of 2.
    holder = TileRopeOffsetHolder()
    holder.set_to_tile_origin(row_start=54, column_start=23)
    check("the rope offset is the latent origin in token units",
          (holder.row_offset_tokens, holder.column_offset_tokens)
          == (54 // KREA2_PATCH_SIZE, 23 // KREA2_PATCH_SIZE),
          f"got {holder.row_offset_tokens}, {holder.column_offset_tokens}")
    holder.clear()
    check("clearing the holder returns both offsets to zero",
          (holder.row_offset_tokens, holder.column_offset_tokens) == (0, 0))

    patch = build_post_input_rope_offset_patch(holder)
    original_ids = torch.zeros(1, 4, 3)
    original_ids[..., 1] = torch.tensor([0.0, 0.0, 1.0, 1.0])
    original_ids[..., 2] = torch.tensor([0.0, 1.0, 0.0, 1.0])

    unchanged = patch({"img_ids": original_ids.clone()})["img_ids"]
    check("a cleared holder leaves position ids untouched",
          torch.equal(unchanged, original_ids))

    holder.set_to_tile_origin(row_start=54, column_start=22)
    shifted = patch({"img_ids": original_ids.clone()})["img_ids"]
    check("the patch shifts rows by the tile's token row offset",
          torch.allclose(shifted[..., 1], original_ids[..., 1] + 27),
          f"got {shifted[..., 1].tolist()}")
    check("the patch shifts columns by the tile's token column offset",
          torch.allclose(shifted[..., 2], original_ids[..., 2] + 11),
          f"got {shifted[..., 2].tolist()}")
    check("the patch leaves the first position id axis alone",
          torch.allclose(shifted[..., 0], original_ids[..., 0]))

    # The loop must clear the offset even if the model raises, or the next
    # generation would silently start from a stale tile position.
    holder.clear()
    def raises_after_being_given_an_offset(tile, timestep, **conditioning):
        raise RuntimeError("model failed mid-tile")
    try:
        denoise_latent_as_fused_tiles(raises_after_being_given_an_offset, latent,
                                     timestep=None, conditioning={}, plan=plan,
                                     rope_offset_holder=holder)
    except RuntimeError:
        pass
    check("a failing model still leaves the rope offset cleared",
          (holder.row_offset_tokens, holder.column_offset_tokens) == (0, 0),
          f"got {holder.row_offset_tokens}, {holder.column_offset_tokens}")

    check("each tile is given its OWN conditioning when per-tile vision is supplied",
          every_tile_receives_its_own_conditioning(plan))
    check("without per-tile vision every tile still shares one conditioning",
          every_tile_receives_the_same_conditioning(plan))

    if failures:
        print(f"\n{len(failures)} failing checks:")
        for failed in failures:
            print(f"  - {failed}")
        return 1
    print("\nall krea2 tiled denoise checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
