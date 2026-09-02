"""Giving one tile its own conditioning during a denoise step.

Each tile carries its own conditioning: its crop's vision tokens encoded together
with the global prompt and that tile's own prompt text. There is no separate
"positive" - a tile's conditioning IS the positive for that tile's pass.

Those conditionings differ in LENGTH from one another, because the per-tile prompt
text differs, so a tile's tensor replaces the step's conditioning outright rather
than being written into a fixed-size one.

ComfyUI labels the rows of the batch it hands the wrapper in `cond_or_uncond`,
0 for positive and 1 for negative (comfy/samplers.py:615, conds = [cond, uncond_]).
Only positive rows take a tile's conditioning; the negative is global and has no
per-tile counterpart. A batch holding both can only exist when the two already
share a shape - Krea2 wraps its cross attention in CONDRegular, whose can_concat
demands exact shape equality (comfy/conds.py:37) - so writing rows in is safe in
that case, and refused if the lengths somehow disagree.
"""
from __future__ import annotations

import logging

import torch

POSITIVE_ROW = 0


def substitute_tile_conditioning_for_positive_rows(step_conditioning: dict,
                                                   cond_or_uncond: list[int],
                                                   tile_conditioning):
    """`step_conditioning` with this tile's embedding standing in for the positive.

    Returns the original object unchanged when there is nothing to do or when the
    substitution cannot be made safely, so the caller can treat identity as
    "left as it arrived".
    """
    batched_embedding = step_conditioning.get("c_crossattn")
    if batched_embedding is None or not cond_or_uncond:
        return step_conditioning
    if POSITIVE_ROW not in cond_or_uncond:
        return step_conditioning

    tile_embedding = tile_conditioning[0][0].to(device=batched_embedding.device,
                                                dtype=batched_embedding.dtype)

    if all(row_kind == POSITIVE_ROW for row_kind in cond_or_uncond):
        substituted = tile_embedding.repeat(len(cond_or_uncond), 1, 1) \
            if tile_embedding.shape[0] == 1 else tile_embedding
        return {**step_conditioning, "c_crossattn": substituted}

    if tile_embedding.shape[1] != batched_embedding.shape[1]:
        logging.warning("Krea2 Tiled Diffusion: a tile's conditioning is %d tokens "
                        "against the %d token batch it shares with the negative; "
                        "this tile stays on the conditioning it arrived with",
                        tile_embedding.shape[1], batched_embedding.shape[1])
        return step_conditioning

    substituted = batched_embedding.clone()
    for row_index, row_kind in enumerate(cond_or_uncond):
        if row_kind == POSITIVE_ROW:
            substituted[row_index] = tile_embedding[0]
    return {**step_conditioning, "c_crossattn": substituted}
