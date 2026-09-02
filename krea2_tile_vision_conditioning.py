"""Cropping the reference image to the sampler's tiles, so each tile has its own vision.

Vision tokens describe whatever image the tower was shown. When every tile receives
the same conditioning, every tile is told to render the whole composition, which is
why tiles invent their own copy of the subject. Here each tile is encoded from the
region it will actually draw.

The reference image is ALREADY at the target resolution - the same pixels that were
encoded into the latent the sampler tiles. That single source of truth is what keeps
crops and tiles in correspondence: the latent geometry is the image's own size
divided by the latent scale, so a tile's crop is its latent rectangle multiplied
straight back up. Nothing here decodes a latent; the vision tower takes pixels, and
ComfyUI's own nodes do the upscaling and encoding upstream.

Per-tile prompts come from a fixed three by three grid of text fields. The grid of
FIELDS is always 3x3 so the row and column labels a user reads stay meaningful
whatever tile grid is selected; a 2x2 run simply uses fields (1,1), (1,2), (2,1)
and (2,2).
"""
from __future__ import annotations

import torch

from krea2_tile_planning import LATENT_SCALE, LatentTile, LatentTilePlan

PROMPT_FIELD_GRID_SIZE = 3


def tile_prompt_field_names() -> list[str]:
    """The nine per-tile prompt field names, row major and one indexed."""
    return [f"row_{row + 1}_col_{column + 1}_prompt"
            for row in range(PROMPT_FIELD_GRID_SIZE)
            for column in range(PROMPT_FIELD_GRID_SIZE)]


def combine_global_and_tile_prompts(global_prompt: str, tile_prompts: dict,
                                    tile_index: int, columns: int) -> str:
    """The prompt for one tile: the global text plus that tile's own field.

    `tile_index` walks the plan's tiles, rows outer and columns inner, so its row
    and column follow from the tile grid's width. Those then index the fixed three
    by three field grid, which is why a 2x2 run reads fields 1, 2, 4 and 5.
    """
    row, column = divmod(tile_index, max(1, columns))
    field_name = f"row_{row + 1}_col_{column + 1}_prompt"

    tile_text = str(tile_prompts.get(field_name, "") or "").strip()
    global_text = str(global_prompt or "").strip()
    if not tile_text:
        return global_text
    if not global_text:
        return tile_text
    return f"{global_text}, {tile_text}"


def latent_size_of_target_image(image: torch.Tensor) -> tuple[int, int]:
    """The latent (width, height) the target-size image corresponds to."""
    return (int(image.shape[2]) // LATENT_SCALE, int(image.shape[1]) // LATENT_SCALE)


def crop_image_to_latent_tile(image: torch.Tensor, tile: LatentTile) -> torch.Tensor:
    """The part of `image` the given latent tile covers, in ComfyUI's IMAGE layout.

    Exact rather than proportional: the image is the latent's own resolution
    multiplied by the latent scale, so the tile's cells map straight onto pixels.
    """
    top = tile.row_start * LATENT_SCALE
    left = tile.column_start * LATENT_SCALE
    bottom = top + tile.height * LATENT_SCALE
    right = left + tile.width * LATENT_SCALE
    return image[:, top:bottom, left:right, :]


def describe_tile_vision_plan(plan: LatentTilePlan) -> str:
    """A log line saying what each tile's vision will actually see."""
    return (f"{len(plan.tiles)} tiles ({plan.columns} x {plan.rows}), each cropped to "
            f"{plan.tile_width * LATENT_SCALE}x{plan.tile_height * LATENT_SCALE} "
            f"pixels, overlap {plan.overlap} latent cells")
