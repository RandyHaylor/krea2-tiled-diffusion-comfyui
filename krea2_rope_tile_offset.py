"""Giving each tile position ids at its true place on the canvas.

Krea2 builds image position ids with torch.arange over the tensor it is handed
(comfy/ldm/krea2/model.py:289-293), so a tile denoised on its own is labelled from
the origin. Left alone that is the "every tile claims the origin" behaviour, which
is what rope offsets OFF should mean.

The runtime offsets them per tile at stable-diffusion.cpp:3126-3129, in TOKEN
units: the latent start divided by the patch size of 2. ComfyUI's Krea2 defaults
patch=2 as well (ldm/krea2/model.py:234), so the same halving applies.

The offset travels through a holder that the denoise loop sets before each tile
and clears after, mirroring how the runtime sets and resets its globals.
"""
from __future__ import annotations

from dataclasses import dataclass

# ldm/krea2/model.py:234. A latent cell covers this many cells per token axis.
KREA2_PATCH_SIZE = 2


@dataclass
class TileRopeOffsetHolder:
    """The offset the post_input patch should apply to the tile being denoised.

    Mutable and shared with the patch, because ComfyUI gives the patch no way to
    learn which tile is in flight. The denoise loop is sequential, exactly as the
    runtime's is, so one holder is enough.
    """
    row_offset_tokens: int = 0
    column_offset_tokens: int = 0

    def set_to_tile_origin(self, row_start: int, column_start: int) -> None:
        self.row_offset_tokens = int(row_start) // KREA2_PATCH_SIZE
        self.column_offset_tokens = int(column_start) // KREA2_PATCH_SIZE

    def clear(self) -> None:
        self.row_offset_tokens = 0
        self.column_offset_tokens = 0


def build_post_input_rope_offset_patch(holder: TileRopeOffsetHolder):
    """A "post_input" patch that shifts image position ids to the tile's place.

    Krea2 hands the patch img_ids shaped (batch, tokens, 3), where [..., 1] is the
    row and [..., 2] the column, and takes back whatever is returned
    (ldm/krea2/model.py:346-351).
    """
    def offset_image_position_ids_for_current_tile(fields: dict) -> dict:
        if holder.row_offset_tokens == 0 and holder.column_offset_tokens == 0:
            return fields
        image_position_ids = fields["img_ids"].clone()
        image_position_ids[..., 1] += holder.row_offset_tokens
        image_position_ids[..., 2] += holder.column_offset_tokens
        fields["img_ids"] = image_position_ids
        return fields

    return offset_image_position_ids_for_current_tile
