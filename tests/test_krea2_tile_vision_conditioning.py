#!/usr/bin/env python3
"""Checks for cropping the target-size image to the sampler's tiles.

The encoder is given the reference image ALREADY at the target resolution - the
same pixels that were encoded into the latent the sampler tiles. The latent
geometry is therefore the image's own size divided by the latent scale, and a
tile's crop is its latent rectangle multiplied straight back up. One source of
truth, no VAE, no decode.

Per-tile prompts come from a fixed three by three grid of text fields, so the
labels a user sees stay meaningful whatever grid is selected: a 2x2 run uses
fields (1,1), (1,2), (2,1), (2,2) out of the nine.

Run with ComfyUI's interpreter:
    ComfyUI/.venv/bin/python tests/test_krea2_tile_vision_conditioning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from krea2_tile_planning import LATENT_SCALE, plan_latent_tiles  # noqa: E402
from krea2_tile_vision_conditioning import (  # noqa: E402
    PROMPT_FIELD_GRID_SIZE,
    combine_global_and_tile_prompts,
    crop_image_to_latent_tile,
    describe_tile_vision_plan,
    latent_size_of_target_image,
    tile_prompt_field_names,
)

failures: list[str] = []


def check(description: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}: {description}{(' ' + detail) if detail else ''}")
    if not passed:
        failures.append(description)


def main() -> int:
    check("a 1024 square image is a 128 square latent",
          latent_size_of_target_image(torch.zeros(1, 1024, 1024, 3)) == (128, 128),
          f"got {latent_size_of_target_image(torch.zeros(1, 1024, 1024, 3))}")
    check("a 1248x1824 image is a 156x228 latent",
          latent_size_of_target_image(torch.zeros(1, 1824, 1248, 3)) == (156, 228),
          f"got {latent_size_of_target_image(torch.zeros(1, 1824, 1248, 3))}")

    image = torch.zeros(1, 1024, 1024, 3)
    latent_width, latent_height = latent_size_of_target_image(image)
    plan = plan_latent_tiles(latent_width=latent_width, latent_height=latent_height,
                             tile_grid="2x2", tile_overlap_pixels=256)

    crops = [crop_image_to_latent_tile(image, tile) for tile in plan.tiles]
    check("a 2x2 plan yields four crops", len(crops) == 4, f"got {len(crops)}")
    check("every crop keeps the batch and channel axes",
          all(c.shape[0] == 1 and c.shape[3] == 3 for c in crops))
    check("all crops are the same size, because all tiles are",
          len({(c.shape[1], c.shape[2]) for c in crops}) == 1,
          f"got {sorted({(c.shape[1], c.shape[2]) for c in crops})}")

    first_tile = plan.tiles[0]
    check("a crop is EXACTLY the tile scaled back up to pixels",
          crops[0].shape[1] == first_tile.height * LATENT_SCALE
          and crops[0].shape[2] == first_tile.width * LATENT_SCALE,
          f"got {crops[0].shape[1]}x{crops[0].shape[2]}, expected "
          f"{first_tile.height * LATENT_SCALE}x{first_tile.width * LATENT_SCALE}")

    marked_image = torch.zeros(1, 1024, 1024, 3)
    marked_image[:, 0:64, 0:64, :] = 1.0
    check("the top-left tile sees a mark in the top-left of the image",
          float(crop_image_to_latent_tile(marked_image, plan.tiles[0]).max()) == 1.0)
    check("the bottom-right tile does not see it",
          float(crop_image_to_latent_tile(marked_image, plan.tiles[3]).max()) == 0.0)

    check("a crop starts at the tile's own origin in pixels",
          crop_starts_at_the_tile_origin())
    check("a single-tile plan crops the whole image",
          single_tile_crop_is_the_whole_image())
    check("a non-square target image crops without running past its bounds",
          non_square_image_crops_stay_in_bounds())

    # --- the fixed three by three grid of per-tile prompt fields ---
    field_names = tile_prompt_field_names()
    check("there are nine prompt fields", len(field_names) == 9, f"got {len(field_names)}")
    check("they are named by row and column, one indexed for the user",
          field_names[0] == "row_1_col_1_prompt"
          and field_names[8] == "row_3_col_3_prompt",
          f"got {field_names[0]} .. {field_names[8]}")
    check("the grid of fields is three wide", PROMPT_FIELD_GRID_SIZE == 3)

    # A 2x2 run takes fields 1, 2, 4 and 5 of the nine.
    tile_prompts = {name: "" for name in field_names}
    tile_prompts["row_1_col_1_prompt"] = "top left"
    tile_prompts["row_1_col_2_prompt"] = "top right"
    tile_prompts["row_2_col_1_prompt"] = "bottom left"
    tile_prompts["row_2_col_2_prompt"] = "bottom right"
    tile_prompts["row_3_col_3_prompt"] = "unused by a 2x2"

    combined = [combine_global_and_tile_prompts("global style", tile_prompts,
                                                tile_index=index, columns=2)
                for index in range(4)]
    check("a 2x2 run reads fields 1, 2, 4 and 5, in tile order",
          combined == ["global style, top left", "global style, top right",
                       "global style, bottom left", "global style, bottom right"],
          f"got {combined}")

    check("a 3x3 run reads all nine fields in order", three_by_three_reads_all_nine())
    check("an empty tile field leaves just the global prompt",
          combine_global_and_tile_prompts("global only", {}, tile_index=0, columns=2)
          == "global only")
    check("an empty global prompt leaves just the tile text",
          combine_global_and_tile_prompts("", tile_prompts, tile_index=0, columns=2)
          == "top left")

    description = describe_tile_vision_plan(plan)
    check("the plan description names the tile count and the crop size in pixels",
          "4" in description and "640" in description, f"got {description!r}")

    if failures:
        print(f"\n{len(failures)} failing checks:")
        for failed in failures:
            print(f"  - {failed}")
        return 1
    print("\nall krea2 tile vision conditioning checks passed")
    return 0


def three_by_three_reads_all_nine() -> bool:
    tile_prompts = {name: f"tile {index}" for index, name
                    in enumerate(tile_prompt_field_names())}
    combined = [combine_global_and_tile_prompts("g", tile_prompts, tile_index=index,
                                                columns=3) for index in range(9)]
    return combined == [f"g, tile {index}" for index in range(9)]


def crop_starts_at_the_tile_origin() -> bool:
    image = torch.arange(1024 * 1024, dtype=torch.float32).reshape(1, 1024, 1024, 1)
    image = image.repeat(1, 1, 1, 3)
    plan = plan_latent_tiles(latent_width=128, latent_height=128,
                             tile_grid="2x2", tile_overlap_pixels=256)
    tile = plan.tiles[3]
    crop = crop_image_to_latent_tile(image, tile)
    expected_first_pixel = image[0, tile.row_start * LATENT_SCALE,
                                 tile.column_start * LATENT_SCALE, 0]
    return bool(crop[0, 0, 0, 0] == expected_first_pixel)


def single_tile_crop_is_the_whole_image() -> bool:
    image = torch.zeros(1, 512, 512, 3)
    plan = plan_latent_tiles(latent_width=64, latent_height=64,
                             tile_grid="1x1", tile_overlap_pixels=256)
    crop = crop_image_to_latent_tile(image, plan.tiles[0])
    return crop.shape[1] == 512 and crop.shape[2] == 512


def non_square_image_crops_stay_in_bounds() -> bool:
    image = torch.zeros(1, 1824, 1248, 3)
    latent_width, latent_height = latent_size_of_target_image(image)
    plan = plan_latent_tiles(latent_width=latent_width, latent_height=latent_height,
                             tile_grid="2x3", tile_overlap_pixels=512)
    crops = [crop_image_to_latent_tile(image, tile) for tile in plan.tiles]
    return all(c.shape[1] > 0 and c.shape[2] > 0
               and c.shape[1] <= 1824 and c.shape[2] <= 1248 for c in crops)


if __name__ == "__main__":
    raise SystemExit(main())
