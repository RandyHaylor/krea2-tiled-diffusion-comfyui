"""Encode a prompt and a reference image into Krea2 conditioning, per tile.

The vision tower lives inside the qwen3vl_4b text encoder, so the CLIP wired into
this node already carries it. Images become vision tokens here, at encode time,
and travel onward inside the conditioning.

There is one conditioning per tile and nothing else. Each is its own crop's vision
tokens encoded TOGETHER with the global prompt and that tile's own prompt text -
vision cannot be attached to an already-encoded conditioning afterwards, so the
two must go through the encoder in one pass. A tile's conditioning IS the positive
for that tile's denoise; there is no separate global positive.

At 1x1 there is a single tile covering everything, so that one encode is the whole
image. That is the mode for testing rather than for tiled operation.

The reference image must ALREADY be at the target resolution - the same pixels
that were encoded into the latent wired to the sampler. This node never resizes,
crops to fit, or touches a VAE; ComfyUI's own nodes do that upstream.
"""
from __future__ import annotations

import logging
import math

import comfy.utils
from comfy_api.latest import io

from krea2_prompt_template import (
    DEFAULT_VISION_SIZE,
    build_krea2_conditioning_template_for_image_count,
    verify_template_matches_comfyui,
    vision_pixel_budget_for_size,
)
from krea2_tile_planning import (
    DEFAULT_TILE_GRID,
    DEFAULT_TILE_OVERLAP_PIXELS,
    TILE_GRIDS,
    parse_tile_grid,
    plan_latent_tiles,
)
from krea2_tile_vision_conditioning import (
    combine_global_and_tile_prompts,
    crop_image_to_latent_tile,
    describe_tile_vision_plan,
    latent_size_of_target_image,
    tile_prompt_field_names,
)
from krea2_vision_token_weighting import (
    count_vision_tokens_for_image_size,
    locate_vision_token_spans,
    scale_vision_tokens_in_conditioning,
)

Krea2TileVision = io.Custom("KREA2_TILE_VISION")


def resize_image_to_vision_pixel_budget(image, pixel_budget: int):
    """The image scaled to hold about `pixel_budget` pixels, keeping its aspect."""
    samples = image.movedim(-1, 1)
    scale = math.sqrt(pixel_budget / (samples.shape[3] * samples.shape[2]))
    width = max(1, round(samples.shape[3] * scale))
    height = max(1, round(samples.shape[2] * scale))
    resized = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return resized.movedim(1, -1)[:, :, :, :3]


def weight_the_vision_tokens(conditioning, tokens, images, weight: float):
    """Scale the vision tokens' rows, warning rather than raising if unlocatable."""
    token_row = tokens[next(iter(tokens))][0]
    image_slot_indices = [index for index, entry in enumerate(token_row)
                          if isinstance(entry[0], dict)]
    vision_token_counts = [count_vision_tokens_for_image_size(
        int(single_image.shape[1]), int(single_image.shape[2])) for single_image in images]

    if len(image_slot_indices) != len(vision_token_counts):
        logging.warning("Krea2-Qwen3 Image and Text Encoder: found %d image slot(s) for "
                        "%d image(s); leaving the vision weight unapplied",
                        len(image_slot_indices), len(vision_token_counts))
        return conditioning

    spans = locate_vision_token_spans(
        image_slot_indices=image_slot_indices,
        token_row_length=len(token_row),
        vision_token_counts=vision_token_counts,
        encoded_length=int(conditioning[0][0].shape[1]))
    weighted = scale_vision_tokens_in_conditioning(conditioning, spans, weight)
    if weighted is conditioning:
        logging.warning("Krea2-Qwen3 Image and Text Encoder: vision token spans %s do "
                        "not fit the %d token conditioning; weight not applied",
                        spans, int(conditioning[0][0].shape[1]))
    else:
        logging.info("Krea2-Qwen3 Image and Text Encoder: vision weight %.3g on %s",
                     weight, spans)
    return weighted


def encode_prompt_with_images(clip, prompt: str, images: list, vision_weight: float):
    """One Krea2 conditioning from a prompt and however many images accompany it.

    The template is passed explicitly. Left to itself the tokenizer would pick
    qwen3vl's image template, whose second conversation turn is the assistant one,
    and Krea2's encoder cuts its prefix at the second turn - which would discard
    the image and the prompt. See krea2_prompt_template.
    """
    tokens = clip.tokenize(
        prompt,
        images=images,
        llama_template=build_krea2_conditioning_template_for_image_count(len(images)),
    )
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if images and vision_weight != 1.0:
        conditioning = weight_the_vision_tokens(conditioning, tokens, images,
                                                vision_weight)
    return conditioning


def encode_one_conditioning_per_tile(clip, global_prompt: str, tile_prompts: dict,
                                     reference_image, pixel_budget: int,
                                     vision_weight: float, tile_grid: str,
                                     tile_overlap: int):
    """A conditioning for every tile, each seeing only the region it will draw.

    Ordered rows outer and columns inner, matching LatentTilePlan.tiles, which is
    the order the sampler visits. With no reference image there is nothing to crop
    and the tiles carry text alone, which is regional prompting without vision.
    """
    columns, rows = parse_tile_grid(tile_grid)
    crops_per_tile = [None] * (columns * rows)

    if reference_image is not None:
        latent_width, latent_height = latent_size_of_target_image(reference_image)
        plan = plan_latent_tiles(latent_width=latent_width,
                                 latent_height=latent_height,
                                 tile_grid=tile_grid, tile_overlap_pixels=tile_overlap)
        logging.info("Krea2-Qwen3 Image and Text Encoder: %s",
                     describe_tile_vision_plan(plan))
        columns = plan.columns
        crops_per_tile = [
            resize_image_to_vision_pixel_budget(
                crop_image_to_latent_tile(reference_image, tile), pixel_budget)
            for tile in plan.tiles]

    conditioning_per_tile = [
        encode_prompt_with_images(
            clip,
            combine_global_and_tile_prompts(global_prompt, tile_prompts,
                                            tile_index, columns),
            [] if crop is None else [crop],
            vision_weight)
        for tile_index, crop in enumerate(crops_per_tile)]

    return {
        "tile_grid": tile_grid,
        "tile_overlap": tile_overlap,
        "conditioning_per_tile": conditioning_per_tile,
    }


class Krea2Qwen3ImageAndTextEncoder(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Krea2Qwen3ImageAndTextEncoder",
            display_name="Krea2-Qwen3 Image and Text Encoder",
            category="conditioning/krea2",
            description=(
                "Encode a prompt for Krea 2 with the text encoder's own vision "
                "tower reading a reference image, one conditioning per tile. The "
                "vision tower is part of the qwen3vl_4b text encoder, so no "
                "separate vision model is needed: wire in the same CLIP you "
                "already load.\n\n"
                "Above a 1x1 grid each tile is encoded from ITS OWN crop, and the "
                "CONDITIONING output carries text alone - all vision travels in "
                "tile_vision. The reference image must already be at the target "
                "resolution, the same pixels encoded into the sampler's latent; "
                "this node never resizes or crops to fit.\n\n"
                "The per-tile prompt fields are always a 3x3 grid so their labels "
                "stay meaningful whatever tile grid is chosen. A 2x2 run uses the "
                "(1,1), (1,2), (2,1) and (2,2) fields."
            ),
            inputs=[
                io.Clip.Input("clip", tooltip="The Krea 2 (qwen3vl_4b) text encoder."),
                io.String.Input(
                    "prompt", multiline=True, dynamic_prompts=True,
                    tooltip="Global prompt, applied to every tile. Keep it to what "
                            "is true of EVERY tile - style, medium, palette. "
                            "Subject matter that appears in only part of the "
                            "picture belongs in that tile's own field, or it will "
                            "be asked for in every tile."),
                io.Image.Input(
                    "reference_image", optional=True,
                    tooltip="ALREADY at the target resolution: the same pixels "
                            "encoded into the latent wired to the sampler. Read by "
                            "the vision tower, cropped per tile."),
                io.Int.Input(
                    "vision_size", default=DEFAULT_VISION_SIZE, min=64, max=2048, step=64,
                    tooltip="Each tile's crop is resized to about this many pixels "
                            "square before the vision tower reads it."),
                io.Float.Input(
                    "vision_weight", default=1.0, min=0.0, max=10.0, step=0.05,
                    tooltip="How hard the vision tokens pull against the text. "
                            "Exactly 1 leaves the conditioning alone."),
                io.Combo.Input(
                    "tile_grid", options=TILE_GRIDS, default=DEFAULT_TILE_GRID,
                    tooltip="Must match the sampler's tile_grid."),
                io.Int.Input(
                    "tile_overlap", default=DEFAULT_TILE_OVERLAP_PIXELS,
                    min=8, max=2048, step=8,
                    tooltip="Must match the sampler's tile_overlap, or the crops "
                            "will not line up with the tiles."),
            ] + [
                io.String.Input(
                    field_name, multiline=True, dynamic_prompts=True, optional=True,
                    tooltip=f"Prompt text for the tile at {field_name.replace('_', ' ')}"
                            f", added to the global prompt for that tile only. "
                            f"Unused when the grid is smaller than this position.")
                for field_name in tile_prompt_field_names()
            ],
            outputs=[
                Krea2TileVision.Output(
                    display_name="tile_vision",
                    tooltip="One conditioning per tile, each its own region's "
                            "vision encoded together with the global prompt and "
                            "that tile's prompt. Wire to the sampler's tile_vision "
                            "input."),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, reference_image=None,
                vision_size=DEFAULT_VISION_SIZE,
                vision_weight=1.0,
                tile_grid=DEFAULT_TILE_GRID,
                tile_overlap=DEFAULT_TILE_OVERLAP_PIXELS,
                **tile_prompts) -> io.NodeOutput:
        stale_template_warning = verify_template_matches_comfyui()
        if stale_template_warning is not None:
            logging.warning("Krea2-Qwen3 Image and Text Encoder: %s", stale_template_warning)

        return io.NodeOutput(encode_one_conditioning_per_tile(
            clip, prompt, tile_prompts, reference_image,
            vision_pixel_budget_for_size(vision_size), vision_weight,
            tile_grid, tile_overlap))
