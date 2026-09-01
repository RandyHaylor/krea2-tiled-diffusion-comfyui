"""Encode a prompt and optional images into Krea2 conditioning.

The vision tower lives inside the qwen3vl_4b text encoder, so the CLIP wired into
this node already carries it. Images become vision tokens here, at encode time,
and travel onward inside the conditioning.
"""
from __future__ import annotations

import logging
import math

import comfy.utils
from comfy_api.latest import io

from krea2_prompt_template import (
    build_krea2_conditioning_template_for_image_count,
    verify_template_matches_comfyui,
)

# The vision tower is fed a fixed pixel budget rather than the image's own size,
# matching what the other qwen3vl-family encode nodes do.
VISION_PIXEL_BUDGET = 1024 * 1024


def resize_image_to_vision_pixel_budget(image, pixel_budget: int = VISION_PIXEL_BUDGET):
    """The image scaled to hold about `pixel_budget` pixels, keeping its aspect."""
    samples = image.movedim(-1, 1)
    scale = math.sqrt(pixel_budget / (samples.shape[3] * samples.shape[2]))
    width = max(1, round(samples.shape[3] * scale))
    height = max(1, round(samples.shape[2] * scale))
    resized = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return resized.movedim(1, -1)[:, :, :, :3]


class Krea2Qwen3ImageAndTextEncoder(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Krea2Qwen3ImageAndTextEncoder",
            display_name="Krea2-Qwen3 Image and Text Encoder",
            category="conditioning/krea2",
            description=(
                "Encode a prompt for Krea 2, optionally letting the text encoder's "
                "vision tower read one or more images. The vision tower is part of "
                "the qwen3vl_4b text encoder, so no separate vision model is "
                "needed: wire in the same CLIP you already load. With no image "
                "attached this behaves as a plain Krea 2 text encode."
            ),
            inputs=[
                io.Clip.Input("clip", tooltip="The Krea 2 (qwen3vl_4b) text encoder."),
                io.String.Input(
                    "prompt", multiline=True, dynamic_prompts=True,
                    tooltip="Prompt text. Leave empty to condition on the image alone."),
                io.Image.Input(
                    "image", optional=True,
                    tooltip="Read by the vision tower. A batch is passed as several images."),
            ],
            outputs=[
                io.Conditioning.Output(
                    tooltip="Wire to the positive or negative input of a sampler."),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, image=None) -> io.NodeOutput:
        stale_template_warning = verify_template_matches_comfyui()
        if stale_template_warning is not None:
            logging.warning("Krea2-Qwen3 Image and Text Encoder: %s", stale_template_warning)

        images_for_vision_tower = []
        if image is not None:
            resized = resize_image_to_vision_pixel_budget(image)
            images_for_vision_tower = [resized[index:index + 1]
                                       for index in range(resized.shape[0])]

        # The template is passed explicitly. Left to itself the tokenizer would pick
        # qwen3vl's image template, whose second conversation turn is the assistant
        # one, and Krea2's encoder cuts its prefix at the second turn - which would
        # discard the image and the prompt. See krea2_prompt_template.
        tokens = clip.tokenize(
            prompt,
            images=images_for_vision_tower,
            llama_template=build_krea2_conditioning_template_for_image_count(
                len(images_for_vision_tower)),
        )
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))
