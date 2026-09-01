"""Weighting how hard the vision tokens pull, inside Krea 2 conditioning.

Ported from `scale_vlm_image_token_hidden_states` in the krea runtime's
conditioner.hpp. Only the image tokens' rows are scaled, and the result is NOT
renormalised afterwards: the runtime notes that restoring the global mean would
push the text tokens the opposite way and half-undo what was asked for.

The vision tokens describe the WHOLE image and every tile of a tiled generation
receives the same conditioning, so they are the one signal telling each tile what
the global composition is. Overlap only shares information between neighbours.

Locating the span takes no magic constants. The tokenizer leaves one slot per
image, which the model expands into many embeddings, and the encoder then strips
a prompt prefix, so

    encoded_length = token_row_length - image_count + total_vision_tokens - strip

pins the strip from values every call already has.
"""
from __future__ import annotations

import torch

# Qwen3-VL vision: 16px patches, merged 2x2 (qwen3vl.py QWEN3VL_VISION_COMMON).
VISION_PATCH_SIZE = 16
VISION_SPATIAL_MERGE = 2


def count_vision_tokens_for_image_size(height: int, width: int) -> int:
    """How many embeddings the vision tower produces for one image.

    Verified against the real encoder at 256x256, 512x256, 384x640 and 224x224.
    """
    per_side = VISION_PATCH_SIZE * VISION_SPATIAL_MERGE
    return max(1, height // per_side) * max(1, width // per_side)


def locate_vision_token_spans(image_slot_indices: list[int], token_row_length: int,
                              vision_token_counts: list[int],
                              encoded_length: int) -> list[tuple[int, int]]:
    """Where each image's tokens sit in the encoded sequence, as (start, length).

    Each slot occupies one position before encoding and many after, so every
    earlier image pushes the later ones along by its expansion less the slot it
    replaced.
    """
    if not image_slot_indices:
        return []

    prompt_prefix_tokens = (token_row_length - len(image_slot_indices)
                            + sum(vision_token_counts) - encoded_length)

    spans = []
    tokens_gained_so_far = 0
    for slot_index, token_count in zip(image_slot_indices, vision_token_counts):
        start = slot_index - prompt_prefix_tokens + tokens_gained_so_far
        spans.append((start, token_count))
        tokens_gained_so_far += token_count - 1
    return spans


def scale_vision_tokens_in_conditioning(conditioning, spans: list[tuple[int, int]],
                                        weight: float):
    """Scale only the vision token rows, leaving text and pooled output alone."""
    if weight == 1.0 or not spans:
        return conditioning

    scaled = []
    for embedding, metadata in conditioning:
        sequence_length = embedding.shape[1]
        usable = [(start, length) for start, length in spans
                  if start >= 0 and start + length <= sequence_length]
        if len(usable) != len(spans):
            return conditioning

        weighted = embedding.clone()
        for start, length in usable:
            weighted[:, start:start + length] *= weight
        scaled.append([weighted, metadata])
    return scaled
