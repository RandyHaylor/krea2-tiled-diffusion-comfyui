#!/usr/bin/env python3
"""Checks for locating and weighting the vision tokens in Krea 2 conditioning.

The expected numbers are MEASURED against the real Krea2 text encoder, by
tokenising and encoding real images and solving for the prefix strip:

    1 image  (256x256): token row 50, image slot [40],     encoded 79
    2 images (256x256): token row 58, image slots [40,48], encoded 150
      -> strip = 34 (which is also conditioner.hpp's prompt_template_encode_start_idx)
      -> 64 vision tokens for a 256x256 image, and the first span starts at 6

    vision tokens = (height // 16 // 2) * (width // 16 // 2), confirmed at
    256x256, 512x256, 384x640 and 224x224.

Run with ComfyUI's interpreter:
    ComfyUI/.venv/bin/python tests/test_krea2_vision_token_weighting.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from krea2_vision_token_weighting import (  # noqa: E402
    count_vision_tokens_for_image_size,
    locate_vision_token_spans,
    scale_vision_tokens_in_conditioning,
)

failures: list[str] = []


def check(description: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}: {description}{(' ' + detail) if detail else ''}")
    if not passed:
        failures.append(description)


def main() -> int:
    for (height, width), expected in (((256, 256), 64), ((512, 256), 128),
                                      ((384, 640), 240), ((224, 224), 49)):
        check(f"a {height}x{width} image is {expected} vision tokens",
              count_vision_tokens_for_image_size(height, width) == expected,
              f"got {count_vision_tokens_for_image_size(height, width)}")

    single = locate_vision_token_spans(image_slot_indices=[40], token_row_length=50,
                                       vision_token_counts=[64], encoded_length=79)
    check("one image's span matches the measured encode",
          single == [(6, 64)], f"got {single}")

    pair = locate_vision_token_spans(image_slot_indices=[40, 48], token_row_length=58,
                                     vision_token_counts=[64, 64], encoded_length=150)
    check("two images give two spans, the second after the first",
          len(pair) == 2 and pair[0] == (6, 64) and pair[1][0] > pair[0][0] + 64,
          f"got {pair}")
    check("no span runs past the encoded sequence",
          all(start + length <= 150 for start, length in pair), f"got {pair}")

    check("no images gives no spans",
          locate_vision_token_spans([], 8, [], 8) == [])

    # The runtime scales ONLY the image token rows and deliberately does not
    # renormalise afterwards (conditioner.hpp:126-128): restoring the global mean
    # would push the text tokens the opposite way and half-undo the request.
    conditioning = [[torch.ones(1, 79, 16), {"pooled_output": torch.ones(1, 16)}]]
    scaled = scale_vision_tokens_in_conditioning(conditioning, [(6, 64)], 2.0)
    embedding = scaled[0][0]
    check("vision token rows are scaled by the weight",
          torch.allclose(embedding[0, 6:70], torch.full((64, 16), 2.0)),
          f"got {float(embedding[0, 6].mean())}")
    check("text rows before the vision span are untouched",
          torch.allclose(embedding[0, :6], torch.ones(6, 16)))
    check("text rows after the vision span are untouched",
          torch.allclose(embedding[0, 70:], torch.ones(9, 16)))
    check("the pooled output is left alone, being no part of the token sequence",
          torch.allclose(scaled[0][1]["pooled_output"], torch.ones(1, 16)))

    unchanged = scale_vision_tokens_in_conditioning(conditioning, [(6, 64)], 1.0)
    check("a weight of one returns the conditioning untouched",
          unchanged is conditioning)

    check("a span outside the sequence is refused rather than silently clipped",
          refuses_out_of_range_span())

    if failures:
        print(f"\n{len(failures)} failing checks:")
        for failed in failures:
            print(f"  - {failed}")
        return 1
    print("\nall krea2 vision token weighting checks passed")
    return 0


def refuses_out_of_range_span() -> bool:
    conditioning = [[torch.ones(1, 20, 16), {}]]
    scaled = scale_vision_tokens_in_conditioning(conditioning, [(15, 64)], 2.0)
    # Out of range means the weighting is skipped, not that the tensor is mangled.
    return torch.allclose(scaled[0][0], torch.ones(1, 20, 16))


if __name__ == "__main__":
    raise SystemExit(main())
