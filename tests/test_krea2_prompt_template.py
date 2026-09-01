#!/usr/bin/env python3
"""Checks for the Krea2 conditioning template used when images are attached.

Run with plain `python3 tests/test_krea2_prompt_template.py`, no pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from krea2_prompt_template import (  # noqa: E402
    DEFAULT_VISION_SIZE,
    KREA2_CONDITIONING_TEMPLATE,
    IMAGE_START_MARKER,
    build_krea2_conditioning_template_for_image_count,
    positions_of_conversation_turn_markers,
    vision_pixel_budget_for_size,
)

failures: list[str] = []


def check(description: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}: {description}{(' ' + detail) if detail else ''}")
    if not passed:
        failures.append(description)


def main() -> int:
    no_image_template = build_krea2_conditioning_template_for_image_count(0)
    check("with no images the template is Krea2's own, untouched",
          no_image_template == KREA2_CONDITIONING_TEMPLATE)

    one_image_template = build_krea2_conditioning_template_for_image_count(1)
    check("one image inserts exactly one vision block",
          one_image_template.count("<|vision_start|><|image_pad|><|vision_end|>") == 1,
          one_image_template)
    check("three images insert three vision blocks",
          build_krea2_conditioning_template_for_image_count(3)
          .count("<|vision_start|><|image_pad|><|vision_end|>") == 3)

    # The working stable-diffusion.cpp runtime labels each vision image before its
    # block: `"Picture " + std::to_string(i + 1) + ": <|vision_start|>"` in the
    # krea2 branch of conditioner.hpp. Reproduce that exactly.
    check("each vision block is labelled Picture N, numbered from one",
          build_krea2_conditioning_template_for_image_count(3).count(
              "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
              "Picture 2: <|vision_start|><|image_pad|><|vision_end|>"
              "Picture 3: <|vision_start|><|image_pad|><|vision_end|>") == 1,
          build_krea2_conditioning_template_for_image_count(3))
    check("no Picture label appears when there are no images",
          "Picture" not in no_image_template)

    # The runtime keeps ONE system prompt for krea2 whether or not images are
    # attached; prompt_template_encode_start_idx stays 34 either way. Only other
    # architectures swap in an edit-specific system prompt.
    system_turn_of = lambda template: template.split("<|im_end|>", 1)[0]
    check("the system turn is identical with and without images",
          system_turn_of(no_image_template) == system_turn_of(one_image_template),
          system_turn_of(one_image_template))

    # Krea2TEModel.encode_token_weights locates the end of the prefix by taking the
    # position of the SECOND conversation-turn marker and requiring "user\n" right
    # after it. A template whose second marker is the assistant turn makes it strip
    # the prompt and the image away, leaving almost nothing conditioned.
    for image_count in (0, 1, 2, 5):
        template = build_krea2_conditioning_template_for_image_count(image_count)
        markers = positions_of_conversation_turn_markers(template)
        check(f"the second turn marker is the user turn, with {image_count} image(s)",
              len(markers) >= 2
              and template[markers[1]:].startswith(f"{IMAGE_START_MARKER}user\n"),
              f"got {template[markers[1]:markers[1] + 24]!r}")

    # Everything before that point is discarded, so a vision block placed ahead of
    # the user turn would be thrown away before the model ever saw it.
    for image_count in (1, 2, 5):
        template = build_krea2_conditioning_template_for_image_count(image_count)
        markers = positions_of_conversation_turn_markers(template)
        check(f"vision blocks sit after the user turn, with {image_count} image(s)",
              template.index("<|vision_start|>") > markers[1],
              f"first vision block at {template.index('<|vision_start|>')}, "
              f"user turn at {markers[1]}")

    for image_count in (0, 1, 4):
        template = build_krea2_conditioning_template_for_image_count(image_count)
        check(f"exactly one prompt slot survives, with {image_count} image(s)",
              template.count("{}") == 1, template)

    check("a negative image count is refused rather than silently treated as zero",
          refuses_negative_image_count())

    # vlm_size=N sets the runtime's min and max vision size to N, and the krea2
    # preset resizes by AREA, where those bounds are squared. The reference
    # generations ran at 512.
    check("the vision pixel budget is the square of the requested size",
          vision_pixel_budget_for_size(512) == 512 * 512)
    check("the default vision size is the 512 the reference generations used",
          DEFAULT_VISION_SIZE == 512)

    if failures:
        print(f"\n{len(failures)} failing checks:")
        for failed in failures:
            print(f"  - {failed}")
        return 1
    print("\nall krea2 prompt template checks passed")
    return 0


def refuses_negative_image_count() -> bool:
    try:
        build_krea2_conditioning_template_for_image_count(-1)
    except ValueError:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
