"""The conditioning template Krea2 is given when images are attached.

Krea2's text encoder strips the system turn and the opening of the user turn back
off after encoding, and it finds where to cut by taking the position of the SECOND
conversation-turn marker and requiring "user\\n" immediately after it. See
`Krea2TEModel.encode_token_weights` in comfy/text_encoders/krea2.py.

That is why this module exists. Krea2's own template satisfies the rule, but the
qwen3vl image template that would otherwise be selected does not: its turns are
user then assistant, so the second marker is the ASSISTANT turn, and the cut lands
past the prompt and the image instead of before them. The vision blocks are
therefore inserted into Krea2's own template rather than swapping the template
out, which keeps the system turn in place and the cut where the encoder expects.
"""
from __future__ import annotations

IMAGE_START_MARKER = "<|im_start|>"

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

# vlm_size=N sets the runtime's vision min and max size to N, and the krea2 preset
# resizes by AREA, where both bounds are squared. Every reference generation ran at
# 512, which is why that is the default rather than the runtime's unset max of 1024.
DEFAULT_VISION_SIZE = 512

# Copied from KREA2_TEMPLATE in comfy/text_encoders/krea2.py. `verify_template_matches_comfyui`
# checks the copy against the installed ComfyUI when it can be imported.
KREA2_CONDITIONING_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

USER_TURN_OPENING = f"{IMAGE_START_MARKER}user\n"


def positions_of_conversation_turn_markers(template: str) -> list[int]:
    """Where each `<|im_start|>` begins, in order."""
    positions = []
    search_from = 0
    while True:
        found = template.find(IMAGE_START_MARKER, search_from)
        if found < 0:
            return positions
        positions.append(found)
        search_from = found + len(IMAGE_START_MARKER)


def vision_pixel_budget_for_size(vision_size: int) -> int:
    """The AREA budget the runtime derives from a vlm_size, which is its square."""
    if vision_size <= 0:
        raise ValueError(f"vision size must be positive: {vision_size}")
    return vision_size * vision_size


def build_krea2_conditioning_template_for_image_count(image_count: int) -> str:
    """Krea2's template with one labelled vision block per image, before the prompt.

    Mirrors the krea2 branch of `conditioner.hpp` in the working runtime: one
    system turn regardless of images, then each image announced as "Picture N: "
    ahead of its vision block, then the prompt text.

    ComfyUI substitutes ONE `<|image_pad|>` per image (qwen3vl.py:192-195), so the
    pads are not expanded to the token count here the way the C++ does.
    """
    if image_count < 0:
        raise ValueError(f"image count cannot be negative: {image_count}")
    if image_count == 0:
        return KREA2_CONDITIONING_TEMPLATE
    labelled_vision_blocks = "".join(
        f"Picture {index + 1}: {VISION_BLOCK}" for index in range(image_count))
    return KREA2_CONDITIONING_TEMPLATE.replace(
        USER_TURN_OPENING, USER_TURN_OPENING + labelled_vision_blocks, 1)


def verify_template_matches_comfyui() -> str | None:
    """Why the copied template disagrees with the installed ComfyUI, or None.

    Returned rather than raised: a mismatch should warn loudly in the log without
    stopping a generation that will most likely still work.
    """
    try:
        from comfy.text_encoders.krea2 import KREA2_TEMPLATE
    except ImportError:
        return None
    if KREA2_TEMPLATE != KREA2_CONDITIONING_TEMPLATE:
        return ("the Krea2 conditioning template in ComfyUI has changed; "
                "krea2_prompt_template.py holds a stale copy")
    return None
