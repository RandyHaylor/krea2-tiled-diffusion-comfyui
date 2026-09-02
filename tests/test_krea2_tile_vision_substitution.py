#!/usr/bin/env python3
"""Checks for giving one tile its own conditioning during a denoise step.

Each tile has its own conditioning - its crop's vision tokens encoded together
with the global prompt and that tile's own prompt text. Those differ in LENGTH
from one another, because the per-tile prompt text differs, so the tile's tensor
replaces the step's conditioning outright rather than being written into it.

ComfyUI labels the rows of the batch it hands the wrapper in `cond_or_uncond`,
0 for positive and 1 for negative (comfy/samplers.py:615, conds = [cond, uncond_]).
Only positive rows take a tile's conditioning; the negative is global and has no
per-tile counterpart.

Run with ComfyUI's interpreter:
    ComfyUI/.venv/bin/python tests/test_krea2_tile_vision_substitution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from krea2_tile_vision_substitution import (  # noqa: E402
    substitute_tile_conditioning_for_positive_rows,
)

failures: list[str] = []


def check(description: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}: {description}{(' ' + detail) if detail else ''}")
    if not passed:
        failures.append(description)


def conditioning_of_length(length: int, fill: float):
    return [[torch.full((1, length, 8), fill), {}]]


def main() -> int:
    # One positive row, the ordinary case: the tile's tensor replaces it entirely.
    single_row = {"c_crossattn": torch.full((1, 10, 8), 1.0)}
    replaced = substitute_tile_conditioning_for_positive_rows(
        single_row, [0], conditioning_of_length(10, 7.0))
    check("a lone positive row takes the tile's conditioning",
          torch.allclose(replaced["c_crossattn"], torch.full((1, 10, 8), 7.0)))
    check("the original dict is not mutated",
          torch.allclose(single_row["c_crossattn"], torch.full((1, 10, 8), 1.0)))

    # The whole point of replacing rather than writing in: lengths need not match,
    # because per-tile prompt text makes tiles differ in length from each other.
    longer = substitute_tile_conditioning_for_positive_rows(
        {"c_crossattn": torch.full((1, 10, 8), 1.0)}, [0],
        conditioning_of_length(162, 7.0))
    check("a LONGER tile conditioning replaces a shorter step conditioning",
          longer["c_crossattn"].shape[1] == 162,
          f"got {tuple(longer['c_crossattn'].shape)}")
    shorter = substitute_tile_conditioning_for_positive_rows(
        {"c_crossattn": torch.full((1, 162, 8), 1.0)}, [0],
        conditioning_of_length(10, 7.0))
    check("a SHORTER tile conditioning replaces a longer step conditioning",
          shorter["c_crossattn"].shape[1] == 10,
          f"got {tuple(shorter['c_crossattn'].shape)}")

    check("the replacement keeps the batch row count",
          batch_row_count_is_preserved())

    # A batch holding both: only the positive row changes. Such a batch can only
    # exist when the shapes already match (comfy/conds.py:37), so writing rows in
    # is safe here.
    stacked = {"c_crossattn": torch.stack([torch.full((10, 8), 1.0),
                                           torch.full((10, 8), 2.0)])}
    result = substitute_tile_conditioning_for_positive_rows(
        stacked, [0, 1], conditioning_of_length(10, 7.0))
    check("in a mixed batch the positive row takes the tile's conditioning",
          torch.allclose(result["c_crossattn"][0], torch.full((10, 8), 7.0)))
    check("in a mixed batch the negative row is left alone",
          torch.allclose(result["c_crossattn"][1], torch.full((10, 8), 2.0)))

    negative_first = {"c_crossattn": torch.stack([torch.full((10, 8), 2.0),
                                                  torch.full((10, 8), 1.0)])}
    by_label = substitute_tile_conditioning_for_positive_rows(
        negative_first, [1, 0], conditioning_of_length(10, 7.0))
    check("substitution follows the labels, not the row order",
          torch.allclose(by_label["c_crossattn"][0], torch.full((10, 8), 2.0))
          and torch.allclose(by_label["c_crossattn"][1], torch.full((10, 8), 7.0)))

    check("a mixed batch whose lengths differ is refused rather than mangled",
          mixed_batch_of_unequal_length_is_refused())

    check("a batch with no positive rows is returned untouched",
          no_positive_rows_is_untouched())

    without_crossattn = {"y": torch.zeros(1, 4)}
    check("a conditioning dict carrying no c_crossattn is returned untouched",
          substitute_tile_conditioning_for_positive_rows(
              without_crossattn, [0], conditioning_of_length(10, 7.0))
          is without_crossattn)

    check("keys other than c_crossattn are carried through unchanged",
          other_keys_survive())

    if failures:
        print(f"\n{len(failures)} failing checks:")
        for failed in failures:
            print(f"  - {failed}")
        return 1
    print("\nall krea2 tile vision substitution checks passed")
    return 0


def batch_row_count_is_preserved() -> bool:
    two_positive_rows = {"c_crossattn": torch.full((2, 10, 8), 1.0)}
    result = substitute_tile_conditioning_for_positive_rows(
        two_positive_rows, [0, 0], conditioning_of_length(162, 7.0))
    return tuple(result["c_crossattn"].shape) == (2, 162, 8)


def mixed_batch_of_unequal_length_is_refused() -> bool:
    mixed = {"c_crossattn": torch.full((2, 10, 8), 1.0)}
    result = substitute_tile_conditioning_for_positive_rows(
        mixed, [0, 1], conditioning_of_length(162, 7.0))
    return result is mixed


def other_keys_survive() -> bool:
    transformer_options = {"sigmas": "whatever the sampler put here"}
    conditioning = {"c_crossattn": torch.full((1, 10, 8), 1.0),
                    "y": torch.zeros(1, 4),
                    "transformer_options": transformer_options}
    result = substitute_tile_conditioning_for_positive_rows(
        conditioning, [0], conditioning_of_length(10, 7.0))
    return (torch.allclose(result["y"], torch.zeros(1, 4))
            and result["transformer_options"] is transformer_options)


def no_positive_rows_is_untouched() -> bool:
    only_negative = {"c_crossattn": torch.full((1, 10, 8), 2.0)}
    result = substitute_tile_conditioning_for_positive_rows(
        only_negative, [1], conditioning_of_length(10, 7.0))
    return result is only_negative


if __name__ == "__main__":
    raise SystemExit(main())
