#!/usr/bin/env python3
"""Checks for the ported discrete sigma schedule.

The reference values are recomputed from the runtime's own formulas:
DiscreteScheduler::get_sigmas (denoiser.hpp:32-53) walking t and
DiscreteFlowDenoiser::t_to_sigma (denoiser.hpp:1262) mapping it, against
ComfyUI's real ModelSamplingDiscreteFlow.

Run with ComfyUI's interpreter:
    ComfyUI/.venv/bin/python tests/test_krea2_discrete_sigmas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, "/media/aikenyon/NVME_2/ubuntu_comfy/ComfyUI")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comfy.model_sampling import ModelSamplingDiscreteFlow, time_snr_shift  # noqa: E402

from krea2_discrete_sigmas import (  # noqa: E402
    RUNTIME_TIMESTEPS,
    build_discrete_sigmas,
    normalised_discrete_timesteps,
    scheduled_step_count_for_denoise,
)

failures: list[str] = []


def check(description: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}: {description}{(' ' + detail) if detail else ''}")
    if not passed:
        failures.append(description)


class Krea2LikeModelConfig:
    """Krea2's sampling settings, supported_models.py:1933-1936."""
    sampling_settings = {"multiplier": 1.0, "shift": 1.15}


def main() -> int:
    check("the runtime's timestep count is a thousand", RUNTIME_TIMESTEPS == 1000)

    positions = normalised_discrete_timesteps(4)
    check("the walk starts at the largest timestep, normalised to one",
          abs(positions[0] - 1.0) < 1e-9, f"got {positions[0]}")
    check("the walk ends one timestep above zero, matching the runtime's t + 1",
          abs(positions[-1] - 1 / 1000) < 1e-9, f"got {positions[-1]}")
    check("the walk descends", positions == sorted(positions, reverse=True),
          f"got {positions}")
    check("a single step asks only for the largest timestep",
          normalised_discrete_timesteps(1) == [1.0])
    check("no steps gives no positions", normalised_discrete_timesteps(0) == [])

    # comfy/samplers.py:1439
    check("a full denoise schedules exactly the requested steps",
          scheduled_step_count_for_denoise(8, 1.0) == 8)
    check("a partial denoise schedules a longer run so the tail is the requested steps",
          scheduled_step_count_for_denoise(8, 0.75) == int(8 / 0.75),
          f"got {scheduled_step_count_for_denoise(8, 0.75)}")

    model_sampling = ModelSamplingDiscreteFlow(Krea2LikeModelConfig())
    check("the model reports Krea2's unusual multiplier of one",
          float(model_sampling.multiplier) == 1.0,
          f"got {model_sampling.multiplier}")

    sigmas = build_discrete_sigmas(model_sampling, steps=8, denoise=1.0)
    check("a full denoise yields steps + 1 sigmas",
          len(sigmas) == 9, f"got {len(sigmas)}")
    check("the schedule ends at zero", float(sigmas[-1]) == 0.0)
    check("the schedule descends",
          all(float(sigmas[i]) > float(sigmas[i + 1]) for i in range(len(sigmas) - 1)),
          f"got {[round(float(s), 4) for s in sigmas]}")

    # The runtime's own arithmetic, recomputed here rather than copied from a run.
    expected_first = float(time_snr_shift(1.15, torch.tensor(1.0)))
    check("the first sigma matches the runtime's t_to_sigma at the largest timestep",
          abs(float(sigmas[0]) - expected_first) < 1e-6,
          f"got {float(sigmas[0])}, expected {expected_first}")

    # Krea2's multiplier of 1.0 is the trap: reading the raw runtime t would ask
    # for sigma(999) instead of sigma(0.999...) and land far off the schedule.
    check("the sigmas stay within the model's own range",
          float(sigmas[0]) <= float(model_sampling.sigma_max) + 1e-6,
          f"got {float(sigmas[0])} against sigma_max {float(model_sampling.sigma_max)}")

    partial = build_discrete_sigmas(model_sampling, steps=8, denoise=0.75)
    check("a partial denoise still yields steps + 1 sigmas",
          len(partial) == 9, f"got {len(partial)}")
    check("a partial denoise starts lower than a full one, because it keeps the tail",
          float(partial[0]) < float(sigmas[0]),
          f"partial {float(partial[0])} vs full {float(sigmas[0])}")

    check("a zero denoise yields an empty schedule",
          len(build_discrete_sigmas(model_sampling, steps=8, denoise=0.0)) == 0)

    check("the result is a plain FloatTensor, which is all SIGMAS is",
          isinstance(sigmas, torch.Tensor) and sigmas.dtype == torch.float32,
          f"got {type(sigmas).__name__} {sigmas.dtype}")

    if failures:
        print(f"\n{len(failures)} failing checks:")
        for failed in failures:
            print(f"  - {failed}")
        return 1
    print("\nall krea2 discrete sigma checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
