"""The discrete sigma schedule, ported from the working runtime.

ComfyUI has no discrete scheduler among its nine (comfy/samplers.py:1365), so it
is built here and handed to a sampler that takes explicit sigmas. Nothing needs
registering: SIGMAS is only a FloatTensor.

The walk is DiscreteScheduler::get_sigmas, denoiser.hpp:32-53. It steps t linearly
from TIMESTEPS-1 down to 0 across n points, maps each through t_to_sigma, then
appends a trailing zero.

Mapping t_to_sigma across the two implementations is the subtle part.
    runtime   DiscreteFlowDenoiser::t_to_sigma, denoiser.hpp:1262
                  time_snr_shift(shift, (t + 1) / 1000)
    ComfyUI   ModelSamplingDiscreteFlow.sigma, model_sampling.py:318
                  time_snr_shift(shift, timestep / multiplier)
Krea2's sampling_settings set multiplier to 1.0 (supported_models.py:1933), NOT
the usual 1000, so its timestep domain is [0, 1]. Passing the runtime's raw t
would ask for a sigma a thousand times off the schedule. The normalised position
is computed first and scaled by whatever multiplier the model reports.

The denoise tail slice follows ComfyUI's convention, not the runtime's: ComfyUI
builds int(steps/denoise) sigmas and keeps the last steps+1, so `steps` already
means steps EXECUTED. The runtime truncates instead, which is why the krea app
pre-scales its step count. That scaling must NOT be ported here.
"""
from __future__ import annotations

import torch

# denoiser.hpp:23
RUNTIME_TIMESTEPS = 1000


def normalised_discrete_timesteps(step_count: int) -> list[float]:
    """The runtime's t walk, expressed in [0, 1] as (t + 1) / TIMESTEPS.

    Pure, so the schedule's shape can be checked without a model.
    """
    if step_count <= 0:
        return []
    largest_timestep = RUNTIME_TIMESTEPS - 1
    if step_count == 1:
        return [(largest_timestep + 1) / RUNTIME_TIMESTEPS]
    step = largest_timestep / (step_count - 1)
    return [((largest_timestep - step * index) + 1) / RUNTIME_TIMESTEPS
            for index in range(step_count)]


def scheduled_step_count_for_denoise(steps: int, denoise: float) -> int:
    """How long a schedule to build so that `steps` of it are executed.

    comfy/samplers.py:1439 and BasicScheduler both do this.
    """
    if denoise >= 1.0:
        return int(steps)
    return int(int(steps) / denoise)


def build_discrete_sigmas(model_sampling, steps: int, denoise: float) -> torch.Tensor:
    """The discrete schedule for this model, sliced to the executed steps."""
    if denoise <= 0.0:
        return torch.FloatTensor([])

    scheduled_steps = scheduled_step_count_for_denoise(steps, denoise)
    timestep_scale = float(getattr(model_sampling, "multiplier", RUNTIME_TIMESTEPS))

    sigmas = [float(model_sampling.sigma(torch.tensor(position * timestep_scale)))
              for position in normalised_discrete_timesteps(scheduled_steps)]
    sigmas.append(0.0)
    return torch.FloatTensor(sigmas)[-(int(steps) + 1):]
