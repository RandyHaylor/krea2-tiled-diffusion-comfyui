"""The Krea2 Tiled Diffusion sampler node.

Applies the identity LoRA, installs the per-step tile fusion, builds the discrete
sigma schedule, and samples. The tiling and the schedule are ours; ComfyUI does
the sampling.
"""
from __future__ import annotations

import logging

import comfy.samplers
import comfy.sample
import comfy.utils
import folder_paths
import latent_preview
import node_helpers
from comfy_api.latest import io

from krea2_discrete_sigmas import build_discrete_sigmas
from krea2_rope_tile_offset import (
    TileRopeOffsetHolder,
    build_post_input_rope_offset_patch,
)
from krea2_tile_planning import plan_latent_tiles
from krea2_tiled_denoise import build_tiled_denoise_wrapper

DISCRETE_SCHEDULER_NAME = "discrete"
NO_LORA_SELECTED = "none"

TILE_GRIDS = ["1x1", "1x2", "2x1", "2x2", "2x3", "3x2", "3x3"]

# From the reference generations. See DESIGN.md for the two recipes.
DEFAULT_TILE_GRID = "2x2"
DEFAULT_TILE_OVERLAP_PIXELS = 256
DEFAULT_STEPS = 8
DEFAULT_DENOISE = 0.75
DEFAULT_CFG = 1.0


def scale_conditioning(conditioning, multiplier: float):
    """Scale the encoded prompt, mirroring ConditioningMultiply (nodes.py:174).

    The embedding itself is scaled, not the `strength` key: strength is consumed
    as an area blending weight when several conditionings are composed
    (samplers.py:52), which is a different operation entirely.
    """
    if multiplier == 1.0:
        return conditioning
    scaled = []
    for embedding, metadata in conditioning:
        updated = {}
        pooled_output = metadata.get("pooled_output", None)
        if pooled_output is not None:
            updated["pooled_output"] = pooled_output * multiplier
        scaled.append(node_helpers.conditioning_set_values(
            [[embedding * multiplier, metadata]], updated)[0])
    return scaled


def apply_identity_lora(model, lora_name: str, strength: float):
    """The Krea2 identity edit LoRA, or the model untouched.

    Never raises for a missing selection: the node says on its face that the LoRA
    is required for high quality, so running without it is a visible choice rather
    than a silent downgrade.
    """
    if not lora_name or lora_name == NO_LORA_SELECTED or strength == 0.0:
        return model
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if lora_path is None:
        logging.warning("Krea2 Tiled Diffusion: LoRA %s was not found; "
                        "sampling without it", lora_name)
        return model
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    import comfy.sd
    return comfy.sd.load_lora_for_models(model, None, lora, strength, 0)[0]


def available_lora_names() -> list[str]:
    """The LoRA dropdown's options, read fresh each time the schema is built.

    Called while building the schema rather than handed over as a callable:
    define_schema runs per /object_info request, so the list stays current, and
    anything left unserialisable in a schema takes that whole endpoint down.
    ComfyUI's own LoraLoaderModelOnly calls it the same way (nodes.py:760).
    """
    return [NO_LORA_SELECTED] + folder_paths.get_filename_list("loras")


class Krea2TiledDiffusion(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Krea2TiledDiffusion",
            display_name="Krea2 Tiled Diffusion",
            category="sampling/krea2",
            description=(
                "Denoise a latent as overlapping tiles that are fused every step, "
                "for Krea 2. Because the fusion happens inside each step rather "
                "than after each tile finishes, neighbouring tiles cannot drift "
                "apart. Expects a latent ALREADY at the target resolution.\n\n"
                "Two recipes work, and the middle between them is worse than "
                "either. High denoise (the defaults: 2x2, 256px overlap, 8 steps, "
                "0.75) rebuilds a new high resolution image. Low denoise (3x3, "
                "512px overlap, 4 steps, 0.10) enhances what is already there.\n\n"
                "Needs a turbo model or a turbo LoRA."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input(
                    "upscaled_reference_image_latent",
                    tooltip="A latent ALREADY at the target resolution. This node "
                            "does no resizing; upscale it upstream."),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=True),
                io.Int.Input("steps", default=DEFAULT_STEPS, min=1, max=10000,
                             tooltip="Steps EXECUTED. ComfyUI lengthens the "
                                     "schedule for a partial denoise itself."),
                io.Float.Input("cfg", default=DEFAULT_CFG, min=0.0, max=100.0, step=0.1),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS,
                               default="euler"),
                io.Combo.Input("scheduler",
                               options=[DISCRETE_SCHEDULER_NAME]
                                       + list(comfy.samplers.SCHEDULER_NAMES),
                               default=DISCRETE_SCHEDULER_NAME,
                               tooltip="discrete is ported from the Krea 2 runtime; "
                                       "ComfyUI does not ship one."),
                io.Float.Input("denoise", default=DEFAULT_DENOISE, min=0.0, max=1.0,
                               step=0.01,
                               tooltip="0.75 rebuilds at high resolution. 0.10 "
                                       "enhances what is there. Between the two is "
                                       "worse than either: pick a recipe."),
                io.Combo.Input("tile_grid", options=TILE_GRIDS, default=DEFAULT_TILE_GRID,
                               tooltip="Columns x rows. Tile SIZE is derived from "
                                       "this and the overlap, per axis."),
                io.Int.Input("tile_overlap", default=DEFAULT_TILE_OVERLAP_PIXELS,
                             min=8, max=2048, step=8,
                             tooltip="In pixels. The overlap that actually results "
                                     "may be smaller once tiles round to whole "
                                     "latent cells."),
                io.Boolean.Input("rope_offsets", default=True,
                                 tooltip="Give each tile position ids at its true "
                                         "place on the canvas instead of letting "
                                         "every tile claim the origin."),
                io.Combo.Input("identity_lora_name", options=available_lora_names(),
                               tooltip="Krea2-identity-edit LoRA. REQUIRED for high "
                                       "quality; sampling proceeds without it."),
                io.Float.Input("identity_lora_strength", default=1.0, min=-10.0,
                               max=10.0, step=0.01),
                io.Float.Input("positive_prompt_weight", default=1.0, min=0.0,
                               max=10.0, step=0.05),
                io.Float.Input("negative_prompt_weight", default=1.0, min=0.0,
                               max=10.0, step=0.05),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, upscaled_reference_image_latent, positive, negative,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                tile_grid, tile_overlap, rope_offsets,
                identity_lora_name, identity_lora_strength,
                positive_prompt_weight, negative_prompt_weight) -> io.NodeOutput:
        model_for_this_run = apply_identity_lora(
            model, identity_lora_name, identity_lora_strength)
        if not identity_lora_name or identity_lora_name == NO_LORA_SELECTED:
            logging.info("Krea2 Tiled Diffusion: no identity LoRA selected; "
                         "quality will be lower than the reference recipes")

        positive = scale_conditioning(positive, positive_prompt_weight)
        negative = scale_conditioning(negative, negative_prompt_weight)

        latent_samples = upscaled_reference_image_latent["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(model_for_this_run,
                                                                latent_samples)
        latent_height, latent_width = latent_samples.shape[-2], latent_samples.shape[-1]

        plan = plan_latent_tiles(latent_width=latent_width,
                                 latent_height=latent_height,
                                 tile_grid=tile_grid,
                                 tile_overlap_pixels=tile_overlap)
        logging.info("Krea2 Tiled Diffusion: latent tile %dx%d, overlap %d, "
                     "%d tiles per step (%d x %d) over a %dx%d latent",
                     plan.tile_width, plan.tile_height, plan.overlap,
                     len(plan.tiles), plan.columns, plan.rows,
                     latent_width, latent_height)

        rope_offset_holder = TileRopeOffsetHolder() if rope_offsets else None
        model_for_this_run = model_for_this_run.clone()
        model_for_this_run.set_model_unet_function_wrapper(
            build_tiled_denoise_wrapper(plan, rope_offset_holder))
        if rope_offset_holder is not None:
            model_for_this_run.set_model_patch(
                build_post_input_rope_offset_patch(rope_offset_holder), "post_input")

        model_sampling = model_for_this_run.get_model_object("model_sampling")
        if scheduler == DISCRETE_SCHEDULER_NAME:
            sigmas = build_discrete_sigmas(model_sampling, steps, denoise)
        else:
            sampler_helper = comfy.samplers.KSampler(
                model_for_this_run, steps=steps, device=latent_samples.device,
                sampler=sampler_name, scheduler=scheduler, denoise=denoise)
            sigmas = sampler_helper.sigmas

        noise = comfy.sample.prepare_noise(
            latent_samples, seed, upscaled_reference_image_latent.get("batch_index"))
        sampler = comfy.samplers.sampler_object(sampler_name)
        callback = latent_preview.prepare_callback(model_for_this_run, len(sigmas) - 1)

        samples = comfy.samplers.sample(
            model_for_this_run, noise, positive, negative, cfg,
            latent_samples.device, sampler, sigmas.to(latent_samples.device),
            model_options=model_for_this_run.model_options,
            latent_image=latent_samples,
            denoise_mask=upscaled_reference_image_latent.get("noise_mask"),
            callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=seed)

        output = upscaled_reference_image_latent.copy()
        output["samples"] = samples
        return io.NodeOutput(output)
