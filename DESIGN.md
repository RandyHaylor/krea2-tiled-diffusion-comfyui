# Krea2 Tiled Diffusion — ComfyUI custom node

Design and requirements. Nothing here is built yet.

Target install: `/media/aikenyon/NVME_2/ubuntu_comfy/ComfyUI`

=== Status Of What Is Written Here ===

Three kinds of claim appear below, marked so they are never confused

- VERIFIED — read out of `stable-diffusion-krea-2-convrot-tinyserver` in this
  repo, or observed in its server log. These are the mechanism we are porting
- OBSERVED — the user's own judgement from real generations. Not metrics
- UNVERIFIED — about ComfyUI's internals. NOT yet read. Every one of these is a
  reading task, not a decision. Do not build against them

The point of the split: the mechanism is known precisely, the host is not.

=== What This Node Is ===

Node display name: **Krea2 Tiled Diffusion**

One sampler node that denoises a latent as overlapping tiles fused every step,
for Krea2, at the settings that have produced the best result so far.

It is NOT a port of the web app. Prompt composition, WD14 tagging, per-stage
routing, the paused two-stage exchange and the metadata writer all stay behind.
ComfyUI has nodes for text and can pass conditioning in directly.

What must survive the port is the tiling mechanism and nothing less.

=== The Mechanism That Must Survive ===

This section is the reason the node exists. A port that loses any of it has
produced a different algorithm that happens to have the same name.

Fusion is PER STEP, not per finished tile
    - VERIFIED: the latent is split, every tile is denoised by the real model for
      that one step, and the results are fused under a raised cosine weight
      before the step returns
    - Anything that denoises a tile to completion and then blends is a different
      algorithm with a different failure mode. That approach was built here,
      lost its comparison twice, and was deleted
    - Because fusion happens every step, neighbours cannot diverge. That is the
      whole reason this beats pixel tiling

Overlap is REQUIRED and an exact doubling does not supply it
    - VERIFIED: two 1024px tiles reach 2048px only by abutting, leaving the join
      nothing to blend across. An exact 2x needs THREE tiles per axis

Tile starts are SPREAD, not marched at the stride
    - VERIFIED: `tile_start_positions_covering_length` computes the count from
      the stride, then spreads that many tiles evenly across the length, so the
      overlap is uniform rather than leaving a short last tile

The grid is chosen, the tile size is DERIVED, per axis
    - VERIFIED: the user picks columns x rows. Size follows:
      `size = (length + overlap * (count - 1)) / count`, rounded up to a whole
      latent cell, per axis independently
    - A grid over a non-square canvas gives non-square tiles. Forcing square
      tiles onto a portrait canvas spends most of their area on redundant overlap
    - Requested overlap is clamped to `length // count`, so a large request comes
      back smaller. The overlap that RESULTS is what should be reported

RoPE offsets ON
    - Each tile gets position ids at its true canvas place instead of every tile
      claiming the origin
    - OBSERVED: produces the better image at 3x3 / 512px overlap / ~0.1 denoise
    - An earlier measurement at 2x2 / 8 real steps put them slightly behind. That
      is not the regime this is used in. Both readings recorded, neither is a
      metric

Judge output by CROPS, not seam metrics
    - VERIFIED as a project rule: a ghost is a smeared low-contrast region, and
      blur reads as a shallow gradient, so a seam-steepness metric rates a
      ghosted result BETTER. Any acceptance test built on seam metrics is lying

=== Node Interface ===

Inputs

    upscaled reference image latent   LATENT   required
    model                             MODEL    required
    positive                          CONDITIONING required. Composed elsewhere
    negative                          CONDITIONING required. Composed elsewhere
    vlm_weights                       optional. Gathered by other nodes
    identity_lora                     in-node. See below

The node does NO resizing
    It expects a latent that is ALREADY at the target resolution, which is what
    the input name says. There are plenty of ways to upscale an image or a latent
    in ComfyUI and no reason to add another.
    This is a deliberate narrowing against the runtime it came from, where the
    hires pass upscaled the latent itself (104x152 -> 156x228). Here that is the
    workflow's job.

Scheduler: discrete, ported, and the default

    CORRECTION to an earlier assumption in this project: the discrete scheduler is
    NOT a custom addition of ours. VERIFIED - the vendor patch contains zero
    occurrences of the string "discrete"; `DISCRETE_SCHEDULER` is upstream
    stable-diffusion.cpp, at include/stable-diffusion.h:66 and
    src/stable-diffusion.cpp:3641.

    It still has to be ported, for a different reason: VERIFIED that ComfyUI has
    no discrete scheduler. Its full set at comfy/samplers.py:1365 is simple,
    sgm_uniform, karras, exponential, ddim_uniform, beta, normal,
    linear_quadratic, kl_optimal.

    VERIFIED the port is small. src/runtime/denoiser.hpp:32-50 walks t linearly
    from TIMESTEPS - 1 down to 0 across n steps and maps each through t_to_sigma.
    No tables, no per-model-version branching, no extra sample args, unlike the
    AYS and GITS schedulers beside it.

    DECISION: implement it in-node and make it the first choice and default. Only
    register it as a normally available ComfyUI scheduler if that costs nothing
    extra beyond making it faithful in-node.
    UNVERIFIED and the thing that decides it: whether ComfyUI's `model_sampling`
    exposes a `t_to_sigma` equivalent that maps cleanly. If it does not, faithful
    means more code, and the scheduler stays in-node only.

Prompt weight dial

    positive_prompt_weight   FLOAT, default 1.0
    negative_prompt_weight   FLOAT, default 1.0

    VERIFIED that weighting encoded conditioning is possible and already shipped:
    `ConditioningMultiply` (nodes.py:174) scales the tensor itself,
    `t[0] * multiplier`, and the pooled_output with it.

    UNDECIDED, because ComfyUI has TWO different mechanisms and they are not the
    same operation:
      a. scale the embedding    ConditioningMultiply, nodes.py:162
      b. set a strength key     ConditioningSetAreaStrength, nodes.py:229, via
                                node_helpers.conditioning_set_values
    (a) changes the vector the model attends to. (b) hands the sampler a number
    to apply. Pick by testing, not by which reads better.

    NOTE this duplicates an existing built-in node. Worth having in-node anyway so
    one node carries a reproducible preset, but it is a convenience, not a
    capability we are adding.

Widgets, with the defaults below

    tile_grid, tile_overlap, rope_offsets,
    steps, cfg, sampler, scheduler, flow_shift, denoise, noise_multiplier,
    pag_enabled, pag_scale, pag_layers, pag_start, pag_end,
    vae_tile_size,
    seed + control_after_generate

Seed
    The standard KSampler arrangement: a seed widget with control_after_generate
    beside it. VERIFIED that `common_ksampler` (nodes.py:1571) already takes
    `seed` in exactly this shape, so this is reuse, not reimplementation.

Krea2-identity-edit LoRA
    identity_lora_name      dropdown over the loras folder
    identity_lora_strength  FLOAT, default 1.0

    FAIL SAFE: if none is selected the node still runs. It does NOT raise.
    The node displays standing text that the LoRA is required for high quality,
    so an unset dropdown is visibly a choice rather than a silent downgrade.

    The turbo LoRA and the base model stay the user's responsibility. The
    defaults assume a turbo setup.

Vision (VLM) tokens, generated in-node
    vision_model            model selector
    vision_enabled          BOOLEAN, default ON
    vision_weight           FLOAT, default 1.0

    FAIL SILENTLY: with vision_enabled on but no model provided, the node
    proceeds without vision. No error, no warning dialog.
    This is the one place complexity is pulled IN rather than pushed out to the
    workflow, because gathering vision tokens for the tiles is specific enough
    to this node that making every user wire it up externally is worse.
    The `vlm_weights` optional input stays, for a workflow that would rather
    supply them itself. UNDECIDED: precedence when both are present.

Outputs

    LATENT

=== Two Reference Results, And What They Agree On ===

Both are OBSERVED standouts on the same prompt, "a real life photo of judy hopps
from the disney movie, zootopia", same model and LoRAs, both refining an existing
image with NO first stage sampled.

They disagree on almost every axis. That disagreement is the most useful thing we
have, because what they SHARE is what is actually carrying the result.

                        REFERENCE A     REFERENCE B     REFERENCE C (newest,
                        (1664x2432)     (1248x1824)      "excellent", 1248x1824)
    grid                3x3             2x2             2x2
    overlap             512 (504 real)  512             256
    rope offsets        on              on              on
    steps (executed)    4               8               8
    denoise             0.1             0.75            0.75
    PAG                 on, s1, layer7  OFF             OFF
    hires vision input  none            img2img_source  img2img_source

    SHARED BY ALL THREE — treat as the load-bearing defaults
        sampler                 euler
        scheduler               discrete
        cfg                     1
        flow_shift              1.15
        noise_multiplier        1
        rope_offsets            on
        vae_tile_size           32
        source sizing           1024px limit, 64px increment
        no first stage sampled
        model  krea2RawBaseInt8Row_v10        cb81322759
        loras  krea2_identity_edit_v1_2_r64   f794b47142
               krea2_raw_to_turbo_r256        71a50a117b

    C is B with the overlap halved, and OBSERVED as excellent. 256 is therefore
    sufficient at this denoise and grid, and it is much cheaper: a smaller overlap
    means smaller derived tiles for the same grid. Overlap 512 is no longer shared
    by all three, so it drops out of the load-bearing set and becomes a default
    chosen on cost.

Node defaults: the shared set, plus Reference C for the axes where they differ,
since it is the newest and cheapest of the strong results.

    tile_grid               2x2
    tile_overlap            256
    steps                   8
    denoise                 0.75
    pag_enabled             off

Why 2x2 is the default even though 3x3 improves quality
    OBSERVED: 3x3 DOES improve quality. It is not doing nothing and it is not
    merely compensating for a defect.
    OBSERVED, and separate from the grid: Krea 2 appears to shine at img2img
    around 0.75 denoise. The user has seen this across other work, not only here.
    Cause unknown, possibly training, possibly coincidence. NOT investigated.
    2x2 is the default because 0.75 denoise reaches a strong result at 4 model
    evaluations per step instead of 9. That is a cost choice, not a claim that
    3x3 is worthless. Raise the grid when the quality is worth the time.

The two references are two different jobs, not two points on a scale

    A is "slightly enhance what is already there". B is "build a new super HD
    image". OBSERVED: the middle ground between them is hot garbage — worse than
    either end, not a blend of them.

    This is documented as guidance, NOT enforced. No preset switch, no snapping,
    no warning when a value lands between the two. Every setting is a plain field
    and the user is trusted to drive it. The reason to write it down is that a
    reasonable person would otherwise assume denoise interpolates sensibly
    between these references, and it does not.

    denoise: FLOAT, clamp 0.0-1.0, step 0.01. The clamp is there because the
    range is genuinely meaningless outside it, not to protect anyone.

=== Open Questions In The Reference Settings ===

Each one changes a default, so none should be guessed.

1. OVERLAP: 512 or 504?
   Settled. VERIFIED these are the same thing from two sides: 512 is REQUESTED,
   504 is what survives rounding the derived tile to a whole latent cell — which
   is why only Reference A shows it, at a 3x3 grid. The node takes 512 as input
   and REPORTS the resulting overlap. Do not hardcode 504.

2. VISION: DOES IT MATTER? The two references answer differently
   Reference A: "Img2img vision on source: stage 1", "Hires vision input: none",
   on a job with no first stage. Traced, NOT tested: the vision tokens went into
   the FIRST STAGE's conditioning, and `re_encode_hires_conditioning` clears
   `vlm_images` unconditionally before re-encoding, adding new ones only when the
   hires stage supplied some. "none" means it did not. That first stage sampled
   zero steps — so vision plausibly contributed NOTHING to Reference A.
   Reference B routes vision to the hires stage properly, so it did contribute.
   This matters for whether `vlm_weights` belongs on the default path. If A is
   as good as B without vision, vision is optional; if B is better because of it,
   it is not. Confirm on hardware.

3. PAG: on in A, OFF in B, at cfg 1 in both
   B being the stronger result with PAG off suggests PAG is not what was carrying
   it. Default off, per B. Note ComfyUI's PAG is a separate node here
   (`comfyui-anima-safe-pag` installed) and may not compose the same way, so
   defaulting it off also removes an integration unknown.

4. PROMPT WEIGHT MECHANISM
   Which of ConditioningMultiply's tensor scaling or the strength key the dial
   should drive. See the Node Interface section. Decide by testing.

=== Reference Implementation: shiimizu/ComfyUI-TiledDiffusion ===

Read for MECHANISM, to avoid reinventing wheels. Findings below are VERIFIED by
reading its source and by checking the API against the local ComfyUI install.

LICENSE WARNING — READ BEFORE COPYING ANYTHING
    Its README states: "The implementation of MultiDiffusion, Mixture of
    Diffusers, and Tiled VAE code is currently under Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License since it was
    borrowed from the wonderful SD-WebUI extension. Anything else GPLv3."
    CC BY-NC-SA 4.0 is non-commercial AND share-alike. GPLv3 is copyleft. NEITHER
    is compatible with this repo being MIT.
    We may read it to learn which ComfyUI APIs exist and how they behave. We may
    NOT copy its code, or paraphrase it closely enough to be a derivative work.
    The tiling algorithm we are porting is our own, from the krea runtime.

The hook, which closes the biggest open question
    `model.set_model_unet_function_wrapper(self.impl)`
    VERIFIED present in this install at comfy/model_patcher.py:655.
    That is the supported way to wrap the per-step denoise call.

Their fusion is per step too, which confirms the paradigm is expressible
    They accumulate into a buffer and divide by accumulated weights:
        self.x_buffer[bbox.slicer] += x_tile_out[...]
        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights,
                            self.x_buffer)
    Same shape of idea as the raised cosine fusion we are porting. Reassuring
    that ComfyUI does not fight this.

Conditioning is sliced per tile and repeated to the tile batch size
    Worth knowing, because our per-tile RoPE offsets have to travel the same path

tile_batch_size is an idea worth stealing (the idea, not the code)
    They batch several tiles into one model call for speed. Our design currently
    runs tiles one at a time. CONSIDER adding, once correctness is established

Where we deliberately differ
    They take tile_width and tile_height. We take a GRID and derive the sizes per
    axis. That difference is intentional and documented above under the mechanism

ARCHITECTURE: they are a MODEL PATCHER, we are a SAMPLER
    Their node patches a model and hands it on, so a normal KSampler does the
    sampling. Ours cannot be only that, because the whole point is a node that
    carries the Krea2 defaults, the discrete scheduler and the identity LoRA.
    RESOLUTION, and it avoids reinventing sampling:
      1. apply the identity LoRA to the model
      2. generate vision tokens, if enabled and a model is set
      3. install the tiling via set_model_unet_function_wrapper
      4. call `common_ksampler` (nodes.py:1571), which already takes
         model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
         latent, denoise
    We write the tiling and the scheduler. ComfyUI keeps doing the sampling.

=== ComfyUI Integration — REMAINING UNVERIFIED ===

Everything in this section is a reading task. I have not read ComfyUI's sampler
internals, and anything I asserted about them would be invented.

Known from a directory listing only

    comfy/samplers.py, comfy/sample.py, comfy/model_patcher.py,
    comfy/patcher_extension.py, comfy/extra_samplers/  exist
    comfy/ldm/krea2/model.py exists, and krea2 appears in supported_models.py,
    model_detection.py, model_base.py, sd.py and lora.py
        -> Krea2 is natively supported. This is a TILING node, not a model port

Must be answered by reading source, before any design is fixed

    a. What hook lets a node wrap the per-step denoise call? The mechanism needs
       to intercept one step, run the model N times on N tile views, fuse, and
       return one result. `patcher_extension.py` and `model_patcher.py` are the
       first places to look
    b. How are position ids / RoPE supplied to the Krea2 model, and can they be
       offset per call? RoPE offsets are a default here, so if this is not
       reachable the default is not deliverable
    c. How does a node express "sample only the tail of the schedule" for a low
       denoise? The executed-steps-vs-scheduled-steps distinction bit this
       project hard: 8 steps at denoise 0.4 spent THREE evaluations, and every
       measurement taken before that was understood was mislabelled. Find out
       what ComfyUI's denoise actually does before trusting a step count
    d. Does the installed `comfyui-anima-asampler` or `comfyui-anima-safe-pag`
       already implement any of this? Read what they DO, not what they claim
    e. Is there an existing tiled-diffusion node in the wild worth reading? Read
       for mechanism only: specifically whether it fuses per step or per tile

Rule for this section, from the user: do not copy other code without verifying
the mechanism it uses. A node that looks like tiled diffusion but fuses finished
tiles is the failure mode this whole design exists to avoid.

=== Testing ===

The Python here is testable the same way this repo's is: pure functions, plain
`python3 test_*.py`, no pytest.

Portable directly, already pure and tested here
    - tile size derivation per axis, with the overlap clamp
    - tile start positions, evenly spread
    - the resulting-overlap report
    - grid parsing

Needs a real model, so not unit-testable
    - the per-step fusion itself
    - RoPE offsets per tile

Acceptance is visual, on crops, at the reference settings. Not a seam metric.

=== Explicitly Out Of Scope ===

    prompt composition and hires prompt overrides
    WD14 tagging and per-stage tag routing
    the paused two-stage exchange
    PNG metadata writing
    the queue, the web UI, the tunnel
    pixel-space tiling of any kind
