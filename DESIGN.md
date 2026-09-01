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

    DECISION: implement it in-node and make it the first choice and default.
    SETTLED, and it costs almost nothing — see the Structural Template section.
    SIGMAS in ComfyUI is just a torch.FloatTensor, and every ModelSampling class
    exposes `sigma(timestep)`, which is the t_to_sigma this needs. We build the
    tensor ourselves and hand it to a sampler that takes explicit sigmas, so no
    registration in SCHEDULER_HANDLERS is required for the node to use it.
    Registering it globally as a named scheduler remains optional and separate.

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

                        REF A           REF B           REF C           REF D
                        (1664x2432)     (1248x1824)     (1248x1824)     (1664x2432)
    grid                3x3             2x2             2x2             3x3
    overlap             512 (504 real)  512             256             512 (504 real)
    rope offsets        on              on              on              on
    steps (executed)    4               8               8               4
    denoise             0.1             0.75            0.75            0.1
    hires vision input  none            img2img_source  img2img_source  img2img_source

    D is A with the vision actually reaching the hires pass, and with a turbo
    MODEL in place of the turbo LoRA (DasiwaKrea2TurboRaw_cutedisasterV2Turbo,
    identity edit LoRA only). It is the reference recipe for LOW DENOISE upscale.

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

Why 2x2 is the default even though 3x3 improves quality
    OBSERVED: 3x3 DOES improve quality. It is not doing nothing and it is not
    merely compensating for a defect.
    OBSERVED, and separate from the grid: Krea 2 appears to shine at img2img
    around 0.75 denoise. The user has seen this across other work, not only here.
    Cause unknown, possibly training, possibly coincidence. NOT investigated.
    2x2 is the default because 0.75 denoise reaches a strong result at 4 model
    evaluations per step instead of 9. That is a cost choice, not a claim that
    3x3 is worthless. Raise the grid when the quality is worth the time.

TWO RECIPES, not two points on a scale

    OBSERVED: the middle ground between them is hot garbage — worse than either
    end, not a blend of them. They are different jobs.

    HIGH DENOISE — "build a new super HD image"   [Reference C, the DEFAULTS]
        grid            2x2
        overlap         256
        steps           8
        denoise         0.75
        vision          on

    LOW DENOISE — "slightly enhance what is already there"   [Reference D]
        grid            3x3
        overlap         512
        steps           4
        denoise         0.1
        vision          on
        More tiles and wider overlap earn their cost here, where the pass is not
        allowed to redraw much and coverage is doing the work.

    Both share euler + discrete, cfg 1, flow shift 1.15, noise multiplier 1,
    RoPE offsets on. Either wants a turbo model or a turbo LoRA.

    Documented as guidance, NOT enforced. No preset switch, no snapping, no
    warning when a value lands between the two. Every setting is a plain field and
    the user is trusted to drive it. It is written down because a reasonable
    person would otherwise assume denoise interpolates sensibly between these, and
    it does not.

    SURFACE BOTH IN THE UI: the README carries both recipes, and the node should
    make the low denoise recipe discoverable without leaving ComfyUI — a tooltip
    or info box on the denoise widget. UNVERIFIED: what ComfyUI supports for
    per-widget tooltips or an info affordance on a node.

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

3. PAG is OUT OF SCOPE, and was never in play
   VERIFIED: the hires call site passes a default-constructed sd_pag_params_t{}
   (stable-diffusion.cpp:6325), while the main pass passes the real
   sample_params.pag (:6135). PAG never reached the hires pass.
   So the "PAG: on" in references A and D is doubly inert - the hires pass got an
   empty struct, and their first stage sampled zero steps anyway. The node has no
   PAG widgets. ComfyUI has PAG nodes of its own for anyone who wants it.

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

NOT OUR STRUCTURAL TEMPLATE: it is a MODEL PATCHER
    Their node patches a model and hands it on, so a normal KSampler does the
    sampling. That cannot give us a full discrete scheduler, our tiling order, or
    the Krea2 defaults carried in one node. Useful for the hook and the per-step
    fusion evidence above; not for how the node is shaped.

=== Structural Template: ComfyUI_UltimateSDUpscale ===

A node that OWNS its sampling is the shape we need, and USDU is one. Read for
NODE STRUCTURE only.

DO NOT TAKE ITS TILING. The Ultimate SD Upscale paradigm is pixel tiling with
exact cells, padding as context and mask blur. It was implemented against this
project's own tiled diffusion, in spike_a_ultimate_sd_upscale.py, and it LOST.
We are reading how its node is wired, not what it does to an image.

LICENSE: GPL-3.0. Copyleft, not compatible with this repo being MIT. Read only.

Its "No Upscale" variant validates our input decision
    It "assumes that the input image is already upscaled ... useful if you already
    have an upscaled image or just want to do the tiled sampling". That is exactly
    our `upscaled reference image latent`.

VERIFIED, the three ways a node can own sampling in ComfyUI
    from nodes import common_ksampler, VAEEncode, VAEDecode, VAEDecodeTiled
    from comfy_extras.nodes_custom_sampler import SamplerCustom
    import comfy.sample

    a. common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                       positive, negative, latent, denoise=denoise)
       nodes.py:1571. Widget-driven, simplest, uses ComfyUI's scheduler NAMES
    b. SamplerCustom, taking an explicit SAMPLER object AND explicit SIGMAS
    c. guider.sample(noise, latent_image, sampler, sigmas, denoise_mask=...,
                     callback=..., disable_pbar=..., seed=...)
       comfy/samplers.py:1276, with a module-level
       comfy.samplers.sample(model, noise, positive, negative, cfg, device,
                             sampler, sigmas, ...) at samplers.py:1349

THE DISCRETE SCHEDULER QUESTION IS NOW SETTLED, and more cheaply than expected
    We do NOT need to register a scheduler in ComfyUI's SCHEDULER_HANDLERS at all.
    Paths (b) and (c) take SIGMAS directly, and VERIFIED that SIGMAS is nothing
    but a torch.FloatTensor: ManualSigmas
    (comfy_extras/nodes_custom_sampler.py:1125-1144) parses a string of numbers
    into `torch.FloatTensor(sigmas)` and outputs it as io.Sigmas.
    VERIFIED the t_to_sigma equivalent exists: every ModelSampling class in
    comfy/model_sampling.py has `sigma(timestep)` — ModelSamplingDiscrete at :203,
    ModelSamplingDiscreteFlow at :318, and so on.
    So the port is: walk t linearly from TIMESTEPS-1 to 0 over n steps, map each
    through model_sampling.sigma(t), build a FloatTensor. That is the upstream
    algorithm at denoiser.hpp:32-50 with a different name for t_to_sigma.

ARCHITECTURE, resolved
      1. apply the identity LoRA to the model
      2. generate vision tokens, if enabled and a model is set
      3. install the tiling via set_model_unet_function_wrapper
      4. build our own discrete SIGMAS from model_sampling.sigma
      5. sample by path (b) or (c), which accept those sigmas directly
    We write the tiling and the sigma schedule. ComfyUI does the sampling.
    Path (a) stays the fallback for the stock schedulers, since it is the only one
    that takes a scheduler by NAME.
    UNDECIDED: (b) versus (c). (c) is lower level and hands back the guider;
    (b) is less code. With PAG out of scope there is no known reason to need the
    guider, so start with (b) and drop to (c) only if something demands it.

=== ComfyUI Integration — ANSWERED BY READING SOURCE ===

Every item here was an open reading task. All are now closed except vision, which
turned out to be a design blocker rather than a fact to look up.

a. THE PER-STEP HOOK. CLOSED
   `model.set_model_unet_function_wrapper(fn)` -> model_patcher.py:655, which just
   stores model_options["model_function_wrapper"].
   It is invoked at comfy/samplers.py:333 as
       output = model_options['model_function_wrapper'](
           model.apply_model,
           {"input": input_x, "timestep": timestep_, "c": c,
            "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
   So the wrapper receives (apply_model, kwargs) and returns a tensor shaped like
   `input`. That is precisely the one-step interception the fusion needs.
   NOTE it is called per cond batch, and `cond_or_uncond` says which. The tiling
   must be correct for a batched input, not assume batch 1.

b. ROPE OFFSETS PER TILE. CLOSED, and reachable
   Krea2 builds position ids in `process_img` (ldm/krea2/model.py:289-293):
       img_ids[..., 1] = torch.arange(h)[:, None]
       img_ids[..., 2] = torch.arange(w)[None, :]
   From torch.arange, so a tile passed alone is labelled from the ORIGIN. That is
   exactly the "every tile claims the origin" behaviour, and it is the default.
   The offset is applied through a patch point at :346-351:
       patches = transformer_options.get("patches", {})
       if "post_input" in patches:
           for p in patches["post_input"]:
               out = p({"img":..., "txt":..., "img_ids": imgpos,
                        "txt_ids": txtpos, "transformer_options":...})
               imgpos, txtpos = out["img_ids"], out["txt_ids"]
   Registered with `set_model_patch(patch, "post_input")` (model_patcher.py:661).
   So: add the tile's (row, col) latent origin to imgpos[..., 1] and [..., 2].

c. DENOISE AND THE STEP COUNT. CLOSED — and it is the INVERSE of the krea runtime
   comfy/samplers.py:1431-1441
       new_steps = int(steps/denoise)
       sigmas = self.calculate_sigmas(new_steps)
       self.sigmas = sigmas[-(steps + 1):]
   BasicScheduler (nodes_custom_sampler.py:17-42) does the same for the SIGMAS
   path. ComfyUI builds a LONGER schedule and keeps the tail, so `steps` already
   means steps EXECUTED.
   The krea runtime did the opposite: t_enc = int(sample_steps * strength)
   truncated, and the app pre-scaled with scheduled_steps_for_executed_steps.
   DO NOT PORT THAT SCALING. Porting it would double-scale and quietly run the
   wrong number of steps, which is the same class of error it was written to fix.
   Our own discrete sigma builder MUST replicate ComfyUI's convention:
   build int(steps/denoise) sigmas, then keep the last steps+1.

d. PROMPT WEIGHT MECHANISM. CLOSED
   Use the ConditioningMultiply approach: scale the embedding tensor and the
   pooled_output (nodes.py:174-183).
   NOT the `strength` key. VERIFIED that `strength` is consumed inside
   get_area_and_mult (samplers.py:52-53), where it scales the AREA blending mask
   that composes multiple conditionings. It is a region weight, not prompt
   emphasis. The two are unrelated operations and the earlier "pick by testing"
   framing was comparing the wrong things.

e. TOOLTIPS. CLOSED, fully supported
   Per-input `tooltip=` (typed in comfy_types/node_typing.py:118, used throughout
   nodes.py and nodes_custom_sampler.py) and node-level `DESCRIPTION`, which the
   UI shows on hover. The low denoise recipe can live on the denoise tooltip.

f. FLOW SHIFT. CLOSED, and the widget is probably unnecessary
   Krea2's sampling_settings in supported_models.py:1933-1936 are already
   {"multiplier": 1.0, "shift": 1.15}. Shift 1.15 is the model's own default and
   matches every reference generation. A widget would only be needed to override
   it, which no reference does.

g. VISION / vlm_weights. NOT A LOOKUP — A DESIGN BLOCKER. See its own section.

Rule for this section, from the user: do not copy other code without verifying
the mechanism it uses. A node that looks like tiled diffusion but fuses finished
tiles is the failure mode this whole design exists to avoid.

=== BLOCKER: in-node vision generation is not implementable as specified ===

VERIFIED: Krea2's text encoder is qwen3vl_4b, a vision-language model
(supported_models.py:1953). Vision images enter through the TOKENIZER:
    Krea2Tokenizer.tokenize_with_weights(self, text, ..., images=[], ...)
        comfy/text_encoders/krea2.py:28
and the node-level pattern is clip.tokenize(prompt, images=images), as
TextEncodeQwenImageEdit does at comfy_extras/nodes_qwen.py:46.

CONSEQUENCE: vision tokens are produced when the PROMPT IS ENCODED, and arrive
already baked into CONDITIONING. By the time this node receives CONDITIONING the
vision is either in it or not. There is nothing for a `vision_model` widget or a
`vlm_weights` socket to do at sampling time.

The requested design — a vision model selector, a toggle and a weight inside this
node — cannot be built without the node also taking CLIP and the raw prompt TEXT
and doing its own encoding. That directly reverses the decision that this node
takes composed pos/neg conditioning and leaves text to the workflow.

OPTIONS, for the user to choose. NOT decided here.
  1. Drop in-node vision. Document that a Krea2 text-encode node upstream takes
     images, and that the img2img source should be given to it. Keeps the node's
     contract clean. Loses the convenience that motivated the request.
  2. Add optional CLIP + IMAGE + prompt-text inputs and encode inside the node
     when they are supplied, falling back to the passed-in CONDITIONING when they
     are not. Delivers the convenience, at the cost of the node knowing about
     text encoding after all.
  3. A separate companion node in this repo that wraps clip.tokenize(images=...)
     with the vision weight, outputting CONDITIONING. Keeps this node clean and
     still saves the user from assembling it by hand.

Until this is decided, the vision widgets and the vlm_weights socket are NOT part
of the interface, and the flow below is written without them.

=== Fully Informed Flow ===

Derived from the WORKING implementation in
stable-diffusion.cpp:3020-3172, not from ComfyUI and not from inference. Every
ComfyUI call named here was checked in the local install with its own interpreter,
ComfyUI/.venv/bin/python.

--- what the runtime actually does, per step ---------------------------------

    if width <= tile_width and height <= tile_height:
        return denoise(x)                                  # :3040, no tiling

    column_starts = tile_start_positions(width,  tile_width)
    row_starts    = tile_start_positions(height, tile_height)

    accumulated        = zeros(width * height * planes)  as float64   # :3097
    accumulated_weight = zeros(width * height)           as float64   # :3098

    for row_start in row_starts:                     # rows outer, columns inner
        for column_start in column_starts:
            tile_w = min(tile_width,  width  - column_start)   # CLIPPED at the
            tile_h = min(tile_height, height - row_start)      # canvas edge

            set_tile_rope_offset(row_start // 2, column_start // 2)
            tile_out = denoise(crop(x, column_start, row_start, tile_w, tile_h))
            set_tile_rope_offset(0, 0)                         # reset every tile

            for row in range(tile_h):
                for column in range(tile_w):
                    weight = taper(column, tile_w) * taper(row, tile_h)
                    accumulated_weight[pixel] += weight
                    accumulated[pixel, :]     += weight * tile_out[column, row, :]

    fused = where(accumulated_weight > 0, accumulated / accumulated_weight, 0)

--- the taper, exactly ------------------------------------------------------

    taper(position, tile_length):
        centred = (position + 0.5) / tile_length
        return 0.0001 + 0.5 * (1 - cos(2 * pi * centred))          # :3063-3066

    A full Hann window across the tile, SEPARABLE: the pixel weight is the
    column taper times the row taper.
    The 0.0001 floor is not decoration. Without it the outermost row and column
    of every tile contribute exactly zero, and a canvas pixel covered only by
    tile edges would divide by zero.
    The taper uses the CLIPPED tile size, so an edge tile gets a window that fits
    it rather than a truncated one.

--- rope offsets are in TOKEN units, not latent cells -----------------------

    Krea2::krea2_tiled_tile_row_offset_tokens() = row_start / 2          # :3126
    Krea2::krea2_tiled_tile_col_offset_tokens() = column_start / 2       # :3128

    Divided by two because Krea2's patch size is 2. VERIFIED the same in ComfyUI:
    ldm/krea2/model.py:234 defaults `patch=2`, and process_img derives h, w as
    `x.shape[-2] // patch`.
    An earlier draft of this flow said to add the latent origin directly. That
    would have offset every tile by twice its true position.

--- how this maps onto ComfyUI ----------------------------------------------

    The wrapper                                                  samplers.py:333
        model_options["model_function_wrapper"](apply_model,
            {"input": x, "timestep": t, "c": c, "cond_or_uncond": [...]})
        Returns a tensor shaped like `input`. Installed with
        set_model_unet_function_wrapper (model_patcher.py:655).
        Called PER COND BATCH, so `input` can hold cond and uncond stacked. The
        tiling is purely spatial, so it operates on the whole batch at once and
        does not care - but nothing may assume batch 1.

    Slice ONLY `input`. Pass `c` through unchanged.
        The C++ additionally crops active_init_latent and active_denoise_mask
        (:3119-3122) because its inner callback composites against them at full
        canvas size. ComfyUI does not composite inside apply_model - masking is
        handled by the sampler around it - so there is nothing to crop.
        The conditioning entries here are text embeddings, not spatially tied to
        the tile. `c` also carries `transformer_options` and possibly `control`.
        CONTROL IS NOT HANDLED. A control net would need its hint cropped per
        tile, and that is out of scope; note it rather than pretend.

    RoPE offsets                                     ldm/krea2/model.py:346-351
        The "post_input" patch is handed img_ids and takes back what it returns:
            img_ids[..., 1] += row_offset_tokens
            img_ids[..., 2] += column_offset_tokens
        Registered with set_model_patch(patch, "post_input")   model_patcher:661
        The patch reads the current tile's offset from a holder the tile loop
        writes before each apply_model call and clears after, mirroring the C++.
        Without the patch, process_img builds ids from torch.arange (:291-292),
        which is exactly the "every tile claims the origin" behaviour, and is
        what `rope_offsets = off` should produce.

    Sigmas                                                     samplers.py:1439
        ComfyUI means steps EXECUTED: it builds int(steps/denoise) sigmas and
        keeps the last steps+1. The krea runtime truncates instead, which is why
        the app pre-scales. DO NOT PORT THAT SCALING - it would double-scale.
        The discrete scheduler is the upstream algorithm at denoiser.hpp:32-50
        with model_sampling.sigma(t) as t_to_sigma, then the same tail slice.

    Sampling                                                   samplers.py:1349
        comfy.samplers.sample(model, noise, positive, negative, cfg, device,
                              sampler, sigmas, latent_image=..., seed=...)
        takes explicit sigmas, so no scheduler needs registering.

--- module layout -----------------------------------------------------------

    krea2_tile_planning.py        pure: grid, tile sizes, starts, taper weights
    krea2_tiled_denoise.py        the wrapper and the fusion, torch only
    krea2_rope_tile_offset.py     the post_input patch and its offset holder
    krea2_discrete_sigmas.py      the ported scheduler
    krea2_tiled_diffusion_node.py the node that assembles the above

    Everything except the node is importable and testable without ComfyUI's
    node machinery, and the torch parts run under ComfyUI/.venv/bin/python.

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
