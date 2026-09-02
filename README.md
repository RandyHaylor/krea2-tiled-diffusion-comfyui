# krea2-tiled-diffusion-comfyui

![The two nodes](docs/images/krea2_tiled_diffusion_the_two_nodes.png)

Two ComfyUI nodes for Krea 2. One denoises a latent as overlapping tiles **fused
every step**. The other gives every tile **its own conditioning**, built from the
region that tile will actually draw.

## What makes this different

**The tiles are fused inside every sampling step**, under a raised cosine weight,
before the step returns. Most tile-based upscaling denoises each tile to
completion and blends the finished tiles afterwards. Because neighbouring tiles
here are recombined at each step, they cannot drift apart and invent conflicting
content — a pixel blend of independently finished tiles cannot repair that
divergence after the fact.

**Every tile gets its own vision tokens.** The text encoder's vision tower reads
the crop that tile covers, not the whole picture. Conditioning a tile on the
entire composition is what tells each tile to draw the entire composition, which
is how tiled passes grow a second head or a spare pair of legs. Vision cannot be
attached to an already-encoded conditioning, so each tile's crop and text go
through the encoder together, in one pass.

**Per-tile prompt text.** Alongside the global prompt, each tile has its own
field. The global prompt should hold only what is true of *every* tile — style,
medium, palette — because it is asked for in every tile. Subject matter that
appears in one region belongs in that region's field.

The approach was developed and measured in a
[llama.cpp-style local Krea 2 runtime](https://github.com/RandyHaylor/stable-diffusion-krea-2-convrot-tinyserver),
where a per-finished-tile implementation was built, compared against this one,
lost twice, and was deleted.

## Example

A 2x2 pass at 2048², four steps, with per-tile vision. Left is the input, right
is the result.

| before | after |
|--------|-------|
| ![before](docs/images/puppy_before_tiled_diffusion.png) | ![after](docs/images/puppy_after_tiled_diffusion.png) |

## The workflow

![Example workflow](docs/images/krea2_tiled_diffusion_example_workflow.png)

[`workflows/krea2 tiled diffusion.json`](workflows/krea2%20tiled%20diffusion.json)
is ready to open. The shape that matters:

```
Load Image ─┬─> Upscale Image ─┬─> VAE Encode ──> Krea2 Tiled Diffusion (latent)
            │                  │
            └──────────────────┴─> Krea2-Qwen3 Encoder (reference_image)
                                        └─> tile_vision ──> Krea2 Tiled Diffusion
Empty CLIPTextEncode ────────────────────────────────────> Krea2 Tiled Diffusion (negative)
```

Three rules the wiring has to follow:

1. **`reference_image` and the latent must be the same pixels.** The encoder
   crops that image to find each tile; the sampler tiles the latent. Feed both
   from the same node, after any resizing.
2. **`positive` is left unwired** when `tile_vision` is connected. Each tile
   brings its own conditioning, so a separate positive has nothing to do. It
   exists for workflows that do not use our encoder at all.
3. **`negative` is required**, normally an empty `CLIPTextEncode`.

The sampler takes `tile_grid` and `tile_overlap` **from the bundle** when one is
wired — its own widgets are ignored, because the crops were already made against
the encoder's geometry. Set them on the encoder.

The nodes never resize, crop, or touch a VAE. They expect a latent already at the
target resolution; ComfyUI's own nodes do the upscaling and encoding.

## Two recipes

The settings come from real generations, not from taste. There are **two** that
work, and they are different jobs rather than two points on one scale — the
middle ground between them is worse than either end.

|                | **High denoise** (default) | **Low denoise** |
|----------------|---------------------------|-----------------|
| what it does   | builds a new high-res image | enhances what is already there |
| grid           | 2x2                       | 3x3             |
| overlap        | 256px                     | 512px           |
| steps          | 8                         | 4               |
| denoise        | 0.75                      | 0.10            |

Shared by both: **euler** + **simple**, cfg **1**, flow shift **1.15**, RoPE
offsets per tile **on**. Either wants a turbo model or a turbo LoRA.

Do not split the difference on denoise. Pick a recipe.

Those overlap figures predate per-tile vision. With each tile conditioned on its
own region, smaller overlaps hold together where they previously fell apart —
`DESIGN.md` records what has and has not been measured.

## Per-tile prompts

The prompt fields are a fixed 3x3 grid, so their row and column labels stay
meaningful whatever tile grid is chosen. A 2x2 run reads fields (1,1), (1,2),
(2,1) and (2,2); the rest are ignored.

Each tile is encoded with `global prompt, tile prompt` together with its own crop.

## Requirements

- A ComfyUI build with native Krea 2 support (`comfy/ldm/krea2/`)
- A turbo model or turbo LoRA, supplied by you
- The identity edit LoRA loads in-node

## License

MIT. See [LICENSE](LICENSE).
