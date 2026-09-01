# krea2-tiled-diffusion-comfyui

A ComfyUI custom node that denoises a latent as overlapping tiles **fused every
step**, tuned for Krea 2.

**Status: design only. No node is implemented yet.** See [DESIGN.md](DESIGN.md).

## What makes this different

Most tile-based upscaling denoises each tile to completion and then blends the
finished tiles together. This fuses the tiles **inside every sampling step**,
under a raised cosine weight, before the step returns.

That difference is the whole point: because neighbouring tiles are recombined at
each step, they cannot drift apart and invent conflicting content. A pixel blend
of independently finished tiles cannot repair that divergence after the fact.

The approach was developed and measured in a
[llama.cpp-style local Krea 2 runtime](https://github.com/RandyHaylor/stable-diffusion-krea-2-convrot-tinyserver),
where a per-finished-tile implementation was built, compared against this one,
lost twice, and was deleted.

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
| vision         | on                        | on              |

Shared by both: **euler** + **discrete**, cfg **1**, flow shift **1.15**, noise
multiplier **1**, RoPE offsets per tile **on**. Either wants a turbo model or a
turbo LoRA.

The extra tiles and wider overlap earn their cost in the low denoise recipe,
where the pass is not allowed to redraw much and tile coverage is doing the work.
At 0.75 denoise a 2x2 grid at 256px overlap reaches an excellent result for 4
model evaluations per step instead of 9.

Do not split the difference on denoise. Pick a recipe.

`DESIGN.md` records the individual reference generations behind both.

## Requirements

- A ComfyUI build with native Krea 2 support (`comfy/ldm/krea2/`)
- A turbo model or turbo LoRA, supplied by you
- The identity edit LoRA loads in-node

## License

MIT. See [LICENSE](LICENSE).
