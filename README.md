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

## Defaults

The defaults come from real generations, not from taste. `DESIGN.md` records two
reference results, what they share, and where they disagree.

Notable ones: a 2x2 grid with 512px overlap, RoPE offsets per tile **on**, euler
+ discrete, cfg 1, flow shift 1.15.

## Requirements

- A ComfyUI build with native Krea 2 support (`comfy/ldm/krea2/`)
- A turbo model or turbo LoRA, supplied by you
- The identity edit LoRA loads in-node

## License

MIT. See [LICENSE](LICENSE).
