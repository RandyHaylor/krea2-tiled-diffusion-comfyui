"""Krea 2 tiled diffusion nodes for ComfyUI."""
import os
import sys

# Matches the convention used by the other custom nodes in this install, and lets
# the modules here import each other by plain name from ComfyUI and from the tests.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from krea2_image_and_text_encoder_node import Krea2Qwen3ImageAndTextEncoder


class Krea2TiledDiffusionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Krea2Qwen3ImageAndTextEncoder,
        ]


async def comfy_entrypoint() -> Krea2TiledDiffusionExtension:
    return Krea2TiledDiffusionExtension()
