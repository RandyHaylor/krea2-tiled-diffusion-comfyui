#!/usr/bin/env python3
"""Every node's schema must survive the trip to the browser.

ComfyUI's /object_info route builds each node's info with GET_NODE_INFO_V1() and
hands the lot to json.dumps (server.py:751-754, :811). Anything in a schema that
json cannot encode - a function passed where a list of options belongs, say -
takes down the WHOLE endpoint, so every node in the install disappears and the UI
renders as an empty grid.

Run with ComfyUI's interpreter:
    ComfyUI/.venv/bin/python tests/test_krea2_node_schemas_serialize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/media/aikenyon/NVME_2/ubuntu_comfy/ComfyUI")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import comfy.samplers  # noqa: E402

from krea2_image_and_text_encoder_node import Krea2Qwen3ImageAndTextEncoder  # noqa: E402
from krea2_tiled_diffusion_node import Krea2TiledDiffusion  # noqa: E402

failures: list[str] = []

NODE_CLASSES = [Krea2TiledDiffusion, Krea2Qwen3ImageAndTextEncoder]


def check(description: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}: {description}{(' ' + detail) if detail else ''}")
    if not passed:
        failures.append(description)


def main() -> int:
    for node_class in NODE_CLASSES:
        name = node_class.__name__
        try:
            node_info = node_class.GET_NODE_INFO_V1()
            built = True
            build_error = ""
        except Exception as exc:  # noqa: BLE001 - reporting whatever the route would hit
            node_info, built, build_error = None, False, f"{type(exc).__name__}: {exc}"
        check(f"{name} builds the info /object_info serves", built, build_error)

        if not built:
            continue

        try:
            json.dumps(node_info)
            serialised = True
            serialise_error = ""
        except TypeError as exc:
            serialised = False
            serialise_error = str(exc)
        check(f"{name} serialises to JSON, so /object_info does not fail",
              serialised, serialise_error)

    # ComfyUI's `simple` was MEASURED bit-exact to Krea 2's own schedule -
    # sigma(linspace(1, 0, steps + 1)) at shift 1.15, from krea-2/sampling.py -
    # against the real ModelSamplingFlux. A hand-rolled scheduler here would only
    # be a chance to get that wrong again, which it already was.
    tiling_schema = Krea2TiledDiffusion.define_schema()
    scheduler_input = next(node_input for node_input in tiling_schema.inputs
                           if getattr(node_input, "id", "") == "scheduler")
    check("the scheduler defaults to simple, which IS Krea 2's own schedule",
          scheduler_input.default == "simple", f"got {scheduler_input.default}")
    check("no hand-rolled scheduler is offered alongside ComfyUI's",
          set(scheduler_input.options) == set(comfy.samplers.SCHEDULER_NAMES),
          f"extras: {set(scheduler_input.options) - set(comfy.samplers.SCHEDULER_NAMES)}")

    # A dropdown's options must be concrete values by the time the schema is
    # built. Passing the function that produces them is the specific mistake that
    # broke the endpoint once.
    for node_class in NODE_CLASSES:
        schema = node_class.define_schema()
        callable_options = [getattr(node_input, "id", "?")
                            for node_input in schema.inputs
                            if callable(getattr(node_input, "options", None))]
        check(f"{node_class.__name__} has no dropdown whose options are a function",
              not callable_options, f"got {callable_options}")

    if failures:
        print(f"\n{len(failures)} failing checks:")
        for failed in failures:
            print(f"  - {failed}")
        return 1
    print("\nall krea2 node schema serialisation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
