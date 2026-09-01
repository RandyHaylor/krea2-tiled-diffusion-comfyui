#!/usr/bin/env python3
"""Checks for loading the Krea2 identity edit LoRA.

Run with ComfyUI's interpreter:
    ComfyUI/.venv/bin/python tests/test_krea2_identity_lora_loading.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/media/aikenyon/NVME_2/ubuntu_comfy/ComfyUI")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

import krea2_tiled_diffusion_node as node_module  # noqa: E402
from krea2_tiled_diffusion_node import NO_LORA_SELECTED, apply_identity_lora  # noqa: E402

failures: list[str] = []


def check(description: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}: {description}{(' ' + detail) if detail else ''}")
    if not passed:
        failures.append(description)


class ModelStandingInForAPatcher:
    """Enough of a model to tell "returned untouched" from "tried to load"."""


def main() -> int:
    untouched = ModelStandingInForAPatcher()

    check("no selection returns the model untouched",
          apply_identity_lora(untouched, NO_LORA_SELECTED, 1.0) is untouched)
    check("an empty selection returns the model untouched",
          apply_identity_lora(untouched, "", 1.0) is untouched)
    check("a strength of zero skips the load entirely",
          apply_identity_lora(untouched, "anything.safetensors", 0.0) is untouched)

    check("a name that resolves to no file returns the model untouched",
          apply_identity_lora(untouched, "definitely-not-present-12345.safetensors", 1.0)
          is untouched,
          "folder_paths returns None and the node must not raise")

    # The real failure: `import comfy.sd` inside the function made `comfy` a local
    # name for the whole scope, so the earlier comfy.utils call raised
    # UnboundLocalError before any file was ever read. Drive the function to the
    # point of loading and confirm the error that surfaces is the loader's, not a
    # scoping mistake.
    class LoadWasReached(Exception):
        pass

    original_get_full_path = folder_paths.get_full_path
    original_load_torch_file = comfy.utils.load_torch_file
    folder_paths.get_full_path = lambda folder, name: "/tmp/pretend-lora.safetensors"
    comfy.utils.load_torch_file = lambda *args, **kwargs: (_ for _ in ()).throw(
        LoadWasReached())
    try:
        apply_identity_lora(untouched, "pretend-lora.safetensors", 1.0)
        outcome = "returned without loading"
    except LoadWasReached:
        outcome = "reached the loader"
    except UnboundLocalError as exc:
        outcome = f"UnboundLocalError: {exc}"
    except Exception as exc:  # noqa: BLE001
        outcome = f"{type(exc).__name__}: {exc}"
    finally:
        folder_paths.get_full_path = original_get_full_path
        comfy.utils.load_torch_file = original_load_torch_file

    check("a resolvable LoRA reaches the loader instead of a scoping error",
          outcome == "reached the loader", outcome)

    check("comfy.sd is imported at module scope, not inside the function",
          getattr(node_module, "comfy", None) is not None
          and hasattr(node_module.comfy, "sd"),
          "a function-local import would shadow the module for the whole scope")

    if failures:
        print(f"\n{len(failures)} failing checks:")
        for failed in failures:
            print(f"  - {failed}")
        return 1
    print("\nall krea2 identity lora checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
