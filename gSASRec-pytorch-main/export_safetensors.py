"""
Convert a checkpoint (.pt) into .safetensors

Use:
    python export_safetensors.py \
        --checkpoint pre_trained/gsasrec-ml1m-step:69216-....pt \
        --config config_ml1m.py \
        --output model.safetensors
"""

import argparse
import sys
import torch
from safetensors.torch import save_file
from safetensors import safe_open


def load_gsasrec_model(checkpoint_path: str, config_path: str):
    """
    Loads the gSASRec model from a PyTorch checkpoint and configuration file.
    Returns (model, config).
    """
    try:
        from utils import load_config, build_model, get_device
        from dataset_utils import get_num_items
    except ImportError:
        print(
            "Error: not able to import utils.py / dataset_utils.py from the repo.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = load_config(config_path)
    device = get_device()
    model = build_model(config)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded from: {checkpoint_path}")
    print(f"  reuse_item_embeddings = {config.reuse_item_embeddings}")
    return model, config


KEY_MAP = {

}


def remap_keys(state_dict: dict) -> dict:
    """
    Rename keys to make them correspond to the paths expected by VarBuilder in Rust.
    """
    remapped = {}
    for key, tensor in state_dict.items():
        new_key = key
        for old, new in KEY_MAP.items():
            new_key = new_key.replace(old, new)
        remapped[new_key] = tensor

    return remapped


def expected_rust_keys(num_blocks: int, reuse_item_embeddings: bool) -> list[str]:
    keys = [
        "item_embedding.weight",
        "position_embedding.weight",
        "seq_norm.weight",
        "seq_norm.bias",
    ]
    for i in range(num_blocks):
        prefix = f"transformer_blocks.{i}"
        keys += [
            f"{prefix}.first_norm.weight",
            f"{prefix}.first_norm.bias",
            f"{prefix}.second_norm.weight",
            f"{prefix}.second_norm.bias",
            f"{prefix}.multihead_attention.query_proj.weight",
            f"{prefix}.multihead_attention.query_proj.bias",
            f"{prefix}.multihead_attention.key_proj.weight",
            f"{prefix}.multihead_attention.key_proj.bias",
            f"{prefix}.multihead_attention.val_proj.weight",
            f"{prefix}.multihead_attention.val_proj.bias",
            f"{prefix}.dense1.weight",
            f"{prefix}.dense1.bias",
            f"{prefix}.dense2.weight",
            f"{prefix}.dense2.bias",
        ]
    if not reuse_item_embeddings:
        keys.append("output_embedding.weight")
    return keys


def verify_keys(remapped: dict, num_blocks: int, reuse_item_embeddings: bool):
    expected = set(expected_rust_keys(num_blocks, reuse_item_embeddings))
    actual = set(remapped.keys())

    missing = expected - actual
    extra = actual - expected

    ok = True
    if missing:
        print(f"\nKeys missing:")
        for k in sorted(missing):
            print(f"   - {k}")
        ok = False
    if extra:
        print(f"\nExtra keys:")
        for k in sorted(extra):
            print(f"   + {k}")

    if ok and not missing:
        print(f"\nAll the {len(expected)} keys correspond.")
    return not missing


def convert(checkpoint_path: str, config_path: str, output_path: str):
    model, config = load_gsasrec_model(checkpoint_path, config_path)

    state_dict = model.state_dict()
    print(f"\n  Tensors in the original checkpoint ({len(state_dict)}):")
    for k, v in sorted(state_dict.items()):
        print(f"    {k:60s} {tuple(v.shape)}  dtype={v.dtype}")

    remapped = remap_keys(state_dict)

    print(f"\n  Tensors after renaming ({len(remapped)}):")
    for k, v in sorted(remapped.items()):
        print(f"    {k:60s} {tuple(v.shape)}  dtype={v.dtype}")

    verify_keys(remapped, config.num_blocks, config.reuse_item_embeddings)
    
    tensors_to_save = {
        k: v.contiguous().cpu() for k, v in remapped.items()
    }

    save_file(tensors_to_save, output_path)
    print(f"\nSaved: {output_path}")


def inspect(output_path: str):
    print(f"\nInspecting {output_path}:")
    with safe_open(output_path, framework="pt") as f:
        for key in sorted(f.keys()):
            t = f.get_tensor(key)
            print(f"  {key:60s} {tuple(t.shape)}  dtype={t.dtype}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert gSASRec .pt into .safetensors for Candle/Rust"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--output",
        default="model.safetensors",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
    )
    args = parser.parse_args()

    convert(args.checkpoint, args.config, args.output)

    if args.inspect:
        inspect(args.output)


if __name__ == "__main__":
    main()