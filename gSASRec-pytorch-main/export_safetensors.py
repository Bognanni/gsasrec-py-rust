"""
Converte un checkpoint gSASRec (.pt) in formato .safetensors
compatibile con il crate Candle in Rust.

Uso:
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


# ---------------------------------------------------------------------------
# Caricamento config e modello dal repo originale
# ---------------------------------------------------------------------------

def load_gsasrec_model(checkpoint_path: str, config_path: str):
    """
    Carica il modello gSASRec dal checkpoint .pt usando la config originale.
    Restituisce (model, config).
    """
    # Importa le utility del repo originale
    try:
        from utils import load_config, build_model, get_device
        from dataset_utils import get_num_items
    except ImportError:
        print(
            "Errore: non riesco a importare utils.py / dataset_utils.py dal repo.\n"
            "Assicurati di eseguire lo script dalla cartella root del repo gSASRec-pytorch.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = load_config(config_path)
    device = get_device()
    model = build_model(config)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"✓ Modello caricato da: {checkpoint_path}")
    print(f"  reuse_item_embeddings = {config.reuse_item_embeddings}")
    return model, config


# ---------------------------------------------------------------------------
# Rinomina chiavi per farle corrispondere ai VarBuilder path in Rust
# ---------------------------------------------------------------------------

# Mappa: nome PyTorch (state_dict) → nome atteso da Candle (Rust)
# La maggior parte dei nomi coincide già perché il Rust usa gli stessi pp(...)
# L'unica differenza strutturale è nei TransformerBlock interni.
KEY_MAP = {
    # item / position / seq_norm / output_embedding: nomi identici, nessuna rinomina
    # TransformerBlock: PyTorch usa torch.nn.ModuleList con indici numerici,
    # Candle VarBuilder usa pp("transformer_blocks.N") → stesso formato.
    #
    # MultiHeadAttention: in PyTorch è self.attention, in Rust è self.multihead_attention
    # → questa è l'unica rinomina necessaria.
    "attention.": "multihead_attention.",
}

# LayerNorm: PyTorch salva "weight" e "bias".
# Candle layer_norm carica "weight" e "bias" con gli stessi nomi. ✓
# Linear: PyTorch salva "weight" e "bias". Candle idem. ✓
# Embedding: PyTorch salva "weight". Candle idem. ✓


def remap_keys(state_dict: dict) -> dict:
    """
    Rinomina le chiavi del state_dict per farle corrispondere
    ai path che VarBuilder si aspetta in Rust.
    """
    remapped = {}
    for key, tensor in state_dict.items():
        new_key = key
        for old, new in KEY_MAP.items():
            new_key = new_key.replace(old, new)
        remapped[new_key] = tensor

    return remapped


# ---------------------------------------------------------------------------
# Verifica la corrispondenza tra chiavi Python e Rust
# ---------------------------------------------------------------------------

# Chiavi attese da Candle in Rust (basate su model.rs e transformer.rs)
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
        print(f"\n⚠ Chiavi MANCANTI (attese da Rust ma non trovate nel checkpoint):")
        for k in sorted(missing):
            print(f"   - {k}")
        ok = False
    if extra:
        print(f"\n⚠ Chiavi EXTRA (nel checkpoint ma non usate da Rust):")
        for k in sorted(extra):
            print(f"   + {k}")
        # Non blocchiamo: chiavi extra vengono ignorate da Candle

    if ok and not missing:
        print(f"\n✓ Tutte le {len(expected)} chiavi corrispondono.")
    return not missing


# ---------------------------------------------------------------------------
# Conversione e salvataggio
# ---------------------------------------------------------------------------

def convert(checkpoint_path: str, config_path: str, output_path: str):
    model, config = load_gsasrec_model(checkpoint_path, config_path)

    state_dict = model.state_dict()
    print(f"\n  Tensori nel checkpoint originale ({len(state_dict)}):")
    for k, v in sorted(state_dict.items()):
        print(f"    {k:60s} {tuple(v.shape)}  dtype={v.dtype}")

    remapped = remap_keys(state_dict)

    print(f"\n  Tensori dopo rinomina ({len(remapped)}):")
    for k, v in sorted(remapped.items()):
        print(f"    {k:60s} {tuple(v.shape)}  dtype={v.dtype}")

    verify_keys(remapped, config.num_blocks, config.reuse_item_embeddings)

    # safetensors richiede tensori contigui in memoria e su CPU
    tensors_to_save = {
        k: v.contiguous().cpu() for k, v in remapped.items()
    }

    save_file(tensors_to_save, output_path)
    print(f"\n✓ Salvato: {output_path}")


# ---------------------------------------------------------------------------
# Verifica del file salvato (opzionale, utile per debug)
# ---------------------------------------------------------------------------

def inspect(output_path: str):
    print(f"\nIspezione di {output_path}:")
    with safe_open(output_path, framework="pt") as f:
        for key in sorted(f.keys()):
            t = f.get_tensor(key)
            print(f"  {key:60s} {tuple(t.shape)}  dtype={t.dtype}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Converte checkpoint gSASRec .pt → .safetensors per Candle/Rust"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Percorso del file .pt (es. pre_trained/gsasrec-ml1m-....pt)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Percorso del file config Python (es. config_ml1m.py)",
    )
    parser.add_argument(
        "--output",
        default="model.safetensors",
        help="Percorso di output per il file .safetensors (default: model.safetensors)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Dopo la conversione, stampa il contenuto del file .safetensors",
    )
    args = parser.parse_args()

    convert(args.checkpoint, args.config, args.output)

    if args.inspect:
        inspect(args.output)


if __name__ == "__main__":
    main()