#!/usr/bin/env python3
"""Quantize a compressed-tensors checkpoint's lm_head to W4A16 (group-32 int4).

The lm_head is the one large tensor these checkpoints leave unquantized (bf16,
2.54 GB).  It is also the last bandwidth-bound item of the decode step: the logits
GEMV reads 1.27 GB per rank per pass, which is ~15% of a 250K step at n=5 and grows
with the speculative depth (docs/upstream-branch.md section 6.12).  Quantizing it to
the same pack-quantized int4 scheme as the rest of the model cuts that to ~0.32 GB
per rank per pass and routes it through the same Marlin kernel the trunk already
uses on SM75.

The output is a new checkpoint directory which symlinks every source file except the
shard holding lm_head.  That shard is rewritten with the packed tensors, the index is
regenerated, and config.json gains "lm_head" in group_0's targets and loses it from
ignore.  Both edits are needed: find_matched_target matches the layer name before the
class name, and ParallelLMHead's MRO has no LinearBase, so targets=["Linear"] alone
can never match the head.

Usage:
  quantize_lm_head.py --src models/Qwen3.8-27B-AWQ-INT4-yarn512k \
                      --dst models/Qwen3.8-27B-AWQ-INT4-yarn512k-head4bit \
                      [--observer minmax|mse] [--bits 4] [--group-size 32]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)

HEAD = "lm_head.weight"
MSE_CLIP_RATIOS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6)


def quantize_chunk(weight, bits, group_size, observer):
    """Group-wise asymmetric quantization of a [rows, K] float32 chunk.

    Each group of `group_size` values along K is mapped onto [0, 2**bits - 1] and then
    shifted into the signed range the compressed-tensors packer expects.  With
    observer="mse" the group range is shrunk by a few candidate ratios and the one
    with the lowest squared error wins, which is what an mse observer records.

    Returns (q_signed int8 [rows, K], zp_signed int8 [rows, K/group_size], scale).
    """
    rows, k = weight.shape
    groups = weight.reshape(rows, k // group_size, group_size)
    w_min = groups.min(dim=2, keepdim=True).values
    w_max = groups.max(dim=2, keepdim=True).values
    q_max = (1 << bits) - 1
    span = w_max - w_min

    best_err = best_q = best_zero = best_scale = None
    for ratio in MSE_CLIP_RATIOS if observer == "mse" else (1.0,):
        shrink = span * (1.0 - ratio) / 2.0
        scale = torch.clamp((span - 2.0 * shrink) / q_max, min=1e-8)
        zero = torch.clamp(torch.round(-(w_min + shrink) / scale), 0, q_max)
        q = torch.clamp(torch.round(groups / scale + zero), 0, q_max)
        err = ((q - zero) * scale - groups).pow(2).sum(dim=2)
        if best_err is None:
            best_err, best_q, best_zero, best_scale = err, q, zero, scale
            continue
        take = (err < best_err).unsqueeze(2)
        best_err = torch.where(err < best_err, err, best_err)
        best_q = torch.where(take, q, best_q)
        best_zero = torch.where(take, zero, best_zero)
        best_scale = torch.where(take, scale, best_scale)

    offset = 1 << (bits - 1)
    return (
        (best_q - offset).to(torch.int8).reshape(rows, k),
        (best_zero.squeeze(2) - offset).to(torch.int8),
        best_scale.squeeze(2),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--observer", choices=("minmax", "mse"), default="minmax")
    ap.add_argument("--row-chunk", type=int, default=16384)
    ap.add_argument("--keep-dtype", action="store_true", help="keep the source dtype for scales")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    if dst.exists():
        sys.exit(f"destination already exists: {dst}")
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    head_files = sorted({f for name, f in weight_map.items() if name.startswith("lm_head.")})
    if head_files != [weight_map[HEAD]]:
        sys.exit(f"unexpected shard layout for {HEAD}: {head_files}")
    shard = head_files[0]

    with safe_open(src / shard, framework="pt") as f:
        others = {k: f.get_tensor(k) for k in f.keys() if k != HEAD}
        head = f.get_tensor(HEAD)
    rows, k = head.shape
    if k % args.group_size:
        sys.exit(f"K={k} is not a multiple of group_size={args.group_size}")
    dtype = head.dtype
    src_bytes = head.numel() * head.element_size()
    print(f"  source head : {HEAD} {tuple(head.shape)} {dtype} ({src_bytes / 2**30:.2f} GiB)")
    print(f"  scheme      : {args.bits}-bit int, group {args.group_size}, asymmetric, observer={args.observer}")
    if others:
        print(f"  note        : the shard also holds {len(others)} other tensors, copied through")

    scale_dtype = dtype if args.keep_dtype else torch.bfloat16
    pack_factor = 32 // args.bits  # values per int32 (8 for int4, 4 for int8)
    if rows % pack_factor:
        sys.exit(f"N={rows} is not a multiple of {pack_factor} (needed for packed zero points)")
    packed = torch.empty(rows, math.ceil(k * args.bits / 32), dtype=torch.int32)
    scales = torch.empty(rows, k // args.group_size, dtype=scale_dtype)
    zeros = torch.empty(rows // pack_factor, k // args.group_size, dtype=torch.int32)
    q_signed = torch.empty(rows, k, dtype=torch.int8)
    zp_signed = torch.empty(rows, k // args.group_size, dtype=torch.int8)
    sq_err = abs_err = src_abs = 0.0
    chunk = args.row_chunk
    assert chunk % pack_factor == 0
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        w = head[start:stop].to(torch.float32)
        q, zp, scale = quantize_chunk(w, args.bits, args.group_size, args.observer)
        q_signed[start:stop] = q
        zp_signed[start:stop] = zp
        scales[start:stop] = scale.to(scale_dtype)
        packed[start:stop] = pack_to_int32(q, args.bits, packed_dim=1)
        zeros[start // pack_factor : stop // pack_factor] = pack_to_int32(
            zp, args.bits, packed_dim=0
        )
        deq = (
            (q.to(torch.float32).reshape(stop - start, -1, args.group_size)
             - zp.to(torch.float32).unsqueeze(2))
            * scale.to(torch.float32).unsqueeze(2)
        ).reshape(stop - start, k)
        sq_err += (deq - w).pow(2).sum().item()
        abs_err += (deq - w).abs().sum().item()
        src_abs += w.abs().sum().item()
        print(f"\r  quantizing  : {stop}/{rows} rows", end="", flush=True)
    print()

    # Round trip through the packer: proves the on-disk layout is what the loader
    # (and therefore Marlin's repack) will unpack.
    rt_w = unpack_from_int32(packed, args.bits, torch.Size([rows, k]), packed_dim=1)
    rt_z = unpack_from_int32(zeros, args.bits, torch.Size([rows, k // args.group_size]), packed_dim=0)
    assert torch.equal(rt_w, q_signed), "packed weight round trip failed"
    assert torch.equal(rt_z, zp_signed), "packed zero point round trip failed"
    print("  round trip  : OK (pack -> unpack == quantized values)")

    print(f"  error       : mean|dW|={abs_err / (rows * k):.5f}  rel={abs_err / src_abs * 100:.3f}%  "
          f"rmse={((sq_err / (rows * k)) ** 0.5):.5f}")
    new_bytes = packed.numel() * 4 + scales.numel() * scales.element_size() + zeros.numel() * 4 + 16
    print(f"  head size   : {src_bytes / 2**30:.2f} GiB -> {new_bytes / 2**30:.2f} GiB "
          f"({src_bytes / new_bytes:.2f}x smaller)")

    dst.mkdir(parents=True)
    linked = 0
    for entry in sorted(os.listdir(src)):
        if entry in (shard, "config.json", "model.safetensors.index.json"):
            continue
        os.symlink(os.path.relpath(os.path.realpath(src / entry), dst), dst / entry)
        linked += 1

    out = dict(others)
    out["lm_head.weight_packed"] = packed
    out["lm_head.weight_scale"] = scales
    out["lm_head.weight_zero_point"] = zeros
    out["lm_head.weight_shape"] = torch.tensor([rows, k], dtype=torch.int64)
    save_file(out, dst / shard, metadata={"format": "pt"})
    print(f"  shard       : wrote {dst / shard}")

    new_map = {name: f for name, f in weight_map.items() if not name.startswith("lm_head.")}
    for suffix in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape"):
        new_map[f"lm_head.{suffix}"] = shard
    index["weight_map"] = new_map
    index["metadata"]["total_size"] = index["metadata"]["total_size"] - src_bytes + new_bytes
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    config = json.loads((src / "config.json").read_text())
    quant = config["quantization_config"]
    if "lm_head" not in quant["ignore"]:
        sys.exit("source config does not ignore lm_head; refusing to guess")
    # vLLM's prefix for the head is language_model.lm_head (the multimodal wrapper
    # mounts the language model there), while a text-only top level calls it lm_head.
    # Match both: targets are matched by exact layer name before class name, and
    # ParallelLMHead's MRO has no LinearBase, so targets=["Linear"] never reaches it.
    head_names = ["lm_head", "language_model.lm_head"]
    trunk = next(
        (g for g in quant["config_groups"].values() if "Linear" in g.get("targets", [])),
        None,
    )
    if trunk is None:
        sys.exit("no config group targets Linear; refusing to guess")
    if args.bits == trunk["weights"]["num_bits"]:
        trunk["targets"] = [*trunk["targets"], *head_names]
    else:
        # The head is also the MTP draft's output layer, so it is more sensitive to
        # precision than the trunk: give it its own group with more bits.
        for group in quant["config_groups"].values():
            group["targets"] = [t for t in group.get("targets", []) if t not in head_names]
        weights = dict(trunk["weights"])
        weights["num_bits"] = args.bits
        quant["config_groups"]["head"] = {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": list(head_names),
            "weights": weights,
        }
    quant["ignore"] = [entry for entry in quant["ignore"] if entry != "lm_head"]
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"  config      : wrote {dst / 'config.json'} (targets +lm_head, ignore -lm_head)")
    print(f"  symlinks    : {linked} source files")
    print("  done")


if __name__ == "__main__":
    main()
