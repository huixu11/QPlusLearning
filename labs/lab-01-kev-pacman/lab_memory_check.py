"""Measure this stage's actual LoRA training stack on its costliest encoded case.

Run with the training interpreter, inside the pinned Kev checkout:
    python lab_memory_check.py --report report.json -- <actual kev.train flags>

This probe never invokes kev.train.main or saves a trained checkpoint. Its two
optimizer updates are discarded when the process exits.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import time
from types import MethodType, SimpleNamespace

from optimized_training import ProgressWatchdog, require_optimized_bindings, track_kernel_calls


SCOPE = ("Two discarded optimizer updates, each accumulating eight copies of the costliest encoded "
         "single-record case found across this stage's configured epochs, including permitted forced "
         "none-pair siblings. Padded token count is a memory-cost proxy, not proof of the maximum "
         "VRAM usage of every full-training step, future recipe, allocator state or concurrent process.")


def probe_arguments(arguments, temporary_output):
    """Keep the real flags while avoiding parse_args' existing-output rejection."""
    result = list(arguments)
    original_output = "runs/kev"
    found = False
    for index, value in enumerate(result):
        if value == "--out":
            if index + 1 >= len(result):
                raise ValueError("--out requires a path")
            original_output = result[index + 1]
            result[index + 1] = str(temporary_output)
            found = True
        elif value.startswith("--out="):
            original_output = value.split("=", 1)[1]
            result[index] = "--out=" + str(temporary_output)
            found = True
    if not found:
        result += ["--out", str(temporary_output)]
    return result, original_output


def pass_description(part, shape, pass_tokens, shared_prefix):
    """Describe actual encoded rows, including multi-question records."""
    shapes = [shape(variant.enc) for variant in part]
    permuted = [shape(variant.permuted[0]) for variant in part if variant.permuted]
    tokens = pass_tokens(shapes, shared_prefix)
    second_tokens = pass_tokens(permuted, shared_prefix) if permuted else 0
    return {"variants": len(part), "question_rows": sum(len(branches) for _, branches in shapes),
            "state_tokens": [state for state, _ in shapes],
            "branch_tokens": [branches for _, branches in shapes],
            "row_tokens": [state + branch for state, branches in shapes for branch in branches],
            "padded_tokens": tokens, "permutation_padded_tokens": second_tokens,
            "cost_tokens": tokens + second_tokens,
            "variant_share": sum(variant.share for variant in part)}


def loss_normalizer(batch, accumulation, source_records=1):
    """Kev's group_records: splits retain shares; siblings do not triple record weight."""
    if source_records < 1 or accumulation < 1:
        raise ValueError("Positive accumulation and source record count required")
    variants = sum(variant.share for variant in batch)
    if variants <= 0:
        raise ValueError("Encoded batch has no positive variant shares")
    return accumulation * source_records * (variants / source_records)


def select_costliest(requests, epochs, encode_case, describe_case, forced_eligible=None, progress=None):
    """Scan real encodings without retaining every encoded request in host memory.

    encode_case(request, epoch, forced) returns the same split Variants the trainer
    uses. describe_case(batch) applies the trainer's row_passes and padded shapes.
    """
    best = None
    scanned = 0
    for epoch in range(epochs):
        for request in requests:
            modes = [False]
            if forced_eligible is not None and id(request) in forced_eligible:
                modes.append(True)
            for forced in modes:
                batch = encode_case(request, epoch, forced)
                descriptions = describe_case(batch)
                if not descriptions:
                    raise ValueError("An encoded request produced no training passes")
                score = max(description["cost_tokens"] for description in descriptions)
                scanned += 1
                if best is None or score > best["cost_tokens"]:
                    best = {"request": request, "epoch": epoch, "forced_none_pair": forced,
                            "cost_tokens": score, "passes": descriptions,
                            "encoded_variants": len(batch),
                            "variant_share": sum(variant.share for variant in batch)}
                if progress is not None and scanned % 500 == 0:
                    progress(scanned, score)
    if best is None:
        raise ValueError("No training requests available for the memory check")
    best["cases_scanned"] = scanned
    return best


def validate_probe_recipe(args):
    if args.full_ft or args.batch != 1 or args.accum != 8:
        raise ValueError("The lab memory probe supports single-GPU LoRA with --batch 1 --accum 8")
    if args.device != "cuda" or args.dtype != "bf16" or not args.checkpointing:
        raise ValueError("The lab memory probe requires CUDA, BF16 and gradient checkpointing")
    if args.length_sort or args.pass_tokens_max:
        raise ValueError("The lab memory probe requires the lab's fixed one-record microbatch plan")


def memory_measurement(torch):
    free, total = torch.cuda.mem_get_info()
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "free_bytes": free, "total_bytes": total}


def verify_stage(arguments, receipt, phase):
    import torch
    from kev.checkpoint import Checkpoint, Meta
    from kev.data import none_pair
    from kev.model import DecisionModel, load_tokenizer
    from kev.suite import read_json, read_manifest
    from kev.train import (MAX_GRAD_NORM, batch_loss, encode_batch, microbatch_plan, none_pairs,
                           parse_args, pass_tokens, pinned_revision, row_passes, shape,
                           state_token_counts, training_requests)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    import random

    with tempfile.TemporaryDirectory(prefix="qplus-memory-check-") as scratch:
        parser_flags, original_output = probe_arguments(arguments, Path(scratch) / "unused-checkpoint")
        saved_argv = sys.argv
        try:
            sys.argv = ["kev-stage-memory-check", *parser_flags]
            args = parse_args()
        finally:
            sys.argv = saved_argv
        args.out = original_output
    validate_probe_recipe(args)
    receipt["args"] = vars(args)
    phase("verifying the required pinned optimized CUDA bindings")
    checked = require_optimized_bindings()
    receipt.update(checked)
    receipt["result"] = "running"
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required for this memory check")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest = read_manifest(args.suite) if args.suite else None
    revision = pinned_revision(args, manifest)
    holdout = manifest["holdout_sources"] if manifest else [source for source in args.holdout.split(",") if source]
    anchors = read_json(args.anchor).get("targets", {}) if args.anchor else {}
    anchor_sources = set(args.anchor_sources.split(",")) if args.anchor_sources else None
    phase("loading the stage tokenizer and verified data, including its replay records")
    tokenizer = load_tokenizer(args.base, revision=revision)
    requests = training_requests(args, tokenizer, manifest, holdout)
    receipt.update(training_requests=len(requests), base_revision=revision,
                   requested_training_records=args.epochs * len(requests))
    # The pinned DecisionModel.encode delegates to the pure tokenizer encoder and
    # reads only option_isolation. Bind that exact method without loading weights.
    encoder = SimpleNamespace(option_isolation=bool(args.option_isolation))
    encoder.encode = MethodType(DecisionModel.encode, encoder)
    states = state_token_counts(tokenizer, requests) if args.none_pair_max_state is not None else None
    epoch_pairs = {epoch: none_pairs(args, requests, epoch, states) if states is not None else None
                   for epoch in range(args.epochs)}
    forced_eligible = {id(request) for request in requests
                       if args.p_none_pair > 0
                       and (states is None or states[id(request)] <= args.none_pair_max_state)
                       and none_pair(request, random.Random(0))}
    def encode_case(request, epoch, forced):
        pairs = {id(request)} if forced else epoch_pairs[epoch]
        return encode_batch(encoder, tokenizer, args, [request], epoch, pairs)
    def describe_case(batch):
        return [pass_description(part, shape, pass_tokens, args.shared_prefix)
                for part in row_passes(batch, args.row_budget, args.shared_prefix)]
    phase("scanning actual encoded pass shapes across the configured training epochs")
    selected = select_costliest(requests, args.epochs, encode_case, describe_case, forced_eligible,
                               lambda scanned, score: phase(f"scanned {scanned} tokenized cases; current pass cost {score}"))
    request = selected.pop("request")
    selected.update(request_id=request["_meta"]["id"], source=request["_meta"]["source"])
    receipt["selected_case"] = selected
    receipt["forced_pair_eligible_records"] = len(forced_eligible)
    print("MEMORY_CHECK_SELECTED " + json.dumps(selected), flush=True)
    phase("loading this stage's real backbone, LoRA and pointer head")
    model = optimizer = scheduler = None
    torch.cuda.reset_peak_memory_stats()
    try:
        model = DecisionModel(args.base, tokenizer, "cuda", lora=args.lora, revision=revision,
                              head_dim=args.head_dim, lora_targets=args.lora_targets,
                              option_isolation=bool(args.option_isolation),
                              special_embeddings=bool(args.special_embeddings),
                              dtype=torch.bfloat16 if args.weights_dtype == "bf16" else torch.float32)
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.lm.config.use_cache = False
        if not model.hybrid or model.lm.config._attn_implementation != "sdpa":
            raise RuntimeError("This probe requires the pinned hybrid backbone and PyTorch SDPA")
        meta = Meta(base=args.base, base_revision=revision, lora=args.lora, head_dim=args.head_dim,
                    option_isolation=bool(args.option_isolation), special_embeddings=bool(args.special_embeddings),
                    weights_dtype=args.weights_dtype, holdout=holdout, weights="lora")
        if args.init_from:
            phase("warm-starting the actual parent adapter and head with Kev's compatibility checks")
            receipt["init_source"] = Checkpoint(args.init_from).warm_start(model, meta)
        head_params = list(model.head.parameters())
        head_ids = {id(parameter) for parameter in head_params}
        groups = [{"params": [parameter for parameter in model.trainable_parameters() if id(parameter) not in head_ids], "lr": args.lr},
                  {"params": head_params, "lr": args.head_lr or args.lr}]
        optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, fused=True)
        per_epoch = sum(ends for _, _, ends in microbatch_plan(requests, args, 1, 0))
        total_steps = args.epochs * per_epoch
        total_steps = min(total_steps, args.max_steps) if args.max_steps else total_steps
        if total_steps < 2:
            raise ValueError("The stage recipe must contain at least two optimizer steps")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[args.lr, args.head_lr or args.lr], total_steps=total_steps, pct_start=0.1)
        receipt["full_recipe_optimizer_steps"] = total_steps
        receipt["memory_before_updates"] = memory_measurement(torch)
        model.train()
        bindings = [("delta_rule", qwen, "torch_chunk_gated_delta_rule"),
                    ("convolution", qwen, "causal_conv1d_fn"),
                    ("fused_adamw", torch, "_fused_adamw_")]
        losses = []
        with track_kernel_calls(bindings) as calls:
            for update in range(2):
                update_loss = 0.0
                for microbatch in range(args.accum):
                    pairs = {id(request)} if selected["forced_none_pair"] else epoch_pairs[selected["epoch"]]
                    batch = encode_batch(model, tokenizer, args, [request], selected["epoch"], pairs)
                    normalizer = loss_normalizer(batch, args.accum)
                    for pass_index, part in enumerate(row_passes(batch, args.row_budget, args.shared_prefix)):
                        detail = pass_description(part, shape, pass_tokens, args.shared_prefix)
                        phase(f"update {update + 1}/2, accumulation {microbatch + 1}/8, pass {pass_index + 1}: loss/backward")
                        print("MEMORY_CHECK_PASS " + json.dumps({"update": update + 1, "microbatch": microbatch + 1,
                              "pass": pass_index + 1, "loss_normalizer": normalizer, **detail}), flush=True)
                        loss, _ = batch_loss(model, args, part, "cuda", anchors, anchor_sources,
                                             torch.autocast("cuda", dtype=torch.bfloat16))
                        (loss / normalizer).backward()
                        update_loss += float(loss.detach()) / normalizer
                        del loss, part
                    del batch
                gradient_names = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
                if not any(name.startswith("head.") for name in gradient_names) or not any("lora_B" in name for name in gradient_names):
                    raise RuntimeError("The probe did not backpropagate through the LoRA and pointer head")
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
                phase(f"update {update + 1}/2: fused AdamW and learning-rate scheduler")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                losses.append(update_loss)
                print("MEMORY_CHECK_UPDATE " + json.dumps({"update": update + 1, "loss": update_loss,
                      "kernel_calls": dict(calls), **memory_measurement(torch)}), flush=True)
        if not all(calls.values()) or calls["fused_adamw"] < 2:
            raise RuntimeError(f"The memory check did not execute every required optimized kernel: {calls}")
        receipt.update(kernel_calls=dict(calls), losses=losses, optimizer_steps=2,
                       accumulated_source_records=16, cuda_loss_backward="passed",
                       optimizer="torch.optim.AdamW(fused=True)", attention="sdpa", checkpointing=True,
                       memory=memory_measurement(torch), result="passed")
    finally:
        # No adapter/head saving: neither the initial nor parent checkpoint is changed.
        receipt["memory_at_cleanup"] = memory_measurement(torch)
        scheduler = optimizer = model = None
        if "groups" in locals():
            groups.clear()
        if "head_params" in locals():
            head_params.clear()
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    cli = parser.parse_args(argv)
    if not cli.train_args or cli.train_args[0] != "--" or len(cli.train_args) < 2:
        parser.error("Pass the actual kev.train arguments after --")
    receipt = {"result": "running", "scope": SCOPE, "train_arguments": cli.train_args[1:]}
    cli.report.parent.mkdir(parents=True, exist_ok=True)
    cli.report.write_text(json.dumps(receipt, indent=2) + "\n")
    started = time.perf_counter()
    try:
        with ProgressWatchdog() as phase:
            verify_stage(cli.train_args[1:], receipt, phase)
    except BaseException as error:
        receipt.update(result="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["seconds_including_scan_load_and_compilation"] = time.perf_counter() - started
        cli.report.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Stage memory check passed. Report: " + str(cli.report), flush=True)


if __name__ == "__main__":
    main()
