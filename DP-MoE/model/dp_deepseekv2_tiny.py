#!/usr/bin/env python3
"""
DeepSeek-VL2-Tiny DP fine-tuning with Opacus.

The `ours` experiment mirrors the one-phase hybrid record-owner method used
by dp_switch_transformer.py and dp_olmoe.py:
  - shared attention LoRA parameters use one sampled DP-SGD stream;
  - expert LoRA parameters use record-owner masked per-expert streams;
  - experts compose in parallel within a sparse layer;
  - shared parameters and selected sparse layers compose sequentially.

Supported experiments (same names as dp_deepseek / dp_olmoe):
  upper_bound        – LoRA without DP (ceiling)
  upper_bound_no_lora – full fine-tune without DP
  naive_dp           – LoRA with standard Opacus DP
  ours               – one-phase hybrid record-owner DP-MoE

Default model: deepseek-ai/deepseek-vl2-tiny
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
import weakref
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Opacus imports
# ---------------------------------------------------------------------------
try:
    from opacus import PrivacyEngine as OpacusPrivacyEngine
    from opacus.accountants import RDPAccountant as OpacusRDPAccountant
    from opacus.accountants.utils import get_noise_multiplier as _opacus_get_noise_multiplier
except ImportError:
    OpacusPrivacyEngine = None
    OpacusRDPAccountant = None
    _opacus_get_noise_multiplier = None

try:
    from opacus.accountants import PRVAccountant as OpacusPRVAccountant
except ImportError:
    OpacusPRVAccountant = None

try:
    from prv_accountant import Accountant as _ExternalPRVAccountant
    from prv_accountant import compute_safe_epsilon_bounds as _external_prv_epsilon_bounds
except ImportError:
    _ExternalPRVAccountant = None
    _external_prv_epsilon_bounds = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------------------------------------------------------------------------
# Re-export all shared utilities from dp_deepseek
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from dp_deepseek import (  # noqa: E402
    _fallback_causal_loss,
    _decode,
    _is_attention_linear_name,
    _is_expert_linear_name,
    _is_expert_param_name,
    _is_head_name,
    _is_multimodal_projector_param_name,
    _is_router_name,
    _is_vision_encoder_param_name,
    _num_experts,
    _torch_dtype,
    LoRALinear,
    SparseLayerRef,
    SubsetDataset,
    VQASample,
    VQATrainDataset,
    apply_lora_with_filter,
    build_prompt_for_generation,
    build_prompt_for_training,
    build_user_prompt,
    enable_only,
    evaluate_generation,
    extract_epsilon_rdp,
    freeze_all,
    freeze_routers,
    generate_answer,
    get_sparse_layers,
    load_jsonl_dataset,
    make_train_collate,
    move_batch_to_device,
    run_eval_mode,
    run_single_infer,
    sanitize_parameters,
    select_all_lora_plus_head,
    select_expert_lora_params,
    select_full_params,
    set_seed,
    tensor_to_python,
)

ACCOUNTING_MODE = "prv"
DEFAULT_MODEL = "deepseek-ai/deepseek-vl2-tiny"


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class AdamWWithGradCast(AdamW):
    """Cast Opacus FP32 output grads back to the BF16 parameter dtype."""

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and p.grad.dtype != p.dtype:
                    p.grad = p.grad.to(dtype=p.dtype)
        return super().step(closure)


# ---------------------------------------------------------------------------
# DeepSeek-VL2-specific model loader
# ---------------------------------------------------------------------------

def load_vlm(
    model_name: str,
    device: torch.device,
    dtype_arg: str,
    trust_remote_code: bool,
):
    """
    Load DeepSeek-VL2-Tiny via the official deepseek_vl2 package.

    The model weights on HuggingFace have no auto_map in config.json and no
    custom Python files in the snapshot, so AutoConfig/AutoModelForCausalLM
    with trust_remote_code cannot find the implementation.  The deepseek_vl2
    package ships the model code and must be installed:
        pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git --no-deps
        pip install attrdict
    """
    try:
        from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
    except ImportError as exc:
        raise RuntimeError(
            "deepseek_vl2 package is required to load DeepSeek-VL2-Tiny.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git --no-deps\n"
            "  pip install attrdict"
        ) from exc

    dtype = _torch_dtype(dtype_arg)

    processor = DeepseekVLV2Processor.from_pretrained(model_name)
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    # DeepseekVLV2Config doesn't define use_cache; set it so the forward pass
    # doesn't raise AttributeError when falling back to self.config.use_cache.
    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return processor, model


# ---------------------------------------------------------------------------
# DeepSeek-VL2-specific sparse layer detection
# ---------------------------------------------------------------------------

def _is_deepseekv2_moe_module(mod: nn.Module) -> bool:
    """DeepseekV2MoE has .experts (ModuleList) and .gate (MoEGate with .weight).
    MoEGate is NOT nn.Linear so the generic check in dp_deepseek fails."""
    if not hasattr(mod, "experts") or not hasattr(mod, "gate"):
        return False
    gate = mod.gate
    # MoEGate has a raw nn.Parameter weight, not wrapped in nn.Linear
    if isinstance(gate, nn.Linear):
        return True
    if hasattr(gate, "weight") and isinstance(gate.weight, nn.Parameter):
        return True
    return False


def get_sparse_layers(model: nn.Module) -> List[SparseLayerRef]:
    """Override: detects DeepseekV2MoE layers in addition to the generic patterns."""
    from dp_deepseek import _is_sparse_moe_module  # generic detector
    refs: List[SparseLayerRef] = []
    lid = 0
    for name, mod in model.named_modules():
        if any(name.startswith(f"{ref.module_name}.") for ref in refs):
            continue
        if _is_sparse_moe_module(mod) or _is_deepseekv2_moe_module(mod):
            refs.append(SparseLayerRef(layer_id=lid, module_name=name, sparse_mlp=mod))
            lid += 1
    return refs


def _select_deepseekv2_shared_lora_params(
    model: nn.Module,
    train_router: bool,
) -> List[Tuple[str, nn.Parameter]]:
    """
    Shared-stream parameter selection for DeepSeek-VL2.

    Only enables attention LoRA adapters (lora_A / lora_B) in the language
    model.  Base linear weights, norms, and embeddings are excluded because
    they are either non-standard Opacus types
    (DeepseekV2RMSNorm) or add non-DP params that confuse the wrapping step.
    Router/gate params are also excluded by default.
    """
    out: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        # Only lora adapters in the language sub-model
        if ("lora_A" not in name and "lora_B" not in name):
            continue
        if name.startswith("vision.") or ".vision." in name:
            continue
        if _is_multimodal_projector_param_name(name):
            continue
        if _is_expert_param_name(name):
            continue
        if _is_router_name(name) and not train_router:
            continue
        out.append((name, p))
    return out


def _router_logits_deepseekv2(sparse_mlp: nn.Module, flat_hidden: torch.Tensor) -> torch.Tensor:
    """Compute router logits for DeepseekV2MoE.gate (MoEGate uses F.linear on .weight)."""
    import dp_deepseek as _ds
    try:
        # Try the generic paths first (nn.Linear gate, wg, router.classifier, etc.)
        return _ds._router_logits.__wrapped__(sparse_mlp, flat_hidden)
    except (RuntimeError, AttributeError):
        pass
    gate = getattr(sparse_mlp, "gate", None)
    if gate is not None and hasattr(gate, "weight"):
        w = gate.weight
        h = flat_hidden.to(dtype=w.dtype) if flat_hidden.dtype != w.dtype else flat_hidden
        return torch.nn.functional.linear(h, w, None)
    raise RuntimeError(f"Cannot find router logits for {type(sparse_mlp)}")


# Monkey-patch dp_deepseek._router_logits so that patch_sparse_mlp_for_record_routing
# and construct_record_assignments use our DeepseekV2MoE-aware version.
import dp_deepseek as _ds_module
_ds_module._router_logits.__wrapped__ = _ds_module._router_logits  # save original
_ds_module._router_logits = _router_logits_deepseekv2


def _select_primary_record_owners_deepseekv2(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    num_experts: int,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if selected_experts.dim() != 3:
        raise ValueError(f"Expected selected_experts [B,S,K], got {tuple(selected_experts.shape)}")
    if routing_weights.shape != selected_experts.shape:
        raise ValueError(
            f"Routing weight shape mismatch: weights={tuple(routing_weights.shape)} "
            f"selected={tuple(selected_experts.shape)}"
        )
    if selected_experts.shape[:2] != (int(batch_size), int(seq_len)):
        raise ValueError(
            f"Selected-expert shape mismatch: selected={tuple(selected_experts.shape)} "
            f"expected batch/seq={(int(batch_size), int(seq_len))}"
        )
    if valid_mask.shape != selected_experts.shape[:2]:
        raise ValueError(
            f"Valid-mask shape mismatch: mask={tuple(valid_mask.shape)} "
            f"selected={tuple(selected_experts.shape)}"
        )
    selected = selected_experts.view(batch_size, seq_len, -1)
    weights = routing_weights.view(batch_size, seq_len, -1)
    scores = torch.zeros(batch_size, num_experts, device=weights.device, dtype=weights.dtype)
    masked_weights = weights * valid_mask.unsqueeze(-1).to(weights.dtype)
    scores.scatter_add_(1, selected.reshape(batch_size, -1), masked_weights.reshape(batch_size, -1))
    return scores.argmax(dim=-1)


def patch_sparse_mlp_for_record_routing(
    model: nn.Module,
    sparse_layers: Sequence[SparseLayerRef],
) -> None:
    """
    Preserve native DeepSeek top-k routing while masking non-owner expert LoRA
    gradients. Native DeepSeek dispatches flattened tokens to each expert, so
    token-level Opacus samples are tagged here and aggregated back to records
    after backward.
    """
    from deepseek_vl2.models.modeling_deepseek import AddAuxiliaryLoss

    for layer in sparse_layers:
        mlp = layer.sparse_mlp
        if getattr(mlp, "_record_routing_patched", False):
            continue

        owner_ref = weakref.ref(model)
        layer_id = int(layer.layer_id)
        orig_forward = mlp.forward

        def _patched_forward(
            self,
            hidden_states: torch.Tensor,
            _owner_ref=owner_ref,
            _layer_id=layer_id,
            _orig_forward=orig_forward,
        ):
            owner = _owner_ref()
            if owner is None:
                raise RuntimeError("Owner model reference was lost in patched sparse MLP")

            batch_size, seq_len, _hidden_dim = hidden_states.shape
            topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
            selected_experts = topk_idx.view(batch_size, seq_len, -1)
            routing_weights = topk_weight.view(batch_size, seq_len, -1)
            self._last_raw_expert_index = selected_experts[..., 0].detach()
            self._last_raw_topk_expert_indices = selected_experts.detach()
            self._last_raw_topk_routing_weights = routing_weights.detach()

            if _layer_id not in owner._record_route_layer_ids:
                return _orig_forward(hidden_states)

            active_mask = getattr(owner, "_active_attention_mask", None)
            if active_mask is None or tuple(active_mask.shape) != (batch_size, seq_len):
                valid_mask = torch.ones(batch_size, seq_len, device=hidden_states.device, dtype=torch.bool)
            else:
                valid_mask = active_mask.to(device=hidden_states.device, dtype=torch.bool)

            forced_expert = getattr(owner, "_forced_record_expert_by_layer", {}).get(_layer_id)
            fixed_assignments = getattr(owner, "_record_route_fixed_assignments", {}).get(_layer_id)
            batch_indices = getattr(owner, "_active_batch_indices", None)
            if forced_expert is not None:
                record_experts = torch.full(
                    (batch_size,),
                    int(forced_expert),
                    device=hidden_states.device,
                    dtype=torch.long,
                )
            elif fixed_assignments is not None and batch_indices is not None:
                idx = batch_indices.detach().to(device="cpu", dtype=torch.long).view(-1)
                if idx.numel() != batch_size:
                    raise RuntimeError(
                        f"Batch index count mismatch at sparse layer {_layer_id}: "
                        f"got {idx.numel()}, expected {batch_size}"
                    )
                if idx.min().item() < 0 or idx.max().item() >= fixed_assignments.numel():
                    raise RuntimeError(
                        f"Batch indices out of range for fixed assignments at sparse layer {_layer_id}: "
                        f"index range=[{idx.min().item()}, {idx.max().item()}], "
                        f"assignment_size={fixed_assignments.numel()}"
                    )
                record_experts = fixed_assignments.index_select(0, idx).to(
                    device=hidden_states.device,
                    dtype=torch.long,
                )
            else:
                record_experts = _select_primary_record_owners_deepseekv2(
                    selected_experts,
                    routing_weights,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_experts=len(self.experts),
                    valid_mask=valid_mask,
                )
            self._last_record_expert_index = record_experts.detach()

            if getattr(owner, "_record_route_weight_mode", "router_prob") == "router_prob":
                flat_experts = topk_idx.reshape(-1)
                top_k = int(topk_idx.shape[-1])
                flat_samples = (
                    torch.arange(batch_size * seq_len, device=hidden_states.device)
                    .repeat_interleave(top_k)
                    .div(seq_len, rounding_mode="floor")
                )

                if self.training:
                    expert_samples = [
                        flat_samples[flat_experts == expert_id]
                        for expert_id in range(len(self.experts))
                    ]
                else:
                    # moe_infer sorts flattened routing slots by expert before
                    # invoking each expert. Match that row order for the masks.
                    order = flat_experts.argsort()
                    sorted_experts = flat_experts.index_select(0, order)
                    sorted_samples = flat_samples.index_select(0, order)
                    expert_samples = [
                        sorted_samples[sorted_experts == expert_id]
                        for expert_id in range(len(self.experts))
                    ]

                touched = []
                for expert_id, (expert, token_to_sample) in enumerate(
                    zip(self.experts, expert_samples)
                ):
                    owner_mask = record_experts.index_select(0, token_to_sample) == int(expert_id)
                    for module in expert.modules():
                        if not isinstance(module, LoRALinear):
                            continue
                        had_mask = hasattr(module, "_record_owner_mask")
                        previous_mask = getattr(module, "_record_owner_mask", None)
                        module._record_owner_mask = owner_mask
                        module._record_token_to_sample = token_to_sample.detach()
                        module._record_num_samples = int(batch_size)
                        touched.append((module, had_mask, previous_mask))
                try:
                    return _orig_forward(hidden_states)
                finally:
                    for module, had_mask, previous_mask in touched:
                        if had_mask:
                            module._record_owner_mask = previous_mask
                        elif hasattr(module, "_record_owner_mask"):
                            delattr(module, "_record_owner_mask")

            # Diagnostic mode retained for experiments that intentionally remove
            # native router weights. Production grids use router_prob above.
            routing_weights = torch.ones_like(routing_weights)

            out_states = torch.zeros_like(hidden_states)
            for expert_id, expert in enumerate(self.experts):
                slot_mask = selected_experts == int(expert_id)
                token_mask = slot_mask.any(dim=-1) & valid_mask
                expert_weights = (
                    routing_weights * slot_mask.to(routing_weights.dtype)
                ).sum(dim=-1)
                expert_input = hidden_states * token_mask.unsqueeze(-1).to(hidden_states.dtype)
                owner_mask = record_experts == int(expert_id)
                expert_output = _ds_module._forward_expert_with_lora_owner_mask(
                    expert,
                    expert_input,
                    owner_mask,
                )
                out_states = out_states + expert_output * expert_weights.unsqueeze(-1).to(expert_output.dtype)

            if aux_loss is not None:
                out_states = AddAuxiliaryLoss.apply(out_states, aux_loss)
            if getattr(self.config, "n_shared_experts", None) is not None:
                out_states = out_states + self.shared_experts(hidden_states)
            return out_states

        mlp.forward = types.MethodType(_patched_forward, mlp)
        mlp._record_routing_patched = True


def _aggregate_routed_expert_grad_samples(
    sparse_layers: Sequence[SparseLayerRef],
    batch_size: int,
) -> None:
    """Convert native token-dispatch grad samples into record-level samples."""
    for layer in sparse_layers:
        for expert in layer.sparse_mlp.experts:
            for module in expert.modules():
                if not isinstance(module, LoRALinear):
                    continue
                token_to_sample = getattr(module, "_record_token_to_sample", None)
                num_samples = int(getattr(module, "_record_num_samples", batch_size))
                if token_to_sample is None:
                    continue
                if num_samples != int(batch_size):
                    raise RuntimeError(
                        "Routed expert grad-sample batch mismatch: "
                        f"context={num_samples}, backward={batch_size}"
                    )
                for projection in (module.lora_A, module.lora_B):
                    for param in projection.parameters():
                        grad_sample = _grad_sample_as_tensor(param)
                        if grad_sample is None:
                            continue
                        if grad_sample.size(0) != token_to_sample.numel():
                            raise RuntimeError(
                                "Routed expert token grad-sample mismatch: "
                                f"grad_rows={grad_sample.size(0)}, "
                                f"route_rows={token_to_sample.numel()}"
                            )
                        record_grad = torch.zeros(
                            (num_samples, *tuple(grad_sample.shape[1:])),
                            device=grad_sample.device,
                            dtype=grad_sample.dtype,
                        )
                        if token_to_sample.numel() > 0:
                            record_grad.index_add_(
                                0,
                                token_to_sample.to(grad_sample.device),
                                grad_sample,
                            )
                        param.grad_sample = record_grad


def patch_sparse_mlp_for_naive_dp(
    model: nn.Module,
    sparse_layers: Sequence[SparseLayerRef],
) -> None:
    """
    Preserve DeepSeek top-k routing while keeping expert inputs batch-first.

    Native DeepSeek-VL2 dispatches selected expert tokens as flat 2D tensors.
    Opacus then interprets token rows as sample rows, which makes routed expert
    LoRA grad_sample tensors have shape [tokens, ...] instead of [batch, ...].
    Calling each expert on a masked [batch, seq, hidden] tensor keeps the same
    routed weighted output but gives Opacus a stable per-record leading dim.
    """
    from deepseek_vl2.models.modeling_deepseek import AddAuxiliaryLoss

    for layer in sparse_layers:
        mlp = layer.sparse_mlp
        if getattr(mlp, "_naive_dp_patched", False):
            continue

        def _naive_forward(self, hidden_states: torch.Tensor):
            batch_size, seq_len, _hidden_dim = hidden_states.shape
            topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
            selected_experts = topk_idx.view(batch_size, seq_len, -1)
            routing_weights = topk_weight.view(batch_size, seq_len, -1)
            self._last_raw_expert_index = selected_experts[..., 0].detach()
            self._last_raw_topk_expert_indices = selected_experts.detach()
            self._last_raw_topk_routing_weights = routing_weights.detach()

            active_mask = getattr(model, "_active_attention_mask", None)
            if active_mask is None or tuple(active_mask.shape) != (batch_size, seq_len):
                valid_mask = torch.ones(
                    batch_size,
                    seq_len,
                    device=hidden_states.device,
                    dtype=torch.bool,
                )
            else:
                valid_mask = active_mask.to(device=hidden_states.device, dtype=torch.bool)

            out_states = torch.zeros_like(hidden_states)
            for expert_id, expert in enumerate(self.experts):
                slot_mask = selected_experts == int(expert_id)
                token_mask = slot_mask.any(dim=-1) & valid_mask
                expert_weights = (
                    routing_weights * slot_mask.to(routing_weights.dtype)
                ).sum(dim=-1)
                expert_input = hidden_states * token_mask.unsqueeze(-1).to(hidden_states.dtype)
                expert_output = expert(expert_input)
                out_states = out_states + expert_output * expert_weights.unsqueeze(-1).to(expert_output.dtype)

            if aux_loss is not None:
                out_states = AddAuxiliaryLoss.apply(out_states, aux_loss)
            if getattr(self.config, "n_shared_experts", None) is not None:
                out_states = out_states + self.shared_experts(hidden_states)
            return out_states

        mlp.forward = types.MethodType(_naive_forward, mlp)
        mlp._naive_dp_patched = True


# ---------------------------------------------------------------------------
# DeepSeek-VL2-specific dataset, collate, generate, and evaluate
# ---------------------------------------------------------------------------

class DeepSeekV2TrainDataset(Dataset):
    """VQA training dataset using DeepseekVLV2Processor.process_one."""

    def __init__(self, samples: Sequence[VQASample], processor, max_length: int):
        self.samples = list(samples)
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from PIL import Image as PILImage
        s = self.samples[idx]
        image = PILImage.open(s.image_path).convert("RGB")
        user_text = build_user_prompt(s.question, s.options)
        conversations = [
            {"role": "user", "content": f"<image>\n{user_text}"},
            {"role": "assistant", "content": s.answer or ""},
        ]
        out = self.processor.process_one(
            conversations=conversations,
            images=[image],
            inference_mode=False,
        )
        seq_len = out.input_ids.shape[0]
        if seq_len > self.max_length:
            out.input_ids = out.input_ids[: self.max_length]
            out.target_ids = out.target_ids[: self.max_length]
            out.images_seq_mask = out.images_seq_mask[: self.max_length]
        return {
            "idx": torch.tensor(int(idx), dtype=torch.long),
            "input_ids": out.input_ids,
            "attention_mask": torch.ones_like(out.input_ids, dtype=torch.long),
            "labels": out.target_ids,
            "images": out.images,
            "images_seq_mask": out.images_seq_mask,
            "images_spatial_crop": out.images_spatial_crop,
        }


def _deepseekv2_collate(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences and stack images into a batch."""
    import torch.nn.functional as F

    max_len = max(x["input_ids"].shape[0] for x in batch)
    max_n_patches = max(x["images"].shape[0] for x in batch)
    max_n_images = max(x["images_spatial_crop"].shape[0] for x in batch)

    idx_list, input_ids_list, attention_mask_list = [], [], []
    labels_list, seq_mask_list, images_list, spatial_list = [], [], [], []
    for x in batch:
        pad = max_len - x["input_ids"].shape[0]
        idx_list.append(x["idx"])
        input_ids_list.append(F.pad(x["input_ids"], (0, pad), value=pad_id))
        attention_mask_list.append(F.pad(x["attention_mask"], (0, pad), value=0))
        labels_list.append(F.pad(x["labels"], (0, pad), value=-100))
        seq_mask_list.append(F.pad(x["images_seq_mask"], (0, pad), value=False))

        n_patch_pad = max_n_patches - x["images"].shape[0]
        img_pad = torch.zeros(n_patch_pad, *x["images"].shape[1:], dtype=x["images"].dtype)
        images_list.append(torch.cat([x["images"], img_pad], dim=0))

        n_image_pad = max_n_images - x["images_spatial_crop"].shape[0]
        spatial_pad = torch.zeros(n_image_pad, 2, dtype=x["images_spatial_crop"].dtype)
        spatial_list.append(torch.cat([x["images_spatial_crop"], spatial_pad], dim=0))

    return {
        "idx": torch.stack(idx_list),
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
        "images": torch.stack(images_list),
        "images_seq_mask": torch.stack(seq_mask_list),
        "images_spatial_crop": torch.stack(spatial_list),
    }


@torch.no_grad()
def deepseekv2_generate_answer(
    *,
    model,
    processor,
    image,
    question: str,
    options: Optional[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    user_text = build_user_prompt(question, options)
    conversations = [
        {"role": "user", "content": f"<image>\n{user_text}"},
        {"role": "assistant", "content": ""},
    ]
    out = processor.process_one(
        conversations=conversations,
        images=[image],
        inference_mode=True,
    )
    model_dtype = next(model.parameters()).dtype
    input_ids = out.input_ids.unsqueeze(0).to(device)
    images = out.images.unsqueeze(0).to(device=device, dtype=model_dtype)
    images_seq_mask = out.images_seq_mask.unsqueeze(0).to(device)
    images_spatial_crop = out.images_spatial_crop.unsqueeze(0).to(device)

    do_sample = temperature > 0
    gen_ids = model.generate(
        input_ids=input_ids,
        images=images,
        images_seq_mask=images_seq_mask,
        images_spatial_crop=images_spatial_crop,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-6) if do_sample else 1.0,
        top_p=top_p,
        pad_token_id=processor.pad_id,
        eos_token_id=processor.eos_id,
    )
    prompt_len = input_ids.shape[1]
    gen_tail = gen_ids[:, prompt_len:]
    return processor.tokenizer.decode(gen_tail[0].tolist(), skip_special_tokens=True).strip()


def deepseekv2_evaluate_generation(
    *,
    model,
    processor,
    samples: Sequence[VQASample],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    limit: int = -1,
    pred_path: str = "",
    show_progress: bool = False,
    progress_desc: str = "eval",
) -> Dict:
    from PIL import Image as PILImage
    import json as _json
    from dp_deepseek import normalize_text, exact_match

    model.eval()
    preds = []
    if limit > 0:
        samples = samples[:limit]
    iterator: Iterable = samples
    if tqdm and show_progress:
        iterator = tqdm(samples, desc=progress_desc, leave=False)
    for s in iterator:
        try:
            image = PILImage.open(s.image_path).convert("RGB")
            pred = deepseekv2_generate_answer(
                model=model, processor=processor, image=image,
                question=s.question, options=s.options,
                device=device, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )
        except Exception as e:
            pred = ""
            print(f"[eval] generation error: {e}")
        preds.append({"pred": pred, "answer": s.answer, "question": s.question})

    n = len(preds)
    if n == 0:
        return {"exact_match": 0.0, "n_samples": 0}
    em = sum(
        exact_match(normalize_text(p["pred"]), normalize_text(p["answer"] or ""))
        for p in preds
    ) / n
    if pred_path:
        with open(pred_path, "w", encoding="utf-8") as f:
            _json.dump(preds, f, indent=2)
    return {"exact_match": float(em), "n_samples": n}


def _prepare_model_batch(
    batch: Dict[str, torch.Tensor],
    *,
    model: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    moved = move_batch_to_device(batch, device)
    indices = moved.pop("idx", None)
    if "images" in moved:
        moved["images"] = moved["images"].to(dtype=next(model.parameters()).dtype)
    return moved, indices


def _forward_with_routing_context(
    module: nn.Module,
    *,
    base_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    batch_indices: Optional[torch.Tensor],
):
    previous_mask = getattr(base_model, "_active_attention_mask", None)
    previous_indices = getattr(base_model, "_active_batch_indices", None)
    base_model._active_attention_mask = batch.get("attention_mask")
    base_model._active_batch_indices = batch_indices
    try:
        return module(**batch, return_dict=True)
    finally:
        base_model._active_attention_mask = previous_mask
        base_model._active_batch_indices = previous_indices


def _causal_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _outputs_loss(outputs, labels: torch.Tensor) -> torch.Tensor:
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss
    return _fallback_causal_loss(outputs, labels)


@torch.no_grad()
def construct_record_assignments(
    model: nn.Module,
    train_ds: Dataset,
    tokenizer,
    target_layer: SparseLayerRef,
    batch_size: int,
    device: torch.device,
    collate_fn,
    show_progress: bool = False,
) -> List[int]:
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    assignments = [-1 for _ in range(len(train_ds))]
    model.eval()
    iterator: Iterable = loader
    if tqdm and show_progress:
        iterator = tqdm(
            loader,
            desc=f"OURS assign L{target_layer.layer_id}",
            leave=False,
        )
    for raw_batch in iterator:
        batch, indices = _prepare_model_batch(raw_batch, model=model, device=device)
        if indices is None:
            raise RuntimeError("DeepSeek record assignment batches require stable sample indices")
        _ = _forward_with_routing_context(
            model,
            base_model=model,
            batch=batch,
            batch_indices=indices,
        )
        selected = getattr(target_layer.sparse_mlp, "_last_raw_topk_expert_indices", None)
        weights = getattr(target_layer.sparse_mlp, "_last_raw_topk_routing_weights", None)
        if selected is None or weights is None:
            raise RuntimeError(
                f"Native top-k routes missing for sparse layer {target_layer.layer_id}"
            )
        valid_mask = batch["attention_mask"].detach().bool()
        owners = _select_primary_record_owners_deepseekv2(
            selected.detach(),
            weights.detach(),
            batch_size=int(selected.size(0)),
            seq_len=int(selected.size(1)),
            num_experts=_num_experts(target_layer.sparse_mlp),
            valid_mask=valid_mask,
        )
        for sample_idx, owner in zip(indices.detach().cpu().tolist(), owners.detach().cpu().tolist()):
            assignments[int(sample_idx)] = int(owner)

    if any(owner < 0 for owner in assignments):
        raise RuntimeError(f"Incomplete record assignments for layer={target_layer.layer_id}")
    return assignments


# ---------------------------------------------------------------------------
# Opacus-incompatible parameter filter
# ---------------------------------------------------------------------------

def _is_opacus_incompatible_param(name: str) -> bool:
    """
    Router / gate parameters are excluded from GradSampleModule wrapping.
    Per-sample gradients for discrete routing decisions are ill-defined and
    the router is frozen during all DP stages anyway.
    """
    return _is_router_name(name)


# ---------------------------------------------------------------------------
# PRV noise calibration (mirrors dp_olmoe.find_noise_multiplier_prv)
# ---------------------------------------------------------------------------

def find_noise_multiplier_prv(
    *,
    sampling_probability: float,
    num_steps: int,
    target_epsilon: float,
    target_delta: float,
    eps_error: float = 0.1,
) -> float:
    if target_epsilon <= 0.0:
        raise ValueError(f"target_epsilon must be positive, got {target_epsilon}")

    if _ExternalPRVAccountant is not None:
        eps_error = max(float(eps_error), 1e-6)

        def _eps_upper(mu: float) -> float:
            bounds = _external_prv_epsilon_bounds(
                noise_multiplier=float(mu),
                sample_rate=float(sampling_probability),
                steps=max(int(num_steps), 1),
                target_delta=float(target_delta),
                eps_error=eps_error / 2.0,
            )
            if bounds is None:
                raise RuntimeError("External PRV accountant unexpectedly unavailable")
            return bounds[2]

        lo, hi = 0.0, 1.0
        while _eps_upper(hi) > float(target_epsilon):
            hi *= math.sqrt(2.0)
            if hi > 100.0:
                raise RuntimeError(
                    "PRV noise calibration did not converge. "
                    "Try raising epsilon or lowering sample rate."
                )
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if _eps_upper(mid) > float(target_epsilon):
                lo = mid
            else:
                hi = mid
            if hi - lo <= max(hi, 1.0) * 1e-4:
                break
        return float(hi)

    if _opacus_get_noise_multiplier is not None:
        try:
            return float(
                _opacus_get_noise_multiplier(
                    target_epsilon=float(target_epsilon),
                    target_delta=float(target_delta),
                    sample_rate=float(sampling_probability),
                    steps=max(int(num_steps), 1),
                    accountant="prv",
                )
            )
        except TypeError:
            solver_epochs = float(max(int(num_steps), 1)) * float(sampling_probability)
            return float(
                _opacus_get_noise_multiplier(
                    target_epsilon=float(target_epsilon),
                    target_delta=float(target_delta),
                    sample_rate=float(sampling_probability),
                    epochs=solver_epochs,
                    accountant="prv",
                )
            )

    raise RuntimeError(
        "PRV noise calibration unavailable. Install `prv-accountant` or a recent Opacus."
    )


def prv_epsilon_for_fixed_steps(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    target_delta: float,
    fallback_epsilon: float,
    eps_error: float = 0.1,
) -> float:
    """Recompute PRV epsilon from (sigma, q, T) — mirrors dp_olmoe.prv_epsilon_for_fixed_steps."""
    if steps <= 0 or noise_multiplier <= 0:
        return 0.0
    if _external_prv_epsilon_bounds is not None:
        bounds = _external_prv_epsilon_bounds(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=steps,
            target_delta=target_delta,
            eps_error=max(float(eps_error), 1e-6),
        )
        if bounds is not None:
            return bounds[2]
    if OpacusPRVAccountant is not None:
        acc = OpacusPRVAccountant()
        for _ in range(int(steps)):
            acc.step(noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
        return float(acc.get_epsilon(float(target_delta)))
    return float(fallback_epsilon)


def rdp_epsilon_for_fixed_steps(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    target_delta: float,
    fallback_epsilon: float,
) -> float:
    if steps <= 0 or noise_multiplier <= 0:
        return 0.0
    if OpacusRDPAccountant is None:
        return float(fallback_epsilon)
    acc = OpacusRDPAccountant()
    for _ in range(int(steps)):
        acc.step(noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
    return float(acc.get_epsilon(float(target_delta)))


def _stacked_noise_multiplier(
    *,
    sample_rate: float,
    steps: int,
    target_epsilon: float,
    target_delta: float,
    args: argparse.Namespace,
) -> float:
    if args.no_dp:
        return 0.0
    if args.noise_multiplier is not None and args.noise_multiplier > 0:
        return float(args.noise_multiplier)
    return find_noise_multiplier_prv(
        sampling_probability=float(sample_rate),
        num_steps=max(int(steps), 1),
        target_epsilon=float(target_epsilon),
        target_delta=float(target_delta),
        eps_error=float(getattr(args, "prv_eps_error", 0.1)),
    )


def _stacked_privacy_epsilon(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    target_delta: float,
    fallback_epsilon: float,
    args: argparse.Namespace,
) -> float:
    return prv_epsilon_for_fixed_steps(
        noise_multiplier=float(noise_multiplier),
        sample_rate=float(sample_rate),
        steps=int(steps),
        target_delta=float(target_delta),
        fallback_epsilon=float(fallback_epsilon),
        eps_error=float(getattr(args, "prv_eps_error", 0.1)),
    )


def resolve_ours_privacy_budget(args: argparse.Namespace, num_layers: int) -> Dict[str, float]:
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive for ours privacy budget, got {num_layers}")
    shared_ratio = float(args.epsilon_shared_ratio)
    if not (0.0 < shared_ratio < 1.0):
        raise ValueError(f"--epsilon_shared_ratio must be in (0, 1), got {shared_ratio}")

    adjacency_factor = 1 if args.adjacency == "add_remove" else 2
    eps_shared = shared_ratio * float(args.epsilon_total)
    eps_experts_total = (1.0 - shared_ratio) * float(args.epsilon_total)
    eps_expert = eps_experts_total / float(adjacency_factor * num_layers)
    delta_shared = shared_ratio * float(args.delta)
    delta_experts_total = (1.0 - shared_ratio) * float(args.delta)
    delta_expert = delta_experts_total / float(adjacency_factor * num_layers)
    return {
        "adjacency_factor": float(adjacency_factor),
        "eps_shared": eps_shared,
        "eps_experts_total": eps_experts_total,
        "eps_expert": eps_expert,
        "eps_layer": float(adjacency_factor) * eps_expert,
        "delta_shared": delta_shared,
        "delta_experts_total": delta_experts_total,
        "delta_expert": delta_expert,
        "delta_layer": float(adjacency_factor) * delta_expert,
    }


def select_sparse_layers_by_args(
    sparse_layers: Sequence[SparseLayerRef],
    args: argparse.Namespace,
) -> List[SparseLayerRef]:
    n_layers = len(sparse_layers)
    start = int(args.start_layer)
    count = int(args.num_layers_to_train)
    if start < 0:
        resolved = n_layers + start
    else:
        resolved = start
    if resolved < 0 or resolved >= n_layers:
        raise RuntimeError(
            f"--start_layer={start} resolves to {resolved}, outside sparse layer range "
            f"[0, {max(n_layers - 1, 0)}]"
        )
    if count < 0:
        first, end = resolved, n_layers
    elif start < 0:
        end = resolved + 1
        first = max(0, end - count)
    else:
        first, end = resolved, min(resolved + count, n_layers)
    selected = list(sparse_layers[first:end])
    if not selected:
        raise RuntimeError("No sparse layers selected for one-phase hybrid training")
    return selected


def _select_deepseekv2_shared_params(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    return _select_deepseekv2_shared_lora_params(model, train_router=False)


def _select_deepseekv2_language_lora_params(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """Select language attention/expert LoRA adapters, excluding the full LM head."""
    return [
        (name, param)
        for name, param in select_all_lora_plus_head(model)
        if not (name.startswith("vision.") or ".vision." in name)
        and not _is_multimodal_projector_param_name(name)
        and not _is_head_name(name)
    ]


def _select_deepseekv2_expert_lora_params(
    model: nn.Module,
    layer: SparseLayerRef,
    expert_id: int,
) -> List[Tuple[str, nn.Parameter]]:
    return [
        (name, p)
        for name, p in select_expert_lora_params(model, layer, expert_id)
        if "lora_A" in name or "lora_B" in name
    ]


def _grad_sample_as_tensor(p: nn.Parameter) -> Optional[torch.Tensor]:
    grad_sample = getattr(p, "grad_sample", None)
    if grad_sample is None:
        return None
    if isinstance(grad_sample, list):
        parts = [part for part in grad_sample if part is not None]
        return torch.cat(parts, dim=0) if parts else None
    return grad_sample


def _clear_grad_sample_state(params: Sequence[nn.Parameter]) -> None:
    for p in params:
        p.grad = None
        if hasattr(p, "grad_sample"):
            p.grad_sample = None
        if hasattr(p, "_current_grad_sample"):
            delattr(p, "_current_grad_sample")
        if hasattr(p, "summed_grad"):
            p.summed_grad = None


def _set_expert_lora_disabled(
    sparse_layers: Sequence[SparseLayerRef],
    disabled: bool,
) -> List[Tuple[LoRALinear, bool, bool]]:
    touched: List[Tuple[LoRALinear, bool, bool]] = []
    for layer in sparse_layers:
        for expert in layer.sparse_mlp.experts:
            for module in expert.modules():
                if not isinstance(module, LoRALinear):
                    continue
                had_attr = hasattr(module, "_disable_lora")
                previous = bool(getattr(module, "_disable_lora", False))
                module._disable_lora = bool(disabled)
                touched.append((module, had_attr, previous))
    return touched


def _restore_expert_lora_disabled(states: Sequence[Tuple[LoRALinear, bool, bool]]) -> None:
    for module, had_attr, previous in states:
        if had_attr:
            module._disable_lora = previous
        elif hasattr(module, "_disable_lora"):
            delattr(module, "_disable_lora")


def _ensure_opacus_grad_samples(
    params: Sequence[nn.Parameter],
    batch_size: int,
) -> None:
    """
    Sparse MoE routing can leave an expert inactive for an entire microbatch,
    so its LoRA params never receive a backward hook from Opacus.  Fill in a
    zero grad_sample so Opacus doesn't error on the missing tensor.
    Mirrors dp_olmoe._ensure_opacus_grad_samples.
    """
    for p in params:
        if not p.requires_grad:
            continue
        if getattr(p, "grad_sample", None) is not None:
            continue
        if getattr(p, "_current_grad_sample", None) is not None:
            continue
        p.grad_sample = torch.zeros(
            (int(batch_size), *tuple(p.shape)),
            device=p.device,
            dtype=p.dtype,
        )


# ---------------------------------------------------------------------------
# Opacus privacy engine
# ---------------------------------------------------------------------------

def make_privacy_engine(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    effective_batch_size: int,
    sample_size: int,
    logical_steps: int,
    epochs: int,
    target_epsilon: float,
    target_delta: float,
    args: argparse.Namespace,
) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader, object, float]:
    if OpacusPrivacyEngine is None:
        raise RuntimeError(
            "Opacus is not installed. Install it with: pip install opacus"
        )

    if getattr(args, "noise_multiplier", None) and args.noise_multiplier > 0:
        sigma = float(args.noise_multiplier)
    else:
        sample_rate = min(1.0, float(effective_batch_size) / float(max(sample_size, 1)))
        sigma = find_noise_multiplier_prv(
            sampling_probability=sample_rate,
            num_steps=max(int(logical_steps), 1),
            target_epsilon=float(target_epsilon),
            target_delta=float(target_delta),
            eps_error=float(getattr(args, "prv_eps_error", 0.1)),
        )

    try:
        pe = OpacusPrivacyEngine(accountant="prv")
        pe._accountant_name = "prv"
    except Exception:
        pe = OpacusPrivacyEngine(accountant="rdp")
        pe._accountant_name = "rdp"

    model.train()
    # bfloat16 models: Opacus accumulates grads in float32 internally; disable
    # grad_dtype check on trainable params so the assignment doesn't raise.
    for p in model.parameters():
        if p.requires_grad and hasattr(p, "grad_dtype"):
            p.grad_dtype = None

    # Pre-wrap with strict=False so Opacus skips modules it can't handle
    # (e.g. MoEGate with weight parameter, LlamaRotaryEmbedding with buffers).
    # When make_private receives an already-wrapped AbstractGradSampleModule it
    # passes through _prepare_model without re-wrapping.
    from opacus.grad_sample import GradSampleModule
    if not isinstance(model, GradSampleModule):
        model = GradSampleModule(model, batch_first=True, loss_reduction="mean", strict=False)

    private_model, private_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=sigma,
        max_grad_norm=float(args.max_grad_norm),
        poisson_sampling=False,
    )
    # Opacus infers expected_batch_size from the physical DataLoader.  Patch it
    # to the logical batch so microbatch changes don't silently shift effective LR.
    if hasattr(private_optimizer, "expected_batch_size"):
        private_optimizer.expected_batch_size = int(effective_batch_size)
    return private_model, private_optimizer, private_loader, pe, sigma


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_train_loop(
    *,
    stage_name: str,
    model: nn.Module,
    train_ds: Dataset,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
    named_params: List[Tuple[str, nn.Parameter]],
    lr: float,
    epochs: int,
    dp: bool,
    target_epsilon: float,
    target_delta: float,
    collate_fn=None,
) -> Dict:
    if not named_params:
        raise RuntimeError(f"[{stage_name}] no trainable params selected")
    if len(train_ds) == 0:
        raise RuntimeError(f"[{stage_name}] empty training dataset")

    for _, p in named_params:
        p.requires_grad_(True)
    params = [p for _, p in named_params]
    n_trainable = sum(p.numel() for p in params)

    effective = min(args.train_batch_size, len(train_ds))
    micro = min(args.micro_batch_size, effective, len(train_ds))
    if dp:
        effective = max(micro, (effective // micro) * micro)
        accum = max(1, effective // micro)
    else:
        accum = max(1, math.ceil(effective / micro))

    # Opacus clips and adds noise in float32. AdamW state follows the BF16 LoRA
    # params, so cast only the final optimizer grad after Opacus has finished its
    # higher-precision work.
    optimizer = AdamWWithGradCast(params, lr=lr, betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay, foreach=False)
    loader = DataLoader(
        train_ds,
        batch_size=micro,
        shuffle=True,
        drop_last=dp,
        num_workers=0,
        collate_fn=collate_fn if collate_fn is not None else make_train_collate(tokenizer),
    )

    logical_steps_per_epoch = (len(loader) // accum) if dp else math.ceil(len(loader) / accum)
    logical_steps_total = int(logical_steps_per_epoch) * int(epochs)
    sample_rate = min(1.0, float(effective) / float(max(len(train_ds), 1)))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(logical_steps_total, 1),
        eta_min=1e-6,
    )

    train_model: nn.Module = model
    base_model: nn.Module = model
    pe = None
    sigma = 0.0

    if dp:
        train_model, optimizer, loader, pe, sigma = make_privacy_engine(
            model=model,
            optimizer=optimizer,
            loader=loader,
            effective_batch_size=effective,
            sample_size=len(train_ds),
            logical_steps=logical_steps_total,
            epochs=epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            args=args,
        )
        if hasattr(train_model, "_module"):
            base_model = train_model._module
        print(
            f"[{stage_name}] DP target ε={target_epsilon:.4f}  δ={target_delta:.3e}  "
            f"n={len(train_ds)}  batch={effective}  micro={micro}  "
            f"q={sample_rate:.5f}  steps={logical_steps_total}  σ={sigma:.4f}"
        )
    else:
        print(f"[{stage_name}] no-DP  n={len(train_ds)}  batch={effective}  micro={micro}")

    print(f"[{stage_name}] trainable={n_trainable:,}  lr={lr}  epochs={epochs}")
    best_loss = float("inf")
    best_state: Dict[str, torch.Tensor] = {}
    patience = 0
    t0 = time.time()
    completed_steps = 0

    def _scale_grads(divisor: int) -> None:
        if divisor <= 1:
            return
        scale = 1.0 / float(divisor)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)

    for epoch in range(epochs):
        train_model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_n = 0
        step_counter = 0
        accum_window_steps = 0

        iterator: Iterable = loader
        if tqdm and args.show_progress:
            iterator = tqdm(loader, desc=f"{stage_name} ep{epoch + 1}/{epochs}", leave=False)

        for raw_batch in iterator:
            batch, batch_indices = _prepare_model_batch(raw_batch, model=base_model, device=device)
            labels = batch.get("labels")
            outputs = _forward_with_routing_context(
                train_model,
                base_model=base_model,
                batch=batch,
                batch_indices=batch_indices,
            )

            loss = (
                outputs.loss
                if hasattr(outputs, "loss") and outputs.loss is not None
                else _fallback_causal_loss(outputs, labels)
            )
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            step_counter += 1
            accum_window_steps += 1
            bs = int(batch["input_ids"].shape[0]) if "input_ids" in batch else micro
            if dp:
                _ensure_opacus_grad_samples(params, bs)
            epoch_loss_sum += float(loss.detach().item()) * bs
            epoch_n += bs

            if dp:
                if step_counter % accum == 0:
                    # Real DP step: clip + noise + update
                    if hasattr(optimizer, "signal_skip_step"):
                        optimizer.signal_skip_step(do_skip=False)
                    optimizer.step()
                    sanitize_parameters(params)
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    completed_steps += 1
                else:
                    # Virtual step: advance Opacus state without a real update
                    if hasattr(optimizer, "signal_skip_step"):
                        optimizer.signal_skip_step(do_skip=True)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    elif hasattr(optimizer, "virtual_step"):
                        optimizer.virtual_step()
            elif step_counter % accum == 0:
                _scale_grads(accum_window_steps)
                nn.utils.clip_grad_norm_(params, float(args.max_grad_norm))
                optimizer.step()
                sanitize_parameters(params)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
                accum_window_steps = 0

        # Handle epoch tail
        tail = step_counter % accum
        if tail != 0 and step_counter > 0:
            if dp:
                # Drop incomplete DP logical batch — stepping would corrupt accounting
                optimizer.zero_grad(set_to_none=True)
                print(f"[{stage_name}] dropped incomplete DP tail ({tail} microbatches)")
            else:
                _scale_grads(accum_window_steps)
                nn.utils.clip_grad_norm_(params, float(args.max_grad_norm))
                optimizer.step()
                sanitize_parameters(params)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1

        epoch_loss = epoch_loss_sum / max(epoch_n, 1)
        print(f"[{stage_name}] epoch={epoch + 1}/{epochs}  loss={epoch_loss:.4f}  n={epoch_n}")

        if not dp:
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in base_model.state_dict().items()
                    if any(k == n for n, _ in named_params)
                }
                patience = 0
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    print(f"[{stage_name}] early stop at epoch {epoch + 1}")
                    break

    if not dp and best_state:
        base_model.load_state_dict(best_state, strict=False)
        print(f"[{stage_name}] restored best checkpoint")

    epsilon = 0.0
    spent_python: Dict = {}
    if pe is not None:
        # Recompute PRV epsilon from logical (sigma, q, T) — not from Opacus's
        # internal accountant which tracks physical microbatch steps.
        epsilon = prv_epsilon_for_fixed_steps(
            noise_multiplier=float(sigma),
            sample_rate=float(sample_rate),
            steps=int(logical_steps_total),
            target_delta=float(target_delta),
            fallback_epsilon=float(target_epsilon),
            eps_error=float(getattr(args, "prv_eps_error", 0.1)),
        )
        eps_rdp = rdp_epsilon_for_fixed_steps(
            noise_multiplier=float(sigma),
            sample_rate=float(sample_rate),
            steps=int(logical_steps_total),
            target_delta=float(target_delta),
            fallback_epsilon=float("nan"),
        )
        spent_python = {
            "eps_prv": epsilon,
            "eps_rdp": eps_rdp,
            "sigma": sigma,
            "sample_rate": sample_rate,
            "logical_steps": logical_steps_total,
            "logical_batch_size": effective,
            "micro_batch_size": micro,
            "opacus_internal_eps": float(pe.get_epsilon(float(target_delta))),
            "opacus_internal_accountant": getattr(pe, "_accountant_name", "unknown"),
            "accounting_note": "reported epsilon uses PRV on logical batches after microbatch accumulation",
        }
        print(f"[{stage_name}] actual ε={epsilon:.4f}  σ={sigma:.4f}")
        if getattr(args, "debug_privacy", False):
            print(f"[{stage_name}] privacy_spent={json.dumps(spent_python, indent=2)}")

    # Unwrap GradSampleModule so the next make_private() call on the same model
    # instance doesn't raise "trying to add hooks twice".
    if dp and hasattr(train_model, "to_standard_module"):
        train_model.to_standard_module()

    for _, p in named_params:
        p.requires_grad_(False)

    elapsed = time.time() - t0
    return {
        "stage": stage_name,
        "epsilon": epsilon,
        "epsilon_total": epsilon,
        "sigma": sigma,
        "accountant": ACCOUNTING_MODE if dp else "none",
        "target_epsilon": target_epsilon if dp else 0.0,
        "target_delta": target_delta if dp else 0.0,
        "privacy_spent": spent_python,
        "n_trainable": n_trainable,
        "n_samples": len(train_ds),
        "logical_batch_size": int(effective),
        "micro_batch_size": int(micro),
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# One-phase hybrid record-owner training
# ---------------------------------------------------------------------------

def _per_record_causal_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[..., :-1, :].detach().float()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid, 0)
    probs = F.softmax(shift_logits, dim=-1)
    true_probs = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    masked = probs.scatter(-1, safe_labels.unsqueeze(-1), 0.0)
    other_probs = masked.max(dim=-1).values
    denom = valid.sum(dim=-1).clamp(min=1)
    mean_true = (true_probs * valid).sum(dim=-1) / denom
    mean_other = (other_probs * valid).sum(dim=-1) / denom
    return mean_true, mean_true - mean_other


def _residual_weights_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> Optional[torch.Tensor]:
    mode = str(args.expert_residual_weighting).strip().lower()
    if mode == "none":
        return None
    true_probs, margins = _per_record_causal_stats(logits, labels)
    if mode == "prob":
        weights = 1.0 - true_probs
    elif mode == "margin":
        weights = torch.where(
            margins < float(args.expert_residual_margin_threshold),
            torch.ones_like(margins),
            torch.full_like(margins, float(args.expert_residual_small_weight)),
        )
    else:
        raise ValueError(f"Unknown --expert_residual_weighting={mode!r}")
    return weights.clamp(min=float(args.expert_residual_min_weight), max=1.0)


def train_one_phase_hybrid_ours(
    *,
    model: nn.Module,
    processor,
    train_ds: Dataset,
    dev_samples: Sequence[VQASample],
    sparse_layers: Sequence[SparseLayerRef],
    collate_fn,
    args: argparse.Namespace,
    device: torch.device,
    zero_shot: Dict,
) -> Dict:
    from opacus.grad_sample import GradSampleModule

    layers = select_sparse_layers_by_args(sparse_layers, args)
    epochs = int(args.ours_epochs if args.ours_epochs > 0 else args.finetune_epochs)
    if epochs <= 0:
        raise ValueError("--ours_epochs or --finetune_epochs must be positive")
    if args.train_batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("--train_batch_size and --micro_batch_size must be positive")

    n_records = len(train_ds)
    sample_rate = float(min(args.train_batch_size, n_records)) / float(max(n_records, 1))
    updates_per_epoch = max(1, int(math.ceil(1.0 / sample_rate)))
    total_updates = updates_per_epoch * epochs
    micro = min(int(args.micro_batch_size), n_records)
    alternating = args.alternating_expert_steps > 0
    shared_steps = max(1, int(args.alternating_shared_steps)) if alternating else 0
    expert_steps = int(args.alternating_expert_steps) if alternating else 0
    if alternating:
        window = shared_steps + expert_steps
        full_windows, remainder = divmod(total_updates, window)
        planned_shared_updates = full_windows * shared_steps + min(remainder, shared_steps)
        planned_expert_updates = full_windows * expert_steps + max(0, remainder - shared_steps)
    else:
        planned_shared_updates = total_updates
        planned_expert_updates = total_updates

    if args.no_dp:
        budget = {
            "adjacency_factor": 1.0 if args.adjacency == "add_remove" else 2.0,
            "eps_shared": 0.0,
            "eps_experts_total": 0.0,
            "eps_expert": 0.0,
            "eps_layer": 0.0,
            "delta_shared": 0.0,
            "delta_experts_total": 0.0,
            "delta_expert": 0.0,
            "delta_layer": 0.0,
        }
        sigma_shared = sigma_expert = 0.0
    else:
        budget = resolve_ours_privacy_budget(args, len(layers))
        sigma_shared = _stacked_noise_multiplier(
            sample_rate=sample_rate,
            steps=max(1, planned_shared_updates),
            target_epsilon=budget["eps_shared"],
            target_delta=budget["delta_shared"],
            args=args,
        )
        sigma_expert = _stacked_noise_multiplier(
            sample_rate=sample_rate,
            steps=max(1, planned_expert_updates),
            target_epsilon=budget["eps_expert"],
            target_delta=budget["delta_expert"],
            args=args,
        )

    print("\n" + "=" * 88)
    print("[OURS ONE-PHASE] hybrid shared + record-owner expert DP training")
    print(f"[OURS ONE-PHASE] selected sparse layers = {[layer.layer_id for layer in layers]}")
    print(
        f"[OURS ONE-PHASE] epochs={epochs} updates_per_epoch={updates_per_epoch} "
        f"total_updates={total_updates} q={sample_rate:.6f} "
        f"expected/update={min(args.train_batch_size, n_records)} micro={micro}"
    )
    print(
        f"[OURS ONE-PHASE] expert_objective={args.expert_objective} "
        f"expert_update_scale_mode={args.expert_update_scale_mode} "
        f"clip_scope={args.ours_clip_scope} expert_lr_multiplier={args.expert_lr_multiplier:g}"
    )
    if args.no_dp:
        print("[OURS ONE-PHASE PRIVACY PLAN] no-DP mode")
    else:
        print(
            f"[OURS ONE-PHASE PRIVACY PLAN] eps_shared={budget['eps_shared']:.4f} "
            f"eps_experts_total={budget['eps_experts_total']:.4f} "
            f"eps_layer={budget['eps_layer']:.4f} expert_eps={budget['eps_expert']:.4f} "
            f"sigma_shared={sigma_shared:.4f} sigma_expert={sigma_expert:.4f}"
        )

    freeze_all(model)
    assignment_cache: Dict[int, torch.Tensor] = {}
    print("[OURS ONE-PHASE] computing fixed record-owner assignments (counts are not logged)")
    for layer in layers:
        owners = construct_record_assignments(
            model=model,
            train_ds=train_ds,
            tokenizer=getattr(processor, "tokenizer", processor),
            target_layer=layer,
            batch_size=args.assignment_batch_size,
            device=device,
            collate_fn=collate_fn,
            show_progress=args.show_progress,
        )
        assignment_cache[int(layer.layer_id)] = torch.tensor(owners, dtype=torch.long)
        print(f"[OURS ONE-PHASE] layer={layer.layer_id} assignments ready")

    shared_named = _select_deepseekv2_shared_params(model)
    if not shared_named:
        raise RuntimeError("[OURS ONE-PHASE] no shared attention LoRA parameters selected")
    shared_params = [p for _, p in shared_named]
    for p in shared_params:
        p.requires_grad_(True)
    shared_optimizer = AdamWWithGradCast(
        shared_params,
        lr=float(args.lr_attention),
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        foreach=False,
    )
    shared_scheduler = torch.optim.lr_scheduler.LambdaLR(shared_optimizer, lr_lambda=lambda _: 1.0)
    shared_buffers = {p: torch.zeros_like(p) for p in shared_params}

    expert_states: Dict[Tuple[int, int], Dict] = {}
    all_named_params: List[Tuple[str, nn.Parameter]] = list(shared_named)
    for layer in layers:
        num_experts = _num_experts(layer.sparse_mlp)
        for expert_id in range(num_experts):
            named = _select_deepseekv2_expert_lora_params(model, layer, expert_id)
            if not named:
                raise RuntimeError(
                    f"[OURS ONE-PHASE] no LoRA params for layer={layer.layer_id} expert={expert_id}"
                )
            params = [p for _, p in named]
            for p in params:
                p.requires_grad_(True)
            optimizer = AdamWWithGradCast(
                params,
                lr=float(args.lr),
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
                foreach=False,
            )
            expert_states[(int(layer.layer_id), expert_id)] = {
                "named_params": named,
                "params": params,
                "optimizer": optimizer,
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0),
                "buffers": {p: torch.zeros_like(p) for p in params},
                "updates": 0,
                "num_experts": num_experts,
            }
            all_named_params.extend(named)

    unique_params: List[nn.Parameter] = []
    seen_ids: set[int] = set()
    for _, p in all_named_params:
        if id(p) not in seen_ids:
            unique_params.append(p)
            seen_ids.add(id(p))
    for p in unique_params:
        if hasattr(p, "grad_dtype"):
            p.grad_dtype = None

    n_shared = sum(p.numel() for p in shared_params)
    n_expert = sum(p.numel() for state in expert_states.values() for p in state["params"])
    print(f"[OURS ONE-PHASE] shared_trainable={n_shared:,} expert_trainable={n_expert:,}")

    saved_layers = set(model._record_route_layer_ids)
    saved_assignments = dict(model._record_route_fixed_assignments)
    model._record_route_layer_ids = {int(layer.layer_id) for layer in layers}
    model._record_route_fixed_assignments = dict(assignment_cache)
    train_model: Optional[nn.Module] = None
    base_model = model
    epoch_losses: List[float] = []
    epoch_evals: List[Dict] = []
    started = time.time()

    def _global_factors(rows: torch.Tensor) -> Optional[torch.Tensor]:
        if args.ours_clip_scope != "global" or rows.numel() == 0:
            return None
        norm_sq = torch.zeros(rows.numel(), device=device, dtype=torch.float32)
        for p in unique_params:
            grad_sample = _grad_sample_as_tensor(p)
            if grad_sample is not None:
                selected = grad_sample.index_select(0, rows.to(grad_sample.device))
                norm_sq += selected.detach().float().flatten(1).pow(2).sum(dim=1)
        return torch.clamp(float(args.max_grad_norm) / (norm_sq.sqrt() + 1e-6), max=1.0)

    def _accumulate(
        params,
        buffers,
        rows: torch.Tensor,
        factors: Optional[torch.Tensor],
        row_weights: Optional[torch.Tensor] = None,
    ) -> None:
        if rows.numel() == 0:
            return
        norm_sq = torch.zeros(rows.numel(), device=device, dtype=torch.float32)
        usable = []
        for p in params:
            grad_sample = _grad_sample_as_tensor(p)
            if grad_sample is None:
                continue
            selected = grad_sample.index_select(0, rows.to(grad_sample.device))
            if row_weights is not None:
                weights = row_weights.to(device=selected.device, dtype=selected.dtype)
                while weights.dim() < selected.dim():
                    weights = weights.unsqueeze(-1)
                selected = selected * weights
            usable.append((p, selected))
            norm_sq += selected.detach().float().flatten(1).pow(2).sum(dim=1)
        if not usable:
            return
        clip = factors
        if clip is None:
            clip = torch.clamp(float(args.max_grad_norm) / (norm_sq.sqrt() + 1e-6), max=1.0)
        for p, selected in usable:
            view = clip.to(device=selected.device, dtype=selected.dtype)
            while view.dim() < selected.dim():
                view = view.unsqueeze(-1)
            buffers[p].add_((selected * view).sum(dim=0).to(buffers[p].dtype))

    def _accumulate_expert(
        layer_id: int,
        expert_id: int,
        batch_indices: torch.Tensor,
        factors: Optional[torch.Tensor],
        residual_weights: Optional[torch.Tensor] = None,
    ) -> None:
        owners = assignment_cache[layer_id].index_select(0, batch_indices.detach().cpu().long())
        rows = (owners == expert_id).nonzero(as_tuple=False).flatten().to(device)
        if factors is not None:
            factors = factors.index_select(0, rows.to(factors.device))
        if residual_weights is not None:
            residual_weights = residual_weights.index_select(0, rows.to(residual_weights.device))
        state = expert_states[(layer_id, expert_id)]
        _accumulate(state["params"], state["buffers"], rows, factors, residual_weights)

    def _debug_assert_non_owner_grad_samples_zero(batch_indices: torch.Tensor) -> None:
        if not getattr(args, "debug_privacy", False):
            return
        for layer in layers:
            layer_id = int(layer.layer_id)
            owners = assignment_cache[layer_id].index_select(
                0,
                batch_indices.detach().cpu().long(),
            ).to(device=device)
            for expert_id in range(_num_experts(layer.sparse_mlp)):
                non_owner_rows = (owners != int(expert_id)).nonzero(as_tuple=False).flatten()
                if non_owner_rows.numel() == 0:
                    continue
                state = expert_states[(layer_id, int(expert_id))]
                worst = 0.0
                worst_name = ""
                for name, p in state["named_params"]:
                    grad_sample = _grad_sample_as_tensor(p)
                    if grad_sample is None:
                        continue
                    if grad_sample.size(0) != owners.numel():
                        raise RuntimeError(
                            f"[DEBUG PRIVACY] grad_sample batch mismatch for layer {layer_id} "
                            f"expert {expert_id} param {name}: got {grad_sample.size(0)}, "
                            f"expected {owners.numel()}"
                        )
                    row_norms = (
                        grad_sample.index_select(0, non_owner_rows.to(grad_sample.device))
                        .detach()
                        .float()
                        .flatten(1)
                        .norm(dim=1)
                    )
                    if row_norms.numel() == 0:
                        continue
                    value = float(row_norms.max().item())
                    if value > worst:
                        worst = value
                        worst_name = name
                if worst > 1e-6:
                    raise RuntimeError(
                        f"[DEBUG PRIVACY] non-owner grad_sample is nonzero for layer {layer_id} "
                        f"expert {expert_id}: max_norm={worst:.6e} param={worst_name}"
                    )

    def _step(params, optimizer, scheduler, buffers, sigma: float, denom: float, scale: float = 1.0) -> None:
        for p in params:
            grad = buffers[p]
            if not args.no_dp:
                noise = torch.randn(grad.shape, device=grad.device, dtype=torch.float32)
                grad = grad + noise.to(grad.dtype) * (float(sigma) * float(args.max_grad_norm))
            p.grad = (grad * float(scale) / float(max(denom, 1.0))).to(p.dtype)
        optimizer.step()
        sanitize_parameters(params)
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        for p in params:
            buffers[p].zero_()
            p.grad = None

    def _shared_only_logits(batch: Dict[str, torch.Tensor], indices: torch.Tensor) -> torch.Tensor:
        was_training = base_model.training
        touched = _set_expert_lora_disabled(layers, True)
        try:
            base_model.eval()
            with torch.no_grad():
                outputs = _forward_with_routing_context(
                    base_model,
                    base_model=base_model,
                    batch=batch,
                    batch_indices=indices,
                )
            return outputs.logits.detach()
        finally:
            _restore_expert_lora_disabled(touched)
            if was_training:
                base_model.train()

    def _train_loss(
        batch: Dict[str, torch.Tensor],
        indices: torch.Tensor,
        *,
        shared_only: bool = False,
        shared_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        touched = _set_expert_lora_disabled(layers, True) if shared_only else []
        try:
            outputs = _forward_with_routing_context(
                train_model,
                base_model=base_model,
                batch=batch,
                batch_indices=indices,
            )
        finally:
            _restore_expert_lora_disabled(touched)
        if shared_logits is None or args.expert_objective != "residual_logit":
            return _outputs_loss(outputs, batch["labels"])
        residual_logits = shared_logits.detach() + (outputs.logits - shared_logits)
        return _causal_loss_from_logits(residual_logits, batch["labels"])

    try:
        train_model = GradSampleModule(
            model,
            batch_first=True,
            loss_reduction="mean",
            strict=False,
        )
        base_model = train_model._module
        for epoch in range(epochs):
            train_model.train()
            epoch_loss_sum = 0.0
            epoch_n = 0
            iterator: Iterable = range(updates_per_epoch)
            if tqdm and args.show_progress:
                iterator = tqdm(iterator, desc=f"OURS ONE-PHASE ep{epoch + 1}/{epochs}", leave=False)
            for update_idx in iterator:
                if sample_rate >= 1.0:
                    selected = torch.randperm(n_records)
                else:
                    selected = (torch.rand(n_records) < sample_rate).nonzero(as_tuple=False).flatten()
                    if selected.numel() > 0:
                        selected = selected[torch.randperm(selected.numel())]
                global_update = epoch * updates_per_epoch + update_idx
                if alternating:
                    schedule_position = global_update % (shared_steps + expert_steps)
                    train_shared = schedule_position < shared_steps
                    train_experts = not train_shared
                else:
                    train_shared = train_experts = True

                selected_indices = selected.tolist()
                for start in range(0, len(selected_indices), micro):
                    chunk = selected_indices[start:start + micro]
                    if not chunk:
                        continue
                    raw_batch = collate_fn([train_ds[int(idx)] for idx in chunk])
                    batch, indices = _prepare_model_batch(raw_batch, model=base_model, device=device)
                    if indices is None:
                        raise RuntimeError("One-phase training requires stable sample indices")
                    rows = torch.arange(len(chunk), device=device, dtype=torch.long)
                    split_backward = (
                        alternating
                        or args.expert_objective == "residual_logit"
                        or args.expert_residual_weighting != "none"
                    )
                    micro_losses: List[float] = []

                    if split_backward and train_shared:
                        _clear_grad_sample_state(unique_params)
                        loss = _train_loss(batch, indices, shared_only=True)
                        if torch.isfinite(loss):
                            loss.backward()
                            _aggregate_routed_expert_grad_samples(layers, len(chunk))
                            _ensure_opacus_grad_samples(unique_params, len(chunk))
                            _accumulate(shared_params, shared_buffers, rows, _global_factors(rows))
                            micro_losses.append(float(loss.detach().item()))

                    if split_backward and train_experts:
                        shared_logits = None
                        if args.expert_objective == "residual_logit" or args.expert_residual_weighting != "none":
                            shared_logits = _shared_only_logits(batch, indices)
                        residual_weights = (
                            _residual_weights_from_logits(shared_logits, batch["labels"], args)
                            if shared_logits is not None
                            else None
                        )
                        _clear_grad_sample_state(unique_params)
                        loss = _train_loss(batch, indices, shared_logits=shared_logits)
                        if torch.isfinite(loss):
                            loss.backward()
                            _aggregate_routed_expert_grad_samples(layers, len(chunk))
                            _ensure_opacus_grad_samples(unique_params, len(chunk))
                            _debug_assert_non_owner_grad_samples_zero(indices)
                            factors = _global_factors(rows)
                            for layer in layers:
                                for expert_id in range(_num_experts(layer.sparse_mlp)):
                                    _accumulate_expert(
                                        int(layer.layer_id),
                                        expert_id,
                                        indices,
                                        factors,
                                        residual_weights,
                                    )
                            micro_losses.append(float(loss.detach().item()))

                    if not split_backward:
                        _clear_grad_sample_state(unique_params)
                        loss = _train_loss(batch, indices)
                        if torch.isfinite(loss):
                            loss.backward()
                            _aggregate_routed_expert_grad_samples(layers, len(chunk))
                            _ensure_opacus_grad_samples(unique_params, len(chunk))
                            _debug_assert_non_owner_grad_samples_zero(indices)
                            factors = _global_factors(rows)
                            _accumulate(shared_params, shared_buffers, rows, factors)
                            for layer in layers:
                                for expert_id in range(_num_experts(layer.sparse_mlp)):
                                    _accumulate_expert(int(layer.layer_id), expert_id, indices, factors)
                            micro_losses.append(float(loss.detach().item()))

                    _clear_grad_sample_state(unique_params)
                    if micro_losses:
                        epoch_loss_sum += sum(micro_losses) / len(micro_losses) * len(chunk)
                        epoch_n += len(chunk)

                if train_shared:
                    _step(
                        shared_params,
                        shared_optimizer,
                        shared_scheduler,
                        shared_buffers,
                        sigma_shared,
                        float(args.train_batch_size),
                    )
                if train_experts:
                    for layer in layers:
                        for expert_id in range(_num_experts(layer.sparse_mlp)):
                            state = expert_states[(int(layer.layer_id), expert_id)]
                            denominator = float(args.train_batch_size)
                            if args.expert_update_scale_mode == "expected_owner":
                                denominator /= float(max(1, state["num_experts"]))
                            _step(
                                state["params"],
                                state["optimizer"],
                                state["scheduler"],
                                state["buffers"],
                                sigma_expert,
                                denominator,
                                float(args.expert_lr_multiplier),
                            )
                            state["updates"] += 1

                for p in unique_params:
                    p.grad = None

            epoch_loss = epoch_loss_sum / max(epoch_n, 1)
            epoch_losses.append(float(epoch_loss))
            active_layers = set(base_model._record_route_layer_ids)
            base_model._record_route_layer_ids.clear()
            try:
                dev = deepseekv2_evaluate_generation(
                    model=base_model,
                    processor=processor,
                    samples=dev_samples,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    limit=args.eval_max_samples,
                    show_progress=args.show_progress,
                    progress_desc=f"OURS eval ep{epoch + 1}/{epochs}",
                )
            finally:
                base_model._record_route_layer_ids.update(active_layers)
            print(
                f"[OURS ONE-PHASE] epoch={epoch + 1}/{epochs} loss={epoch_loss:.4f} "
                f"dev_EM={dev['exact_match']:.4f} dev_n={dev['n_samples']}"
            )
            epoch_evals.append({
                "epoch": epoch + 1,
                "train_loss": float(epoch_loss),
                "dev_exact_match": float(dev["exact_match"]),
                "dev_n": int(dev["n_samples"]),
            })
    finally:
        if train_model is not None and hasattr(train_model, "to_standard_module"):
            train_model.to_standard_module()
        model._record_route_layer_ids = saved_layers
        model._record_route_fixed_assignments = saved_assignments
        model._active_batch_indices = None
        _clear_grad_sample_state(unique_params)
        for _, p in all_named_params:
            p.requires_grad_(False)

    if args.no_dp:
        eps_shared_actual = eps_expert_actual = 0.0
    else:
        eps_shared_actual = _stacked_privacy_epsilon(
            noise_multiplier=sigma_shared,
            sample_rate=sample_rate,
            steps=max(1, planned_shared_updates),
            target_delta=budget["delta_shared"],
            fallback_epsilon=budget["eps_shared"],
            args=args,
        )
        eps_expert_actual = _stacked_privacy_epsilon(
            noise_multiplier=sigma_expert,
            sample_rate=sample_rate,
            steps=max(1, planned_expert_updates),
            target_delta=budget["delta_expert"],
            fallback_epsilon=budget["eps_expert"],
            args=args,
        )
    layer_eps = (
        2.0 * eps_expert_actual
        if args.adjacency == "replace_one"
        else eps_expert_actual
    )
    eps_total = eps_shared_actual + len(layers) * layer_eps
    final_dev = epoch_evals[-1] if epoch_evals else {
        "dev_exact_match": zero_shot["exact_match"],
        "dev_n": zero_shot["n_samples"],
    }
    elapsed = time.time() - started
    return {
        "method": "one_phase_hybrid_dp_moe",
        "zero_shot": zero_shot,
        "final_dev": final_dev,
        "selected_layer_ids": [int(layer.layer_id) for layer in layers],
        "training": {
            "epochs": epochs,
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
            "planned_shared_updates": planned_shared_updates,
            "planned_expert_updates": planned_expert_updates,
            "sample_rate": sample_rate,
            "logical_batch_size": int(args.train_batch_size),
            "micro_batch_size": micro,
            "epoch_losses": epoch_losses,
            "epoch_evals": epoch_evals,
        },
        "privacy_plan": {
            **budget,
            "sigma_shared": sigma_shared,
            "sigma_expert": sigma_expert,
            "accounting": "shared sequential + expert parallel within layer",
        },
        "privacy_summary": {
            "epsilon_shared_actual": eps_shared_actual,
            "epsilon_expert_layer_actual": layer_eps,
            "epsilon_total_actual": eps_total,
            "epsilon_total_target": float(args.epsilon_total),
            "adjacency": args.adjacency,
        },
        "dev_exact_match": float(final_dev["dev_exact_match"]),
        "dev_n": int(final_dev["dev_n"]),
        "epsilon_total": float(eps_total),
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Experiment dispatcher
# ---------------------------------------------------------------------------

def run_finetune_experiment(
    *,
    exp: str,
    model: nn.Module,
    processor,
    train_samples: Sequence[VQASample],
    dev_samples: Sequence[VQASample],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict:
    train_ds = DeepSeekV2TrainDataset(train_samples, processor, max_length=args.max_length)
    tokenizer = getattr(processor, "tokenizer", processor)
    collate_fn = lambda batch: _deepseekv2_collate(batch, pad_id=processor.pad_id)  # noqa: E731
    sparse_layers = get_sparse_layers(model)

    model._record_route_layer_ids = set()
    model._forced_record_expert_by_layer = {}
    model._record_route_fixed_assignments = {}
    model._active_batch_indices = None
    model._record_route_weight_mode = args.record_route_weight_mode
    model._active_attention_mask = None

    if sparse_layers:
        patch_sparse_mlp_for_record_routing(model, sparse_layers)

    zero = deepseekv2_evaluate_generation(
        model=model, processor=processor, samples=dev_samples, device=device,
        max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0,
        limit=args.eval_max_samples, pred_path="",
        show_progress=args.show_progress, progress_desc=f"{exp} zero-shot eval",
    )
    print(f"[{exp}] zero-shot EM={zero.get('exact_match', 0.0):.4f}  n={zero.get('n_samples', 0)}")
    result: Dict = {"experiment": exp, "zero_shot": zero}

    # --- LoRA injection (no ExpertLinear; Opacus handles per-sample grads) ---
    # Exclude vision encoder: its .attn. layers match _is_attention_linear_name
    # but they must stay frozen; LoRA on them would cause Opacus buffer errors.
    def _lang_attn_filter(name: str) -> bool:
        if not _is_attention_linear_name(name):
            return False
        # Exclude generic vision encoder names (vision_tower, image_tower, etc.)
        if _is_vision_encoder_param_name(name):
            return False
        # Exclude DeepSeek-VL2's SigLIP tower which is named 'vision' (not 'vision_tower')
        if name.startswith("vision.") or ".vision." in name:
            return False
        # Exclude multimodal projector
        if _is_multimodal_projector_param_name(name):
            return False
        return True

    if exp in ("upper_bound", "naive_dp", "ours"):
        n_attn = apply_lora_with_filter(
            model, filter_fn=_lang_attn_filter,
            r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
            use_expert_linear=False,
        )
        n_expert = apply_lora_with_filter(
            model, filter_fn=_is_expert_linear_name,
            r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
            use_expert_linear=False,
        )
        print(f"[{exp}] LoRA injected: attention={n_attn}  expert={n_expert}")

    # -----------------------------------------------------------------------
    # upper_bound
    # -----------------------------------------------------------------------
    if exp == "upper_bound":
        freeze_all(model)
        named = _select_deepseekv2_language_lora_params(model)
        enable_only(named)
        stats = run_train_loop(
            collate_fn=collate_fn,
            stage_name="UPPER_BOUND", model=model, train_ds=train_ds,
            tokenizer=tokenizer, device=device, args=args,
            named_params=named, lr=args.lr, epochs=args.finetune_epochs,
            dp=False, target_epsilon=0.0, target_delta=0.0,
        )
        dev = deepseekv2_evaluate_generation(
            model=model, processor=processor, samples=dev_samples, device=device,
            max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0,
            limit=args.eval_max_samples, pred_path="",
            show_progress=args.show_progress, progress_desc="upper_bound eval",
        )
        stats.update({"dev_exact_match": dev.get("exact_match", 0.0), "dev_n": dev.get("n_samples", 0)})
        result["result"] = stats
        return result

    # -----------------------------------------------------------------------
    # upper_bound_no_lora
    # -----------------------------------------------------------------------
    if exp == "upper_bound_no_lora":
        freeze_all(model)
        named = select_full_params(model, freeze_vision=args.freeze_vision_in_full_finetune)
        enable_only(named)
        stats = run_train_loop(
            collate_fn=collate_fn,
            stage_name="UPPER_BOUND_NO_LORA", model=model, train_ds=train_ds,
            tokenizer=tokenizer, device=device, args=args,
            named_params=named, lr=args.lr_full_finetune, epochs=args.finetune_epochs,
            dp=False, target_epsilon=0.0, target_delta=0.0,
        )
        dev = deepseekv2_evaluate_generation(
            model=model, processor=processor, samples=dev_samples, device=device,
            max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0,
            limit=args.eval_max_samples, pred_path="",
            show_progress=args.show_progress, progress_desc="upper_bound_no_lora eval",
        )
        stats.update({"dev_exact_match": dev.get("exact_match", 0.0), "dev_n": dev.get("n_samples", 0)})
        result["result"] = stats
        return result

    # -----------------------------------------------------------------------
    # naive_dp  (standard Opacus DP over all LoRA params)
    # -----------------------------------------------------------------------
    if exp == "naive_dp":
        if sparse_layers:
            patch_sparse_mlp_for_naive_dp(model, sparse_layers)
        freeze_all(model)
        named = _select_deepseekv2_language_lora_params(model)
        enable_only(named)
        stats = run_train_loop(
            collate_fn=collate_fn,
            stage_name="NAIVE_DP", model=model, train_ds=train_ds,
            tokenizer=tokenizer, device=device, args=args,
            named_params=named, lr=args.lr, epochs=args.finetune_epochs,
            dp=not args.no_dp, target_epsilon=args.epsilon_total, target_delta=args.delta,
        )
        dev = deepseekv2_evaluate_generation(
            model=model, processor=processor, samples=dev_samples, device=device,
            max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0,
            limit=args.eval_max_samples, pred_path="",
            show_progress=args.show_progress, progress_desc="naive_dp eval",
        )
        stats.update({"dev_exact_match": dev.get("exact_match", 0.0), "dev_n": dev.get("n_samples", 0)})
        result["result"] = stats
        return result

    # -----------------------------------------------------------------------
    # ours  (one-phase hybrid record-owner DP-MoE)
    # -----------------------------------------------------------------------
    if exp == "ours":
        if not sparse_layers:
            raise RuntimeError(
                "No sparse MoE layers detected. "
                "Record-routing DP-MoE requires sparse experts."
            )
        result["result"] = train_one_phase_hybrid_ours(
            model=model,
            processor=processor,
            train_ds=train_ds,
            dev_samples=dev_samples,
            sparse_layers=sparse_layers,
            collate_fn=collate_fn,
            args=args,
            device=device,
            zero_shot=zero,
        )
        return result

    raise ValueError(f"Unknown experiment: {exp!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepSeek-VL2-Tiny DP fine-tuning with Opacus")
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--mode", choices=["infer", "eval", "finetune"], default="finetune")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--image_path", default="")
    p.add_argument("--prompt", default="")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)

    p.add_argument("--train_jsonl", default="")
    p.add_argument("--dev_jsonl", default="")
    p.add_argument("--dataset_jsonl", default="")
    p.add_argument("--image_root", default="")
    p.add_argument("--train_max_samples", type=int, default=-1)
    p.add_argument("--eval_max_samples", type=int, default=-1)
    p.add_argument("--output_dir", default="./deepseekv2_tiny_opacus_output")
    p.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--show_progress", action="store_true")

    p.add_argument("--experiment", default="ours",
                   choices=["upper_bound", "upper_bound_no_lora", "naive_dp", "ours"])
    p.add_argument("--max_length", type=int, default=512)

    # DP budget
    p.add_argument("--epsilon_total", type=float, default=8.0)
    p.add_argument("--delta", type=float, default=-1.0)
    p.add_argument("--no_dp", action="store_true")
    p.add_argument("--noise_multiplier", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--adjacency", choices=["add_remove", "replace_one"], default="add_remove")
    p.add_argument("--prv_eps_error", type=float, default=0.1)

    # Training
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--finetune_epochs", type=int, default=1)
    p.add_argument("--ours_epochs", type=int, default=-1,
                   help="One-phase hybrid epochs; <=0 falls back to --finetune_epochs.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_attention", type=float, default=2e-4)
    p.add_argument("--lr_full_finetune", type=float, default=2e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--early_stop_patience", type=int, default=3)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # MoE / routing
    p.add_argument("--start_layer", type=int, default=0)
    p.add_argument("--num_layers_to_train", type=int, default=-1)
    p.add_argument("--min_expert_size", type=int, default=1)
    p.add_argument("--assignment_batch_size", type=int, default=8)
    p.add_argument("--record_route_weight_mode", default="router_prob")
    p.add_argument("--freeze_vision_in_full_finetune", action="store_true")
    p.add_argument("--eval_each_layer", action="store_true")
    p.add_argument("--debug_privacy", action="store_true")
    p.add_argument("--epsilon_shared_ratio", type=float, default=0.9)
    p.add_argument("--expert_residual_weighting", choices=["none", "prob", "margin"], default="none")
    p.add_argument("--expert_residual_margin_threshold", type=float, default=0.2)
    p.add_argument("--expert_residual_small_weight", type=float, default=0.0)
    p.add_argument("--expert_residual_min_weight", type=float, default=0.05)
    p.add_argument("--expert_update_scale_mode", choices=["batch", "expected_owner"],
                   default="expected_owner")
    p.add_argument("--ours_clip_scope", choices=["role", "global"], default="role")
    p.add_argument("--expert_lr_multiplier", type=float, default=5.0)
    p.add_argument("--expert_objective", choices=["ce", "residual_logit"], default="residual_logit")
    p.add_argument("--alternating_shared_steps", type=int, default=1)
    p.add_argument("--alternating_expert_steps", type=int, default=1)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if args.delta <= 0:
        args.delta = 1e-5

    print(f"[main] loading model {args.model_name!r} ...")
    processor, model = load_vlm(args.model_name, device, args.dtype, args.trust_remote_code)

    if args.mode == "infer":
        run_single_infer(args, processor, model, device)
        return

    if args.mode == "eval":
        run_eval_mode(args, processor, model, device)
        return

    # --- finetune ---
    if not args.train_jsonl or not args.dev_jsonl:
        raise ValueError("--train_jsonl and --dev_jsonl are required for finetune mode")

    train_samples = load_jsonl_dataset(args.train_jsonl, args.image_root, require_answer=True)
    dev_samples = load_jsonl_dataset(args.dev_jsonl, args.image_root, require_answer=False)
    if args.train_max_samples > 0:
        train_samples = train_samples[: args.train_max_samples]
    print(f"[main] train={len(train_samples)}  dev={len(dev_samples)}")

    experiments = [e.strip() for e in args.experiment.split(",") if e.strip()]

    print(
        f"\n{'=' * 80}\n[EXPERIMENT] {', '.join(experiments)}\n{'=' * 80}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    all_results: List[Dict] = []

    for exp in experiments:
        result = run_finetune_experiment(
            exp=exp, model=model, processor=processor,
            train_samples=train_samples, dev_samples=dev_samples,
            args=args, device=device,
        )
        all_results.append(result)

        log_path = os.path.join(args.output_dir, f"log_{exp}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[{exp}] saved → {log_path}")

        dev_em = (result.get("result") or {}).get("dev_exact_match", 0.0)
        eps = (result.get("result") or {}).get("epsilon_total", 0.0)
        print(f"[{exp}] dev_EM={dev_em:.4f}  ε_total={eps:.4f}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[main] summary → {summary_path}")


if __name__ == "__main__":
    main()
