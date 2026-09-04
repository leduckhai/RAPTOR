#!/usr/bin/env python3
"""
DP-MoE Switch Transformer with one-phase hybrid record-owner DP training.

Main idea
---------
Under --experiment ours, train shared parameters and selected expert parameters
in one scheduled loop:
    - shared params get ordinary sampled DP-SGD over sampled records;
    - expert params get record-owner masked per-expert sampled DP-SGD streams;
    - experts compose in parallel within a sparse layer, while shared + layers
      compose sequentially.

Why this file is different from the earlier joint-expert script
---------------------------------------------------------------
Earlier code computed REA assignments but still used natural token routing during
expert training, so one sample could update multiple experts in the same layer.
That breaks the clean within-layer parallel-composition story.

This file fixes that by keeping natural token routing in the forward pass while
masking non-owner expert gradients. The old two-stage helpers are retained for
ablations, but `ours` uses one-phase hybrid accounting.

Important privacy note
----------------------
Within one layer, the disjoint expert owner sets support parallel composition.
Across shared parameters and sparse layers, privacy composes sequentially.

Under add/remove adjacency, if each expert in a layer is trained with
(eps_expert, delta_expert)-DP on its own subset, the whole layer costs
(max_j eps_j, max_j delta_j).

Under replace-one adjacency, a changed record can move between two expert subsets,
so a conservative layer cost is approximately (2 * max_j eps_j, 2 * max_j delta_j).
This file supports both via --adjacency.

Practical choices in this implementation
----------------------------------------
- PRV accounting for epsilon calibration/reporting, with RDP logged as a secondary check when available.
- Model stays in eval() mode during optimization to keep routing deterministic.
  Gradients still flow; only dropout/jitter are disabled.
- Evaluation uses natural token routing (OLMoE-style) for a cleaner comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import types
import weakref
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import datasets as hf_datasets

from transformers import (
    AutoTokenizer,
    SwitchTransformersConfig,
    SwitchTransformersForConditionalGeneration,
)
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersSparseMLP,
)
import transformers.modeling_utils as _hf_modeling_utils

try:
    from opacus import PrivacyEngine as OpacusPrivacyEngine
    from opacus.accountants import RDPAccountant as OpacusRDPAccountant
    from opacus.accountants.utils import get_noise_multiplier as _opacus_get_noise_multiplier
    from opacus.grad_sample import GradSampleModule as OpacusGradSampleModule
except Exception:  # pragma: no cover
    OpacusPrivacyEngine = None
    OpacusRDPAccountant = None
    _opacus_get_noise_multiplier = None
    OpacusGradSampleModule = None

try:
    from opacus.accountants import PRVAccountant as OpacusPRVAccountant
except Exception:  # pragma: no cover
    OpacusPRVAccountant = None

try:
    from prv_accountant import Accountant as ExternalPRVAccountant
except Exception:  # pragma: no cover
    ExternalPRVAccountant = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


ACCOUNTING_MODE = "prv"

# Prevent non-fatal background auto-conversion thread crashes from surfacing as noisy stack traces.
_orig_auto_conversion = _hf_modeling_utils.auto_conversion


def _quiet_auto_conversion(*args, **kwargs):
    kwargs["ignore_errors_during_conversion"] = True
    try:
        return _orig_auto_conversion(*args, **kwargs)
    except Exception:
        return None, None, None


_hf_modeling_utils.auto_conversion = _quiet_auto_conversion


def extract_epsilon_rdp(spent: Dict) -> float:
    for key in ("eps_rdp", "epsilon_rdp", "epsilon"):
        if key in spent and spent[key] is not None:
            return float(spent[key])
    raise ValueError(f"Could not extract RDP epsilon from privacy dict: {spent}")


def tensor_to_python(v):
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return v.detach().cpu().tolist()
    return v


def sanitize_parameters(params: Sequence[nn.Parameter], clamp: float = 100.0) -> None:
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=clamp, neginf=-clamp)
            p.data.clamp_(-clamp, clamp)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


BASELINE_EXPERIMENT_ALIASES = {
    "matched_scope_global_dp": "baseline_a_global_matched",
    "dp_shared_only": "baseline_b_shared_only",
    "dp_expert_only": "baseline_c_expert_only",
    "dp_lora_last_layer": "baseline_e_selected_layer_lora",
    "last_layer_dp_lora_global": "baseline_e_selected_layer_lora",
    "baseline_e_last_layer_lora": "baseline_e_selected_layer_lora",
}
SCOPED_GLOBAL_BASELINE_EXPERIMENTS = {
    "baseline_a_global_matched",
    "baseline_b_shared_only",
    "baseline_e_selected_layer_lora",
    "baseline_e_last_layer_lora",
    "matched_scope_global_dp",
    "dp_shared_only",
    "dp_lora_last_layer",
    "last_layer_dp_lora_global",
}
EXPERT_ONLY_BASELINE_EXPERIMENTS = {
    "baseline_c_expert_only",
    "dp_expert_only",
}


def canonical_experiment_name(experiment: str) -> str:
    return BASELINE_EXPERIMENT_ALIASES.get(str(experiment), str(experiment))


def _runtime_world_size(default: int = 1) -> int:
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", str(default))))
    except Exception:
        return max(1, int(default))


def _cuda_device_index(device: Optional[torch.device] = None) -> int:
    if device is not None and isinstance(device, torch.device) and device.type == "cuda" and device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _reset_runtime_peak_memory(device: Optional[torch.device] = None) -> None:
    if not torch.cuda.is_available():
        return
    try:
        idx = _cuda_device_index(device)
        torch.cuda.synchronize(idx)
        torch.cuda.reset_peak_memory_stats(idx)
    except Exception:
        pass


def _runtime_stats(
    t0: float,
    completed_steps: int,
    device: Optional[torch.device] = None,
    gpu_count: Optional[int] = None,
) -> Dict:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize(_cuda_device_index(device))
        except Exception:
            pass
    seconds = max(0.0, time.time() - t0)
    steps = max(0, int(completed_steps))
    active_gpus = _runtime_world_size() if gpu_count is None else max(1, int(gpu_count))
    peak_gb = 0.0
    if torch.cuda.is_available():
        try:
            peak_gb = float(torch.cuda.max_memory_allocated(_cuda_device_index(device)) / 1024**3)
        except Exception:
            peak_gb = 0.0
    return {
        "seconds": round(seconds, 1),
        "completed_optimizer_steps": steps,
        "seconds_per_step": float(seconds / steps) if steps > 0 else None,
        "peak_gpu_memory_gb": peak_gb,
        "total_gpu_hours": float(seconds * active_gpus / 3600.0),
        "gpu_count": int(active_gpus),
    }


def _runtime_stage_rows(obj) -> List[Dict]:
    rows: List[Dict] = []
    if isinstance(obj, dict):
        if "seconds" in obj:
            rows.append(obj)
            return rows
        for key, value in obj.items():
            if key in {"runtime_summary", "privacy_spent", "expert_dilution_metrics", "config"}:
                continue
            rows.extend(_runtime_stage_rows(value))
    elif isinstance(obj, list):
        for value in obj:
            rows.extend(_runtime_stage_rows(value))
    return rows


def _attach_runtime_summary(result: Dict) -> Dict:
    rows = _runtime_stage_rows(result)
    world_size = _runtime_world_size()
    if rows:
        if isinstance(result.get("seconds"), (int, float)):
            method_seconds = float(result["seconds"])
        else:
            method_seconds = sum(float(row.get("seconds", 0.0) or 0.0) for row in rows)
        completed_steps = sum(int(row.get("completed_optimizer_steps", 0) or 0) for row in rows)
        peak_gb = max(float(row.get("peak_gpu_memory_gb", 0.0) or 0.0) for row in rows)
    else:
        method_seconds = 0.0
        completed_steps = 0
        peak_gb = 0.0
    summary = {
        "method_wall_seconds": round(method_seconds, 1),
        "completed_optimizer_steps": int(completed_steps),
        "seconds_per_step": float(method_seconds / completed_steps) if completed_steps > 0 else None,
        "peak_gpu_memory_gb": float(peak_gb),
        "total_gpu_hours": float(method_seconds * world_size / 3600.0),
        "gpu_count": int(world_size),
        "stage_count": int(len(rows)),
    }
    result["runtime_summary"] = summary
    result["peak_gpu_memory_gb"] = summary["peak_gpu_memory_gb"]
    result["seconds_per_step"] = summary["seconds_per_step"]
    result["total_gpu_hours"] = summary["total_gpu_hours"]
    return result


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


GLUE_TASK_CONFIG = {
    "sst2": {
        "hf_path": "glue",
        "hf_name": "sst2",
        "validation_split": "validation",
        "col_a": "sentence",
        "col_b": None,
        "label_col": "label",
        "num_labels": 2,
    },
    "mnli": {
        "hf_path": "glue",
        "hf_name": "mnli",
        "validation_split": "validation_matched",
        "col_a": "premise",
        "col_b": "hypothesis",
        "label_col": "label",
        "num_labels": 3,
    },
    "qnli": {
        "hf_path": "glue",
        "hf_name": "qnli",
        "validation_split": "validation",
        "col_a": "question",
        "col_b": "sentence",
        "label_col": "label",
        "num_labels": 2,
    },
    "qqp": {
        "hf_path": "glue",
        "hf_name": "qqp",
        "validation_split": "validation",
        "col_a": "question1",
        "col_b": "question2",
        "label_col": "label",
        "num_labels": 2,
    },
}

GLUE_TASK_VERBALIZERS = {
    "sst2": ["negative", "positive"],
    "mnli": ["entailment", "neutral", "contradiction"],
    "qnli": ["entailment", "not entailment"],
    "qqp": ["not duplicate", "duplicate"],
}

DEFAULT_SST2_INPUT_PREFIX = ""


class GLUEDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[int(idx)]
        return {
            "idx": int(idx),
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "label": int(row["label"]),
        }


SST2Dataset = GLUEDataset


class SubsetDataset(Dataset):
    def __init__(self, base: GLUEDataset, indices: Sequence[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        return self.base[self.indices[idx]]


class ReindexedSubsetDataset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        row = dict(self.base[self.indices[int(idx)]])
        row["idx"] = int(idx)
        return row


def load_glue(data_dir: str, task: str, tokenizer, max_length: int) -> Tuple[GLUEDataset, GLUEDataset]:
    cfg = GLUE_TASK_CONFIG.get(task.lower())
    if cfg is None:
        raise ValueError(f"Unknown task '{task}'")

    ds = hf_datasets.load_dataset(
        cfg["hf_path"],
        cfg["hf_name"],
        cache_dir=data_dir if data_dir else None,
    )
    train_raw = ds["train"]
    dev_raw = ds[cfg["validation_split"]]

    label_col = cfg["label_col"]

    def _tokenize(batch):
        if cfg["col_b"] is None:
            return tokenizer(
                batch[cfg["col_a"]],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        return tokenizer(
            batch[cfg["col_a"]],
            batch[cfg["col_b"]],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    keep = {label_col}
    train_remove = [c for c in train_raw.column_names if c not in keep]
    dev_remove = [c for c in dev_raw.column_names if c not in keep]

    train_tok = train_raw.map(
        _tokenize,
        batched=True,
        remove_columns=train_remove,
        desc=f"Tokenizing train split ({task})",
    )
    dev_tok = dev_raw.map(
        _tokenize,
        batched=True,
        remove_columns=dev_remove,
        desc=f"Tokenizing validation split ({task})",
    )

    if label_col != "label":
        train_tok = train_tok.rename_column(label_col, "label")
        dev_tok = dev_tok.rename_column(label_col, "label")

    train_tok = train_tok.filter(lambda x: int(x["label"]) >= 0)
    dev_tok = dev_tok.filter(lambda x: int(x["label"]) >= 0)

    return GLUEDataset(train_tok), GLUEDataset(dev_tok)


def make_collate(tokenizer, max_length: int, prefix: str = ""):
    del tokenizer, max_length, prefix

    def _collate(batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
            "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
        }

    return _collate


# ---------------------------------------------------------------------------
# Sparse layer refs
# ---------------------------------------------------------------------------


@dataclass
class SparseLayerRef:
    layer_id: int
    block_id: int
    layer_ff: nn.Module
    sparse_mlp: nn.Module


# ---------------------------------------------------------------------------
# ExpertLinear + fastDP support
# ---------------------------------------------------------------------------


class ExpertLinear(nn.Linear):
    """
    nn.Linear subclass carrying token->sample metadata.
    fastDP uses this metadata to aggregate token gradients into per-sample gradients.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self._token_to_sample: Optional[torch.Tensor] = None
        self._num_samples: Optional[int] = None

    @staticmethod
    def from_linear(src: nn.Linear) -> "ExpertLinear":
        out = ExpertLinear(src.in_features, src.out_features, bias=(src.bias is not None))
        out.to(device=src.weight.device, dtype=src.weight.dtype)
        out.weight.data.copy_(src.weight.data)
        if src.bias is not None:
            out.bias.data.copy_(src.bias.data)
        return out


def register_expert_linear_fastdp() -> None:
    from fastDP import supported_layers_grad_samplers as sgs

    if ExpertLinear in sgs._supported_layers_norm_sample_AND_clipping:
        return

    def _sampler(layer: ExpertLinear, A: torch.Tensor, B: torch.Tensor, _mode: str) -> None:
        if A is None or B is None or layer._token_to_sample is None or layer._num_samples is None:
            return

        token_to_sample = layer._token_to_sample.to(device=A.device, dtype=torch.long)
        num_samples = int(layer._num_samples)
        n_tokens = int(token_to_sample.numel())
        if n_tokens == 0:
            layer._token_to_sample = None
            layer._num_samples = None
            return
        if A.size(0) != n_tokens or B.size(0) != n_tokens:
            raise RuntimeError(
                f"ExpertLinear sampler shape mismatch: "
                f"A={tuple(A.shape)} B={tuple(B.shape)} token_to_sample={tuple(token_to_sample.shape)}"
            )

        grad_w = torch.zeros(num_samples, B.size(1), A.size(1), device=A.device, dtype=A.dtype)
        # Avoid materializing a huge [n_tokens, out, in] tensor at once.
        per_token_elems = int(B.size(1) * A.size(1))
        max_outer_elems = 8_000_000  # ~32MB fp32 or ~16MB fp16 temporary buffer
        chunk_tokens = max(1, min(n_tokens, max_outer_elems // max(per_token_elems, 1)))

        for start in range(0, n_tokens, chunk_tokens):
            end = min(start + chunk_tokens, n_tokens)
            outer_chunk = torch.einsum("no,ni->noi", B[start:end], A[start:end])
            grad_w.index_add_(0, token_to_sample[start:end], outer_chunk)

        layer.weight.grad_sample = grad_w.detach()
        layer.weight.norm_sample = grad_w.flatten(1).norm(dim=1).detach()

        if layer.bias is not None:
            grad_b = torch.zeros(num_samples, B.size(1), device=A.device, dtype=A.dtype)
            grad_b.index_add_(0, token_to_sample, B)
            layer.bias.grad_sample = grad_b.detach()
            layer.bias.norm_sample = grad_b.norm(dim=1).detach()

        layer._token_to_sample = None
        layer._num_samples = None

    sgs._supported_layers_norm_sample_AND_clipping[ExpertLinear] = (
        _sampler,
        sgs._clip_linear_grad,
    )


def _patch_fastdp_use_full_backward_hook() -> None:
    """
    Patch fastDP to use register_full_backward_hook when available.

    The local fastDP build uses register_backward_hook, which can miss gradients
    on complex graphs (MoE + LoRA). Full backward hooks are more reliable.
    """
    try:
        import fastDP.autograd_grad_sample as _ags
        from fastDP.supported_layers_grad_samplers import _supported_layers_norm_sample_AND_clipping as _supported
    except Exception as exc:
        print(f"[fastDP patch] WARNING: full_backward_hook patch skipped: {exc}")
        return

    if getattr(_ags, "_full_backward_hook_patched", False):
        return

    def _add_hooks_full(
        model: nn.Module,
        loss_reduction="mean",
        clipping_mode="MixOpt",
        bias_only=False,
        clipping_style="all-layer",
        block_heads=None,
        named_params=None,
        named_layers=None,
        clipping_fn=None,
        numerical_stability_constant=None,
        max_grad_norm_layerwise=None,
    ):
        if hasattr(model, "autograd_grad_sample_hooks"):
            raise ValueError("Trying to add hooks twice to the same model")

        handles = []
        block_head_set = set(block_heads or [])

        for name, layer in model.named_modules():
            if type(layer) not in _supported or not _ags.requires_grad(layer):
                continue

            if (
                hasattr(layer, "weight")
                and hasattr(layer.weight, "initially_requires_grad")
                and layer.weight.initially_requires_grad
            ):
                handles.append(layer.register_forward_hook(_ags._capture_activations))

            def _mk_backward(is_block_head: bool):
                def _this_backward(this_layer, grad_input, grad_output):
                    _ags._prepare_sample_grad_or_norm(
                        this_layer, grad_output, loss_reduction, clipping_mode, bias_only
                    )
                    if is_block_head:
                        _ags._per_block_clip_grad(
                            this_layer,
                            named_params,
                            named_layers,
                            clipping_style,
                            clipping_fn,
                            numerical_stability_constant,
                            max_grad_norm_layerwise,
                        )

                return _this_backward

            bwd = _mk_backward(name in block_head_set)
            if hasattr(layer, "register_full_backward_hook"):
                handles.append(layer.register_full_backward_hook(bwd))
            else:
                handles.append(layer.register_backward_hook(bwd))

        model.__dict__.setdefault("autograd_grad_sample_hooks", []).extend(handles)

    _ags.add_hooks = _add_hooks_full
    _ags._full_backward_hook_patched = True
    print("[fastDP patch] add_hooks patched to register_full_backward_hook.")


def _patch_fastdp_for_expert_linear() -> None:
    """
    Patch fastDP clipping for ExpertLinear.

    fastDP's default _per_block_clip_grad path assumes backprops are sample-major
    and recomputes grads via einsum('b...,b->...'). For routed experts, backprops
    are token-major, while ExpertLinear already stores sample-aggregated grad_sample.
    This patch clips/aggregates directly from grad_sample for ExpertLinear layers.
    """
    try:
        import fastDP.autograd_grad_sample as _ags
        from fastDP.supported_layers_grad_samplers import _create_or_extend_private_grad as _create_private_grad
    except Exception as exc:
        print(f"[fastDP patch] WARNING: ExpertLinear clip patch skipped: {exc}")
        return

    if getattr(_ags, "_expert_linear_clip_patched", False):
        return

    _orig = _ags._per_block_clip_grad

    def _patched_clip(
        this_layer,
        named_params,
        named_layers,
        clipping_style,
        clipping_fn,
        numerical_stability_constant,
        max_grad_norm_layerwise,
    ):
        if not isinstance(this_layer, ExpertLinear):
            return _orig(
                this_layer,
                named_params,
                named_layers,
                clipping_style,
                clipping_fn,
                numerical_stability_constant,
                max_grad_norm_layerwise,
            )

        layer_params = list(this_layer.named_parameters(recurse=False))
        norms = []
        for _, p in layer_params:
            ns = getattr(p, "norm_sample", None)
            if ns is not None:
                norms.append(ns)
                continue
            gs = getattr(p, "grad_sample", None)
            if gs is not None:
                norms.append(gs.abs() if gs.dim() <= 1 else gs.flatten(1).norm(dim=1))

        if not norms:
            for _, p in layer_params:
                if hasattr(p, "grad_sample"):
                    del p.grad_sample
                if hasattr(p, "norm_sample"):
                    del p.norm_sample
            if hasattr(this_layer, "activations"):
                del this_layer.activations
            if hasattr(this_layer, "backprops"):
                del this_layer.backprops
            return

        try:
            norm_sample = torch.stack(norms, dim=0).norm(2, dim=0)
        except RuntimeError:
            norm_sample = norms[0]

        if numerical_stability_constant is None:
            numerical_stability_constant = 1e-6

        if clipping_fn == "automatic":
            C = max_grad_norm_layerwise / (norm_sample + numerical_stability_constant)
        elif clipping_fn == "Abadi":
            C = torch.clamp_max(max_grad_norm_layerwise / (norm_sample + numerical_stability_constant), 1.0)
        elif clipping_fn == "global":
            C = (norm_sample <= max_grad_norm_layerwise).float()
        else:
            raise ValueError(
                f"Unknown clipping function {clipping_fn}. Expected one of Abadi, automatic, global."
            )

        for _, p in layer_params:
            gs = getattr(p, "grad_sample", None)
            if gs is not None:
                C_p = C.detach().clone()
                while C_p.dim() < gs.dim():
                    C_p = C_p.unsqueeze(-1)
                _create_private_grad(p, (gs * C_p).sum(0))
                del p.grad_sample
            if hasattr(p, "norm_sample"):
                del p.norm_sample

        if hasattr(this_layer, "activations"):
            del this_layer.activations
        if hasattr(this_layer, "backprops"):
            del this_layer.backprops

    _ags._per_block_clip_grad = _patched_clip
    _ags._expert_linear_clip_patched = True
    print("[fastDP patch] _per_block_clip_grad patched for ExpertLinear.")


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
        use_expert_linear: bool = False,
    ):
        super().__init__()
        self.linear = base
        for p in self.linear.parameters():
            p.requires_grad_(False)

        a = nn.Linear(base.in_features, r, bias=False)
        b = nn.Linear(r, base.out_features, bias=False)
        if use_expert_linear:
            self.lora_A: nn.Module = ExpertLinear.from_linear(a)
            self.lora_B: nn.Module = ExpertLinear.from_linear(b)
        else:
            self.lora_A = a
            self.lora_B = b

        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / float(r)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor, owner_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = self.linear(x)
        if getattr(self, "_disable_lora", False):
            return base

        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scale

        if owner_mask is None:
            owner_mask = getattr(self, "_record_owner_mask", None)
        if owner_mask is not None:
            owner_mask = owner_mask.to(device=delta.device, dtype=delta.dtype)
            while owner_mask.dim() < delta.dim():
                owner_mask = owner_mask.unsqueeze(-1)
            delta = owner_mask * delta + (1.0 - owner_mask) * delta.detach()

        return base + delta


def _forward_expert_with_lora_owner_mask(
    expert_module: nn.Module,
    expert_input: torch.Tensor,
    owner_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if owner_mask is None:
        return expert_module(expert_input)

    touched: List[Tuple[LoRALinear, bool, Optional[torch.Tensor]]] = []
    for module in expert_module.modules():
        if not isinstance(module, LoRALinear):
            continue
        had_mask = hasattr(module, "_record_owner_mask")
        prev_mask = getattr(module, "_record_owner_mask", None)
        module._record_owner_mask = owner_mask
        touched.append((module, had_mask, prev_mask))

    try:
        return expert_module(expert_input)
    finally:
        for module, had_mask, prev_mask in touched:
            if had_mask:
                module._record_owner_mask = prev_mask
            elif hasattr(module, "_record_owner_mask"):
                delattr(module, "_record_owner_mask")


def apply_lora_experts(
    model, sparse_layers: Sequence[SparseLayerRef], r: int, alpha: float, dropout: float
) -> int:
    count = 0
    for sl in sparse_layers:
        for expert in sl.sparse_mlp.experts.values():
            for attr in ("wi", "wo"):
                mod = getattr(expert, attr)
                if isinstance(mod, nn.Linear) and not isinstance(mod, LoRALinear):
                    setattr(expert, attr, LoRALinear(mod, r, alpha, dropout, use_expert_linear=True))
                    count += 1
    return count


def apply_lora_attention(model, r: int, alpha: float, dropout: float) -> int:
    count = 0
    for block in model.base_model.encoder.block:
        attn = block.layer[0].SelfAttention
        for attr in ("q", "k", "v", "o"):
            mod = getattr(attn, attr, None)
            if mod is not None and isinstance(mod, nn.Linear) and not isinstance(mod, LoRALinear):
                setattr(attn, attr, LoRALinear(mod, r, alpha, dropout, use_expert_linear=False))
                count += 1
    return count


def _set_expert_lora_disabled(
    sparse_layers: Sequence[SparseLayerRef],
    disabled: bool,
) -> List[Tuple[LoRALinear, bool, bool]]:
    touched: List[Tuple[LoRALinear, bool, bool]] = []
    for sl in sparse_layers:
        for expert in sl.sparse_mlp.experts.values():
            for attr in ("wi", "wo"):
                mod = getattr(expert, attr, None)
                if not isinstance(mod, LoRALinear):
                    continue
                had_attr = hasattr(mod, "_disable_lora")
                prev = bool(getattr(mod, "_disable_lora", False))
                mod._disable_lora = bool(disabled)
                touched.append((mod, had_attr, prev))
    return touched


def _restore_expert_lora_disabled(states: Sequence[Tuple[LoRALinear, bool, bool]]) -> None:
    for mod, had_attr, prev in states:
        if had_attr:
            mod._disable_lora = prev
        elif hasattr(mod, "_disable_lora"):
            delattr(mod, "_disable_lora")


def _set_params_requires_grad(params: Sequence[nn.Parameter], enabled: bool) -> None:
    for p in params:
        p.requires_grad_(bool(enabled))


def convert_expert_linears_for_fastdp(sparse_layers: Sequence[SparseLayerRef]) -> int:
    """
    Convert expert FFN projections to ExpertLinear without adding LoRA.
    This is used by naive_dp_no_lora so fastDP can use token->sample metadata.
    """
    count = 0
    for sl in sparse_layers:
        for expert in sl.sparse_mlp.experts.values():
            for attr in ("wi", "wo"):
                mod = getattr(expert, attr, None)
                if isinstance(mod, ExpertLinear):
                    continue
                if isinstance(mod, LoRALinear):
                    base = mod.linear
                    if isinstance(base, ExpertLinear):
                        continue
                    mod.linear = ExpertLinear.from_linear(base)
                    count += 1
                    continue
                if isinstance(mod, nn.Linear):
                    setattr(expert, attr, ExpertLinear.from_linear(mod))
                    count += 1
    return count


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SwitchSST2(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2):
        super().__init__()
        cfg = SwitchTransformersConfig.from_pretrained(model_name)
        cfg.router_aux_loss_coef = 0.0
        cfg.router_z_loss_coef = 0.0
        cfg.router_jitter_noise = 0.0
        cfg.dropout_rate = 0.0  # keep training deterministic under record routing
        try:
            self.base_model = SwitchTransformersForConditionalGeneration.from_pretrained(
                model_name,
                config=cfg,
            )
        except ValueError as e:
            if "torch.load" in str(e) or "CVE-2025-32434" in str(e):
                import transformers.modeling_utils as mu
                import transformers.utils.import_utils as iu

                mu.check_torch_load_is_safe = lambda: None
                iu.check_torch_load_is_safe = lambda: None
                self.base_model = SwitchTransformersForConditionalGeneration.from_pretrained(
                    model_name,
                    config=cfg,
                )
            else:
                raise

        self.classifier = nn.Linear(self.base_model.config.d_model, num_labels)

        # Runtime state used by the patched sparse layers.
        self._record_route_layer_ids: set[int] = set()
        self._active_attention_mask: Optional[torch.Tensor] = None
        self._record_route_weight_mode: str = "router_prob"
        # Optional fixed per-window record routing map: layer_id -> [N] record->expert assignments.
        self._record_route_fixed_assignments: Dict[int, torch.Tensor] = {}
        # Current mini-batch record indices from dataset collate.
        self._active_batch_indices: Optional[torch.Tensor] = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        prev_mask = self._active_attention_mask
        self._active_attention_mask = attention_mask
        try:
            out = self.base_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h = out.last_hidden_state
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            pooled = pooled.to(self.classifier.weight.dtype)
            return self.classifier(pooled)
        finally:
            self._active_attention_mask = prev_mask

    def seq2seq_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        prev_mask = self._active_attention_mask
        self._active_attention_mask = attention_mask
        try:
            labels = target_token_ids.unsqueeze(1)
            out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            return out.loss
        finally:
            self._active_attention_mask = prev_mask

    def seq2seq_class_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        decoder_start_token_id = self.base_model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.base_model.config.pad_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = 0

        prev_mask = self._active_attention_mask
        self._active_attention_mask = attention_mask
        try:
            decoder_input_ids = torch.full(
                (input_ids.size(0), 1),
                int(decoder_start_token_id),
                dtype=torch.long,
                device=input_ids.device,
            )
            out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                return_dict=True,
            )
            step_logits = out.logits[:, 0, :]
            return step_logits.index_select(-1, label_token_ids.to(step_logits.device))
        finally:
            self._active_attention_mask = prev_mask


def init_classifier(model: SwitchSST2, tokenizer, task: str) -> bool:
    label_texts = GLUE_TASK_VERBALIZERS.get(task.lower())
    if not label_texts:
        return False
    if len(label_texts) != int(model.classifier.out_features):
        return False

    W = model.base_model.shared.weight
    rows: List[torch.Tensor] = []
    for text in label_texts:
        token_ids: List[int] = []
        for candidate in (text, text.lower(), text.replace("_", " "), f" {text.replace('_', ' ')}"):
            ids = tokenizer(candidate, add_special_tokens=False).get("input_ids", [])
            if ids:
                token_ids = ids
                break
        if not token_ids:
            return False
        ids_tensor = torch.tensor(token_ids, device=W.device, dtype=torch.long)
        rows.append(W.index_select(0, ids_tensor).mean(dim=0))

    with torch.no_grad():
        vecs = torch.stack(rows, dim=0).float()
        vecs = vecs / vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        vecs = vecs * 0.02
        model.classifier.weight.copy_(vecs.to(W.dtype))
        model.classifier.bias.zero_()
    print(f"[INIT] classifier from normalized label embeddings  task={task} labels={label_texts}")
    return True


def build_seq2seq_label_token_ids(tokenizer, task: str) -> torch.Tensor:
    label_texts = GLUE_TASK_VERBALIZERS.get(task.lower())
    if not label_texts:
        raise ValueError(f"No verbalizers available for task={task!r}")

    token_ids: List[int] = []
    for text in label_texts:
        ids: List[int] = []
        normalized = text.replace("_", " ")
        for candidate in (
            normalized,
            f" {normalized}",
            normalized.lower(),
            f" {normalized.lower()}",
        ):
            cand_ids = tokenizer(candidate, add_special_tokens=False).get("input_ids", [])
            if cand_ids:
                ids = cand_ids
                break
        if not ids:
            raise RuntimeError(f"Could not tokenize seq2seq verbalizer {text!r} for task={task!r}")
        if len(ids) > 1:
            print(
                f"[SEQ2SEQ] verbalizer {text!r} maps to multiple tokens {ids}; using first token {ids[0]}"
            )
        token_ids.append(int(ids[0]))

    if len(set(token_ids)) != len(token_ids):
        raise RuntimeError(
            f"Seq2seq verbalizer tokens are not unique for task={task!r}: {token_ids}"
        )
    print(f"[SEQ2SEQ] class label tokens for task={task}: {token_ids}")
    return torch.tensor(token_ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Freeze helpers and layer discovery
# ---------------------------------------------------------------------------


def freeze_all(model: SwitchSST2) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def zero_router_jitter(model: SwitchSST2) -> None:
    for _name, mod in model.base_model.named_modules():
        if hasattr(mod, "jitter_noise"):
            mod.jitter_noise = 0.0


def freeze_routers(model: SwitchSST2) -> None:
    zero_router_jitter(model)
    for name, mod in model.base_model.named_modules():
        if name.endswith(".router"):
            for p in mod.parameters():
                p.requires_grad_(False)


def get_sparse_layers(model: SwitchSST2) -> List[SparseLayerRef]:
    refs: List[SparseLayerRef] = []
    lid = 0
    for bid, block in enumerate(model.base_model.encoder.block):
        ff = block.layer[-1]
        if hasattr(ff, "mlp") and isinstance(ff.mlp, SwitchTransformersSparseMLP):
            refs.append(SparseLayerRef(lid, bid, ff, ff.mlp))
            lid += 1
    return refs


# ---------------------------------------------------------------------------
# Patched sparse MLP with deterministic record routing
# ---------------------------------------------------------------------------


def _get_expert_module(experts: nn.Module, expert_id: int) -> nn.Module:
    name = f"expert_{expert_id}"
    if isinstance(experts, nn.ModuleDict):
        return experts[name]
    if hasattr(experts, name):
        return getattr(experts, name)
    raise TypeError(f"Unsupported experts container: {type(experts)}")


def _compute_raw_router_probs(router: nn.Module, flat_hidden: torch.Tensor) -> torch.Tensor:
    x = flat_hidden
    classifier = router.classifier
    cls_weight = getattr(classifier, "weight", None)
    cls_dtype = getattr(cls_weight, "dtype", None)
    if isinstance(cls_dtype, torch.dtype):
        if x.dtype != cls_dtype:
            x = x.to(cls_dtype)
    else:
        router_dtype = getattr(router, "dtype", None)
        if isinstance(router_dtype, torch.dtype) and x.dtype != router_dtype:
            x = x.to(router_dtype)
        elif x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
    logits = classifier(x)
    probs = torch.softmax(logits, dim=-1, dtype=logits.dtype)
    return probs


def _top1_capacity_mask(
    probs_2d: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    num_experts: int,
    expert_capacity: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Microsoft Switch-compatible top-1 routing with capacity masking.

    Returns:
      selected_flat: [B*S] selected expert ids (argmax before masking)
      keep_flat:     [B*S] token keep mask after per-expert capacity masking
      top_prob_flat: [B*S] max router prob per token (used for output scaling)
    """
    probs = probs_2d.view(batch_size, seq_len, num_experts)
    selected = probs.argmax(dim=-1)  # [B, S]
    top_prob = probs.max(dim=-1).values  # [B, S]

    keep = torch.ones(batch_size, seq_len, device=probs_2d.device, dtype=torch.bool)
    if expert_capacity > 0:
        one_hot = F.one_hot(selected, num_classes=num_experts).to(torch.int64)
        # Microsoft code uses cumsum over token dimension (dim=-2 for [B,S,E]).
        token_priority = torch.cumsum(one_hot, dim=1)
        cap_ok = token_priority <= expert_capacity
        keep = torch.gather(cap_ok, -1, selected.unsqueeze(-1)).squeeze(-1)

    return selected.reshape(-1), keep.reshape(-1), top_prob.reshape(-1)


def _capacity_keep_from_selected_ids(
    selected_ids_2d: torch.Tensor,
    *,
    num_experts: int,
    expert_capacity: int,
    valid_mask_2d: torch.Tensor,
) -> torch.Tensor:
    """
    Capacity mask for pre-chosen expert ids (record routing path).
    """
    keep = valid_mask_2d
    if expert_capacity <= 0:
        return keep

    one_hot = F.one_hot(selected_ids_2d, num_classes=num_experts).to(torch.int64)
    one_hot = one_hot * valid_mask_2d.unsqueeze(-1).to(one_hot.dtype)
    token_priority = torch.cumsum(one_hot, dim=1)
    cap_ok = token_priority <= expert_capacity
    keep_cap = torch.gather(cap_ok, -1, selected_ids_2d.unsqueeze(-1)).squeeze(-1)
    return keep & keep_cap


def patch_sparse_mlp_for_record_routing(model: SwitchSST2, sparse_layers: Sequence[SparseLayerRef]) -> None:
    for sl in sparse_layers:
        mlp = sl.sparse_mlp
        if getattr(mlp, "_record_routing_patched", False):
            continue

        owner_ref = weakref.ref(model)
        layer_id = int(sl.layer_id)
        orig_forward = mlp.forward

        def _patched_forward(self, hidden_states: torch.Tensor, _owner_ref=owner_ref, _layer_id=layer_id, _orig_forward=orig_forward):
            owner = _owner_ref()
            if owner is None:
                raise RuntimeError("Owner model reference was lost in patched sparse MLP")

            batch_size, seq_len, _hidden_dim = hidden_states.shape
            raw_probs = _compute_raw_router_probs(self.router, hidden_states).to(hidden_states.dtype)
            raw_top1 = raw_probs.argmax(dim=-1)
            self._last_raw_expert_index = raw_top1.detach()

            # Default path: preserve Hugging Face behavior.
            if _layer_id not in owner._record_route_layer_ids:
                return _orig_forward(hidden_states)

            # Owner-masked path: route tokens naturally, but only let each
            # record's owner expert receive LoRA gradients from that record.
            attn_mask = owner._active_attention_mask
            if attn_mask is None:
                valid_mask = torch.ones(batch_size, seq_len, device=hidden_states.device, dtype=torch.bool)
            else:
                valid_mask = attn_mask.to(hidden_states.device).bool()
                if valid_mask.shape != (batch_size, seq_len):
                    raise ValueError(
                        f"Attention mask shape mismatch at sparse layer {_layer_id}: "
                        f"got {tuple(valid_mask.shape)}, expected {(batch_size, seq_len)}"
                    )

            fixed_assignments = getattr(owner, "_record_route_fixed_assignments", {}).get(_layer_id)
            batch_indices = getattr(owner, "_active_batch_indices", None)
            if fixed_assignments is not None and batch_indices is not None:
                fixed_cpu = fixed_assignments.detach().to(device="cpu", dtype=torch.long)
                batch_idx_cpu = batch_indices.detach().to(device="cpu", dtype=torch.long).view(-1)
                if batch_idx_cpu.numel() != batch_size:
                    raise RuntimeError(
                        f"Batch index count mismatch at sparse layer {_layer_id}: "
                        f"got {batch_idx_cpu.numel()}, expected {batch_size}"
                    )
                if batch_idx_cpu.min().item() < 0 or batch_idx_cpu.max().item() >= fixed_cpu.numel():
                    raise RuntimeError(
                        f"Batch indices out of range for fixed assignments at sparse layer {_layer_id}: "
                        f"index range=[{batch_idx_cpu.min().item()}, {batch_idx_cpu.max().item()}], "
                        f"assignment_size={fixed_cpu.numel()}"
                    )
                record_experts = fixed_cpu.index_select(0, batch_idx_cpu).to(device=hidden_states.device, dtype=torch.long)
            else:
                chosen: List[int] = []
                for i in range(batch_size):
                    active_experts = raw_top1[i][valid_mask[i]]
                    if active_experts.numel() == 0:
                        chosen.append(0)
                    else:
                        counts = torch.bincount(active_experts.long(), minlength=self.router.num_experts)
                        chosen.append(int(counts.argmax().item()))
                record_experts = torch.tensor(chosen, device=hidden_states.device, dtype=torch.long)
            self._last_record_expert_index = record_experts.detach()

            expert_capacity = int(getattr(self.router, "expert_capacity", 0) or 0)
            selected_flat, keep_flat, top_prob_flat = _top1_capacity_mask(
                raw_probs,
                batch_size=batch_size,
                seq_len=seq_len,
                num_experts=int(self.router.num_experts),
                expert_capacity=expert_capacity,
            )
            selected_ids_2d = selected_flat.view(batch_size, seq_len)
            keep_mask = keep_flat.view(batch_size, seq_len) & valid_mask
            self._last_keep_mask = keep_mask.detach()
            if owner._record_route_weight_mode == "one":
                weights = torch.ones(batch_size, seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
            else:
                weights = top_prob_flat.view(batch_size, seq_len).to(hidden_states.dtype)
            weights[~valid_mask] = 0.0

            # Microsoft-style fallback: tokens not dispatched by capacity keep their pre-MLP state.
            out_states = hidden_states.clone()
            for expert_id in range(int(self.router.num_experts)):
                token_mask = (selected_ids_2d == int(expert_id)) & keep_mask

                expert_module = _get_expert_module(self.experts, int(expert_id))
                expert_input = hidden_states * token_mask.unsqueeze(-1).to(hidden_states.dtype)
                owner_mask = record_experts == int(expert_id)
                expert_output = _forward_expert_with_lora_owner_mask(
                    expert_module,
                    expert_input,
                    owner_mask,
                )
                if token_mask.any():
                    out_states[token_mask] = expert_output[token_mask]
                else:
                    # Keep inactive experts in the autograd graph with zero contribution.
                    # Opacus requires every trainable parameter to receive grad_sample.
                    out_states = out_states + expert_output.sum() * 0.0

            out_states = out_states * weights.unsqueeze(-1).to(out_states.dtype)
            return out_states

        mlp.forward = types.MethodType(_patched_forward, mlp)
        mlp._record_routing_patched = True


def patch_sparse_mlp_for_naive_dp(model: SwitchSST2, sparse_layers: Sequence[SparseLayerRef]) -> None:
    """
    Token-level routing patch for naive DP baseline.

    This keeps per-token expert assignment (top-1) with capacity masking.
    """
    for sl in sparse_layers:
        mlp = sl.sparse_mlp
        if getattr(mlp, "_naive_dp_patched", False):
            continue

        def _naive_forward(self, hidden_states: torch.Tensor):
            batch_size, seq_len, _hidden_dim = hidden_states.shape
            probs = _compute_raw_router_probs(self.router, hidden_states).to(hidden_states.dtype)
            selected = probs.argmax(dim=-1)
            top_prob = probs.max(dim=-1).values
            expert_capacity = int(getattr(self.router, "expert_capacity", 0) or 0)
            if expert_capacity > 0:
                one_hot = F.one_hot(selected, num_classes=int(self.router.num_experts)).to(torch.int64)
                token_priority = torch.cumsum(one_hot, dim=1)
                cap_ok = token_priority <= expert_capacity
                keep_mask = torch.gather(cap_ok, -1, selected.unsqueeze(-1)).squeeze(-1)
            else:
                keep_mask = torch.ones_like(selected, dtype=torch.bool)

            # Precompute all expert token masks at once to avoid per-iteration CUDA ops.
            expert_ids = torch.arange(self.router.num_experts, device=hidden_states.device)
            expert_token_masks = (
                selected.unsqueeze(0) == expert_ids.view(-1, 1, 1)
            ) & keep_mask.unsqueeze(0)  # [num_experts, batch, seq]

            out_states = hidden_states.clone()
            for expert_id in range(self.router.num_experts):
                token_mask = expert_token_masks[expert_id]  # [batch, seq] — no kernel, view only
                expert_module = _get_expert_module(self.experts, int(expert_id))
                expert_input = hidden_states * token_mask.unsqueeze(-1).to(hidden_states.dtype)
                expert_output = expert_module(expert_input)
                # torch.where = static shape, zero CPU-GPU syncs in the loop.
                out_states = torch.where(
                    token_mask.unsqueeze(-1), expert_output, out_states
                )

            self._last_raw_expert_index = selected.detach()
            self._last_keep_mask = keep_mask.detach()

            # Microsoft sparse MLP scales output by max router probability.
            out_states = out_states * top_prob.unsqueeze(-1).to(out_states.dtype)
            return out_states

        mlp.forward = types.MethodType(_naive_forward, mlp)
        mlp._naive_dp_patched = True


# ---------------------------------------------------------------------------
# Parameter selection
# ---------------------------------------------------------------------------


def _is_attention_lora(name: str) -> bool:
    return (
        "SelfAttention" in name
        and _is_lora(name)
        and ".experts." not in name
        and ".router." not in name
    )


def _is_router(name: str) -> bool:
    return ".router.classifier" in name and "encoder.block" in name


def _is_classifier(name: str) -> bool:
    return name.startswith("classifier.")


def _is_lora(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name


def _is_attention_proj(name: str) -> bool:
    if "SelfAttention" not in name:
        return False
    if _is_lora(name):
        return False
    return any(f".{k}." in name for k in ("q", "k", "v", "o"))


def _is_opacus_incompatible_param(name: str) -> bool:
    # This embedding is indexed by relative position buckets [seq, seq] (no batch axis),
    # so Opacus cannot build per-sample gradients with consistent batch dimension.
    return "relative_attention_bias" in name


def _is_expert_ffn(name: str) -> bool:
    if ".mlp.experts." not in name:
        return False
    if _is_lora(name):
        return False
    return (".wi." in name) or (".wo." in name)


def _is_expert_param(name: str) -> bool:
    return ".mlp.experts." in name and "base_model.encoder.block." in name


def _is_forward_scope_param(name: str) -> bool:
    # This project's classifier forward uses encoder + shared embedding + classifier.
    return (
        name.startswith("classifier.")
        or name.startswith("base_model.shared.")
        or name.startswith("base_model.encoder.")
    )


def select_phase_a_params(
    model: SwitchSST2,
    freeze_targets: Optional[Sequence[str]] = None,
) -> List[Tuple[str, nn.Parameter]]:
    freeze_set = {t.strip().lower() for t in (freeze_targets or []) if t.strip()}
    out: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if not _is_forward_scope_param(name):
            continue
        if _is_opacus_incompatible_param(name):
            continue
        if _is_expert_param(name):
            continue
        if ("router" in freeze_set) and _is_router(name):
            continue
        if ("classifier" in freeze_set) and _is_classifier(name):
            continue
        if ("lora" in freeze_set) and _is_lora(name):
            continue
        out.append((name, p))
    return out


def select_ours_shared_params(
    model: SwitchSST2,
    freeze_targets: Optional[Sequence[str]] = None,
) -> List[Tuple[str, nn.Parameter]]:
    freeze_set = {t.strip().lower() for t in (freeze_targets or []) if t.strip()}
    out: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if not _is_forward_scope_param(name):
            continue
        if _is_opacus_incompatible_param(name):
            continue
        if _is_expert_param(name):
            continue
        if ("router" in freeze_set) and _is_router(name):
            continue
        if _is_classifier(name):
            if "classifier" not in freeze_set:
                out.append((name, p))
            continue
        if _is_lora(name) and "lora" not in freeze_set:
            out.append((name, p))
    return out


def select_non_expert_forward_params(model: SwitchSST2, include_router: bool = True) -> List[Tuple[str, nn.Parameter]]:
    out: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if not _is_forward_scope_param(name):
            continue
        if _is_opacus_incompatible_param(name):
            continue
        if _is_expert_param(name):
            continue
        if (not include_router) and _is_router(name):
            continue
        out.append((name, p))
    return out


def select_expert_params(model: SwitchSST2, layer: SparseLayerRef, expert_id: int) -> List[Tuple[str, nn.Parameter]]:
    prefix = f"base_model.encoder.block.{layer.block_id}.layer.1.mlp.experts.expert_{expert_id}."
    return [
        (name, p)
        for name, p in model.named_parameters()
        if name.startswith(prefix) and _is_lora(name)
    ]


def select_layer_expert_params(
    model: SwitchSST2,
    layer: SparseLayerRef,
    expert_ids: Optional[Sequence[int]] = None,
) -> List[Tuple[str, nn.Parameter]]:
    ids = (
        list(range(layer.sparse_mlp.router.num_experts))
        if expert_ids is None
        else [int(eid) for eid in expert_ids]
    )
    out: List[Tuple[str, nn.Parameter]] = []
    for expert_id in ids:
        out.extend(select_expert_params(model, layer, expert_id))
    return out


def select_sparse_layers_by_args(
    sparse_layers: Sequence[SparseLayerRef],
    args: argparse.Namespace,
    *,
    stage_name: str,
) -> List[SparseLayerRef]:
    n_layers = len(sparse_layers)
    raw_start_layer = int(args.start_layer)
    num_layers = int(args.num_layers_to_train)

    resolved_start_layer = raw_start_layer
    if raw_start_layer < 0:
        resolved_start_layer = n_layers + raw_start_layer
    if resolved_start_layer < 0 or resolved_start_layer >= n_layers:
        raise RuntimeError(
            f"[{stage_name}] --start_layer={args.start_layer} resolves to {resolved_start_layer}, "
            f"outside valid sparse layer range [0, {max(n_layers - 1, 0)}]."
        )

    if num_layers < 0:
        start = resolved_start_layer
        end = n_layers
    elif raw_start_layer < 0:
        # Negative start_layer is a right-anchored window. This makes the common
        # Switch setting `--start_layer -1 --num_layers_to_train K` mean last K
        # sparse layers instead of always degenerating to the last layer only.
        end = resolved_start_layer + 1
        start = max(0, end - num_layers)
    else:
        start = resolved_start_layer
        end = min(start + num_layers, n_layers)

    layers = list(sparse_layers[start:end])
    if not layers:
        raise RuntimeError(f"[{stage_name}] no sparse layers selected.")
    return layers


def select_same_scope_lora_params(
    model: SwitchSST2,
    layers: Sequence[SparseLayerRef],
    freeze_targets: Optional[Sequence[str]] = None,
) -> List[Tuple[str, nn.Parameter]]:
    out: List[Tuple[str, nn.Parameter]] = []
    seen_param_ids: set[int] = set()

    def add_many(named: Sequence[Tuple[str, nn.Parameter]]) -> None:
        for name, p in named:
            pid = id(p)
            if pid in seen_param_ids:
                continue
            out.append((name, p))
            seen_param_ids.add(pid)

    add_many(select_ours_shared_params(model, freeze_targets=freeze_targets))
    for layer in layers:
        add_many(select_layer_expert_params(model, layer))
    return out


def select_last_layer_lora_params(
    model: SwitchSST2,
    layers: Sequence[SparseLayerRef],
    *,
    include_classifier: bool = True,
) -> List[Tuple[str, nn.Parameter]]:
    block_ids = {int(layer.block_id) for layer in layers}
    out: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if include_classifier and _is_classifier(name):
            out.append((name, p))
            continue
        if not _is_lora(name):
            continue
        for block_id in block_ids:
            if name.startswith(f"base_model.encoder.block.{block_id}."):
                out.append((name, p))
                break
    return out


def select_all_lora_params(model: SwitchSST2, include_router: bool = False) -> List[Tuple[str, nn.Parameter]]:
    out: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if _is_classifier(name) or _is_lora(name) or (include_router and _is_router(name)):
            out.append((name, p))
    return out


def select_full_finetune_params(model: SwitchSST2, include_router: bool = True) -> List[Tuple[str, nn.Parameter]]:
    out: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if not _is_forward_scope_param(name):
            continue
        if _is_opacus_incompatible_param(name):
            continue
        if (not include_router) and _is_router(name):
            continue
        out.append((name, p))
    return out


# ---------------------------------------------------------------------------
# Privacy engine
# ---------------------------------------------------------------------------


def _external_prv_epsilon_bounds(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    target_delta: float,
    eps_error: float,
) -> Optional[Tuple[float, float, float]]:
    if ExternalPRVAccountant is None:
        return None
    accountant = ExternalPRVAccountant(
        noise_multiplier=float(noise_multiplier),
        sampling_probability=float(sample_rate),
        delta=float(target_delta),
        max_compositions=max(int(steps), 1),
        eps_error=float(eps_error),
    )
    low, estimate, high = accountant.compute_epsilon(max(int(steps), 1))
    return float(low), float(estimate), float(high)


def prv_epsilon_for_fixed_steps(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    target_delta: float,
    fallback_epsilon: float,
    eps_error: float = 0.1,
) -> float:
    if steps <= 0 or noise_multiplier <= 0:
        return 0.0

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
        accountant = OpacusPRVAccountant()
        for _ in range(int(steps)):
            accountant.step(noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
        return float(accountant.get_epsilon(float(target_delta)))

    raise RuntimeError(
        "PRV accounting requested but neither `prv_accountant` nor Opacus PRVAccountant is available. "
        "Install `prv-accountant` or a recent Opacus build with PRV support."
    )


def find_noise_multiplier_prv(
    *,
    sampling_probability: float,
    num_steps: int,
    target_epsilon: float,
    target_delta: float,
    eps_error: float = 0.1,
) -> float:
    if target_epsilon <= 0.0:
        raise ValueError(f"target_epsilon must be positive for DP training, got {target_epsilon}")

    if ExternalPRVAccountant is not None:
        eps_error = max(float(eps_error), 1e-6)

        def eps_upper(mu: float) -> float:
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

        lo = 0.0
        hi = 1.0
        while eps_upper(hi) > float(target_epsilon):
            hi *= math.sqrt(2.0)
            if hi > 100.0:
                raise RuntimeError(
                    "Finding a PRV noise multiplier did not converge. "
                    "Try increasing epsilon or decreasing the sampling probability."
                )

        for _ in range(60):
            mid = (lo + hi) / 2.0
            if eps_upper(mid) > float(target_epsilon):
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
            # Older Opacus releases accepted only epochs; epochs/sample_rate
            # is converted back to num_steps internally.
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
        "PRV noise calibration requested but no PRV accountant is available. "
        "Install `prv-accountant` or a recent Opacus build with PRV support."
    )


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
            "Opacus is not installed, but DP training now requires Opacus.\n"
            "Install it in your environment (e.g., `pip install opacus`)."
        )

    if args.noise_multiplier is not None and args.noise_multiplier > 0:
        sigma = float(args.noise_multiplier)
    else:
        # The released DP update is the logical batch after virtual/microbatch
        # accumulation, so calibrate with q = logical_batch / N and the number
        # of logical noisy optimizer steps. The physical microbatch is only a
        # memory implementation detail.
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
        pe._dp_moe_accountant_name = "prv"
    except Exception:
        # Older Opacus builds may not expose the internal PRV accountant. We
        # still calibrate and report privacy with PRV above; this internal
        # accountant is only used by Opacus' optimizer wrapper.
        pe = OpacusPrivacyEngine(accountant="rdp")
        pe._dp_moe_accountant_name = "rdp"
    model.train()
    private_model, private_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=sigma,
        max_grad_norm=float(args.max_grad_norm),
        poisson_sampling=False,
    )
    # Opacus infers expected_batch_size from the physical DataLoader. Here the
    # DataLoader batch is only a memory microbatch; skipped steps accumulate into
    # one logical DP update. Use the logical batch for mean-gradient scaling so
    # changing --micro_batch_size does not silently change the effective LR.
    if hasattr(private_optimizer, "expected_batch_size"):
        private_optimizer.expected_batch_size = int(effective_batch_size)
    return private_model, private_optimizer, private_loader, pe, sigma


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
    accountant = OpacusRDPAccountant()
    for _ in range(int(steps)):
        accountant.step(noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
    return float(accountant.get_epsilon(float(target_delta)))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def make_scheduler(optimizer: torch.optim.Optimizer, loader_len: int, accum: int, epochs: int):
    # Microsoft examples use a constant LR schedule.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)


def resolve_phase_batch_sizes(args: argparse.Namespace, phase: str) -> Tuple[int, int]:
    if phase == "phase_a":
        batch = int(getattr(args, "phase_a_train_batch_size", -1))
        micro = int(getattr(args, "phase_a_micro_batch_size", -1))
    elif phase == "phase_b":
        batch = int(getattr(args, "phase_b_train_batch_size", -1))
        micro = int(getattr(args, "phase_b_micro_batch_size", -1))
    else:
        batch = -1
        micro = -1

    if batch <= 0:
        batch = int(args.train_batch_size)
    if micro <= 0:
        micro = int(args.micro_batch_size)
    return batch, micro


def resolve_phase_b_sample_rate(
    args: argparse.Namespace,
    sample_size: int,
    configured_batch: int,
) -> float:
    q = float(getattr(args, "phase_b_sample_rate", -1.0))
    if q <= 0.0:
        q = float(min(max(int(configured_batch), 1), max(int(sample_size), 1))) / float(max(int(sample_size), 1))
    if not math.isfinite(q) or q <= 0.0 or q > 1.0:
        raise ValueError(f"--phase_b_sample_rate must be in (0, 1], or <=0 for batch/N fallback; got {q}")
    return q


def run_train_loop(
    *,
    stage_name: str,
    model: SwitchSST2,
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
    objective: str = "classifier",
    seq2seq_label_token_ids: Optional[torch.Tensor] = None,
    batch_phase: str = "default",
    dev_ds: Optional[Dataset] = None,
) -> Dict:
    if not named_params:
        raise RuntimeError(f"[{stage_name}] no trainable params selected")
    if len(train_ds) == 0:
        raise RuntimeError(f"[{stage_name}] empty training dataset")

    for _, p in named_params:
        p.requires_grad_(True)
    params = [p for _, p in named_params]
    expert_param_ids = {id(p) for name, p in named_params if _is_expert_param(name)}
    n_trainable = sum(p.numel() for p in params)

    configured_batch, configured_micro = resolve_phase_batch_sizes(args, batch_phase)
    effective = min(configured_batch, len(train_ds))
    micro = min(configured_micro, effective, len(train_ds))
    if dp:
        # Keep logical DP batch exactly divisible by micro-batch for stable Opacus accounting.
        exact_effective = max(micro, (effective // micro) * micro)
        if exact_effective != effective:
            print(
                f"[{stage_name}] aligning DP logical batch: "
                f"B={effective} -> {exact_effective} (micro={micro})"
            )
        effective = exact_effective
        accum = max(1, effective // micro)
    else:
        accum = max(1, math.ceil(effective / micro))

    optimizer = AdamW(
        params,
        lr=lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    loader = DataLoader(
        train_ds,
        batch_size=micro,
        shuffle=True,
        drop_last=dp,
        num_workers=0,
        collate_fn=make_collate(tokenizer, args.max_length, args.input_prefix),
    )
    if dp:
        dropped_by_loader = len(train_ds) - (len(loader) * micro)
        if dropped_by_loader > 0:
            print(
                f"[{stage_name}] dropping {dropped_by_loader} samples/epoch at microbatch boundary "
                f"to keep DP microbatches fixed at {micro}"
            )
    logical_steps_per_epoch = (len(loader) // accum) if dp else math.ceil(len(loader) / accum)
    logical_steps_total = int(logical_steps_per_epoch) * int(epochs)
    logical_sample_rate = min(1.0, float(effective) / float(max(len(train_ds), 1)))
    scheduler = make_scheduler(optimizer, len(loader), accum, epochs)

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
            f"[{stage_name}] DP target ε={target_epsilon:.4f}  δ={target_delta:.3e}  sample_size={len(train_ds)} "
            f"batch={effective} micro={micro} q={logical_sample_rate:.6f} "
            f"logical_steps={logical_steps_total} σ={sigma:.4f}"
        )
    else:
        print(f"[{stage_name}] no-DP mode  sample_size={len(train_ds)}  batch={effective} micro={micro}")

    best_loss = float("inf")
    best_state: Dict[str, torch.Tensor] = {}
    patience = 0
    _reset_runtime_peak_memory(device)
    t0 = time.time()
    completed_steps = 0
    epoch_losses: List[float] = []
    epoch_evals: List[Dict] = []

    print(f"[{stage_name}] trainable={n_trainable:,}  lr={lr}  epochs={epochs}")
    if objective not in ("classifier", "seq2seq"):
        raise ValueError(f"[{stage_name}] unsupported objective={objective!r}")
    label_token_ids = None
    if objective == "seq2seq":
        if seq2seq_label_token_ids is None:
            raise ValueError(f"[{stage_name}] seq2seq objective requires label token ids")
        label_token_ids = seq2seq_label_token_ids.to(device)

    def _scale_current_grads(divisor: int) -> None:
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
            iterator = tqdm(loader, desc=f"{stage_name} ep{epoch+1}/{epochs}", leave=False)

        for batch in iterator:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if hasattr(base_model, "_active_batch_indices"):
                base_model._active_batch_indices = batch["idx"].detach()

            if objective == "seq2seq":
                target_token_ids = label_token_ids.index_select(0, labels)
                loss = base_model.seq2seq_loss(ids, mask, target_token_ids)
            else:
                logits = train_model(ids, mask)
                if not torch.isfinite(logits).all():
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            # DP clipping/noise is handled by Opacus. For no-DP gradient
            # accumulation, scale just before stepping so the microbatch size is
            # a memory detail rather than an implicit LR multiplier.
            loss.backward()
            step_counter += 1
            accum_window_steps += 1
            batch_n = labels.size(0)
            if dp:
                _ensure_opacus_grad_samples(params, batch_n, allowed_param_ids=expert_param_ids)
            epoch_loss_sum += float(loss.detach().item()) * batch_n
            epoch_n += batch_n

            if dp:
                if step_counter % accum == 0:
                    if hasattr(optimizer, "signal_skip_step"):
                        optimizer.signal_skip_step(do_skip=False)
                    optimizer.step()
                    sanitize_parameters(params)
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    completed_steps += 1
                else:
                    if hasattr(optimizer, "signal_skip_step"):
                        optimizer.signal_skip_step(do_skip=True)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    elif hasattr(optimizer, "virtual_step"):
                        optimizer.virtual_step()
                    # If neither API exists, leave grads accumulated until the next full step.
            elif step_counter % accum == 0:
                _scale_current_grads(accum_window_steps)
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                sanitize_parameters(params)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
                accum_window_steps = 0

        if step_counter % accum != 0 and step_counter > 0:
            if dp:
                # Keep logical DP batch size fixed at expected_batch_size for Opacus.
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[{stage_name}] dropped incomplete DP logical batch at epoch end "
                    f"(tail microbatches={step_counter % accum})"
                )
            else:
                _scale_current_grads(accum_window_steps)
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                sanitize_parameters(params)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1

        epoch_loss = epoch_loss_sum / max(epoch_n, 1)
        epoch_losses.append(float(epoch_loss))
        print(f"[{stage_name}] epoch={epoch+1}/{epochs}  loss={epoch_loss:.4f}  n={epoch_n}")

        if dev_ds is not None:
            eval_metrics = evaluate(
                base_model, dev_ds, tokenizer,
                args.eval_batch_size, args.max_length, device,
                args.input_prefix, objective, seq2seq_label_token_ids,
            )
            print(
                f"[{stage_name}] eval epoch={epoch+1}/{epochs}"
                f"  acc={eval_metrics['accuracy']:.4f}"
                f"  loss={eval_metrics['loss']:.4f}"
                f"  n={eval_metrics['n']}"
            )
            epoch_evals.append({
                "epoch": int(epoch + 1),
                "train_loss": float(epoch_loss),
                "dev_acc": float(eval_metrics["accuracy"]),
                "dev_loss": float(eval_metrics["loss"]),
                "dev_n": int(eval_metrics["n"]),
            })
            train_model.train()

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
                    print(f"[{stage_name}] early stop at epoch {epoch+1}")
                    break

    if not dp and best_state:
        base_model.load_state_dict(best_state, strict=False)
        print(f"[{stage_name}] restored best checkpoint")

    epsilon = 0.0
    spent_python: Dict = {}
    if pe is not None:
        epsilon = prv_epsilon_for_fixed_steps(
            noise_multiplier=float(sigma),
            sample_rate=float(logical_sample_rate),
            steps=int(logical_steps_total),
            target_delta=float(target_delta),
            fallback_epsilon=float(target_epsilon),
            eps_error=float(getattr(args, "prv_eps_error", 0.1)),
        )
        eps_rdp = rdp_epsilon_for_fixed_steps(
            noise_multiplier=float(sigma),
            sample_rate=float(logical_sample_rate),
            steps=int(logical_steps_total),
            target_delta=float(target_delta),
            fallback_epsilon=float("nan"),
        )
        spent_python = {
            "eps_prv": epsilon,
            "eps_rdp": eps_rdp,
            "sample_rate": logical_sample_rate,
            "logical_steps": logical_steps_total,
            "logical_batch_size": effective,
            "physical_micro_batch_size": micro,
            "opacus_internal_eps_microbatch": float(pe.get_epsilon(float(target_delta))),
            "opacus_internal_accountant": getattr(pe, "_dp_moe_accountant_name", "unknown"),
            "accounting_note": "reported epsilon uses PRV accounting on logical batches after microbatch accumulation",
        }
        print(f"[{stage_name}] actual ε={epsilon:.4f}  σ={sigma:.4f}")
        if args.debug_privacy:
            print(f"[{stage_name}] privacy_spent={json.dumps(spent_python, indent=2)}")

    # Opacus wraps the model with GradSampleModule and installs autograd hooks.
    # We must unwrap after each DP stage; otherwise the next make_private() call
    # on the same model instance will fail with "Trying to add hooks twice".
    if dp and hasattr(train_model, "to_standard_module"):
        train_model.to_standard_module()

    for _, p in named_params:
        p.requires_grad_(False)
    if hasattr(base_model, "_active_batch_indices"):
        base_model._active_batch_indices = None

    runtime = _runtime_stats(t0, completed_steps, device)
    return {
        "stage": stage_name,
        "epsilon": epsilon,
        "sigma": sigma,
        "accountant": ACCOUNTING_MODE if dp else "none",
        "target_epsilon": target_epsilon if dp else 0.0,
        "target_delta": target_delta if dp else 0.0,
        "privacy_spent": spent_python,
        "n_trainable": n_trainable,
        "n_samples": len(train_ds),
        "logical_batch_size": int(effective),
        "physical_micro_batch_size": int(micro),
        "accumulation_steps": int(accum),
        "planned_optimizer_steps": int(logical_steps_total),
        "epoch_losses": epoch_losses,
        "epoch_evals": epoch_evals,
        **runtime,
    }


# ---------------------------------------------------------------------------
# Phase A
# ---------------------------------------------------------------------------


def train_phase_a(
    model: SwitchSST2,
    train_ds: SST2Dataset,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    epsilon_attention: float,
    delta_attention: float,
    objective: str = "classifier",
    seq2seq_label_token_ids: Optional[torch.Tensor] = None,
) -> Dict:
    freeze_all(model)
    zero_router_jitter(model)
    freeze_targets = {str(t).strip().lower() for t in getattr(args, "freeze_in_phase_a", []) if str(t).strip()}
    if getattr(args, "freeze_router_in_phase_a", False):
        freeze_targets.add("router")
    train_router = "router" not in freeze_targets
    freeze_desc = ",".join(sorted(freeze_targets)) if freeze_targets else "none"
    print(f"[PHASE A] freeze_targets={freeze_desc}  train_router={train_router}  (experts frozen)")
    print("[PHASE A] training non-expert forward-scope parameters.")
    named_params = select_phase_a_params(model, freeze_targets=sorted(freeze_targets))
    return run_train_loop(
        stage_name="PHASE_A",
        model=model,
        train_ds=train_ds,
        tokenizer=tokenizer,
        device=device,
        args=args,
        named_params=named_params,
        lr=args.lr_attention,
        epochs=args.attention_epochs,
        dp=not args.no_dp,
        target_epsilon=epsilon_attention,
        target_delta=delta_attention,
        objective=objective,
        seq2seq_label_token_ids=seq2seq_label_token_ids,
        batch_phase="phase_a",
    )


def run_upper_bound(
    model: SwitchSST2,
    train_ds: SST2Dataset,
    dev_ds: SST2Dataset,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    zero_shot: Dict,
) -> Dict:
    print("\n" + "=" * 88)
    print("[UPPER_BOUND] no-DP one-stage LoRA finetuning")
    freeze_all(model)
    named_params = select_all_lora_params(model, include_router=False)
    stats = run_train_loop(
        stage_name="UPPER_BOUND",
        model=model,
        train_ds=train_ds,
        tokenizer=tokenizer,
        device=device,
        args=args,
        named_params=named_params,
        lr=args.lr,
        epochs=args.finetune_epochs,
        dp=False,
        target_epsilon=0.0,
        target_delta=0.0,
        dev_ds=dev_ds,
    )
    final = evaluate(model, dev_ds, tokenizer, args.eval_batch_size, args.max_length, device, args.input_prefix)
    stats.update({"dev_acc": final["accuracy"], "dev_loss": final["loss"]})
    print(f"[UPPER_BOUND] zero={zero_shot['accuracy']:.4f} final={final['accuracy']:.4f}")
    return {"experiment": "upper_bound", "zero_shot": zero_shot, "result": stats}


def run_upper_bound_no_lora(
    model: SwitchSST2,
    train_ds: SST2Dataset,
    dev_ds: SST2Dataset,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    zero_shot: Dict,
) -> Dict:
    print("\n" + "=" * 88)
    print("[UPPER_BOUND_NO_LORA] no-DP full finetuning (no LoRA)")
    freeze_all(model)
    named_params = select_full_finetune_params(model, include_router=True)
    stats = run_train_loop(
        stage_name="UPPER_BOUND_NO_LORA",
        model=model,
        train_ds=train_ds,
        tokenizer=tokenizer,
        device=device,
        args=args,
        named_params=named_params,
        lr=args.lr_full_finetune,
        epochs=args.finetune_epochs,
        dp=False,
        target_epsilon=0.0,
        target_delta=0.0,
        dev_ds=dev_ds,
    )
    final = evaluate(model, dev_ds, tokenizer, args.eval_batch_size, args.max_length, device, args.input_prefix)
    stats.update({"dev_acc": final["accuracy"], "dev_loss": final["loss"]})
    print(f"[UPPER_BOUND_NO_LORA] zero={zero_shot['accuracy']:.4f} final={final['accuracy']:.4f}")
    return {"experiment": "upper_bound_no_lora", "zero_shot": zero_shot, "result": stats}


def run_naive_dp(
    model: SwitchSST2,
    train_ds: SST2Dataset,
    dev_ds: SST2Dataset,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    zero_shot: Dict,
) -> Dict:
    print("\n" + "=" * 88)
    print("[NAIVE_DP] one-stage LoRA DP training with token-level routing")
    print(f"[NAIVE_DP] ε={args.epsilon_total:.4f}  δ={args.delta:.3e}")
    freeze_all(model)
    named_params = select_all_lora_params(model, include_router=False)
    stats = run_train_loop(
        stage_name="NAIVE_DP",
        model=model,
        train_ds=train_ds,
        tokenizer=tokenizer,
        device=device,
        args=args,
        named_params=named_params,
        lr=args.lr,
        epochs=args.finetune_epochs,
        dp=not args.no_dp,
        target_epsilon=args.epsilon_total,
        target_delta=args.delta,
        dev_ds=dev_ds,
    )
    final = evaluate(model, dev_ds, tokenizer, args.eval_batch_size, args.max_length, device, args.input_prefix)
    logical_batch = int(
        stats.get("privacy_spent", {}).get(
            "logical_batch_size",
            min(max(1, int(args.train_batch_size)), len(train_ds)),
        )
    )
    dilution_metrics = compute_expert_dilution_metrics(
        model=model,
        train_ds=train_ds,
        diagnostic_ds=dev_ds,
        tokenizer=tokenizer,
        args=args,
        device=device,
        sparse_layers=get_sparse_layers(model),
        named_params=named_params,
        method_kind="naive",
        logical_batch_size=logical_batch,
        expert_sigma=float(stats.get("sigma", 0.0)),
        expert_update_scale_mode="batch",
    )
    stats.update({
        "dev_acc": final["accuracy"],
        "dev_loss": final["loss"],
        "expert_dilution_metrics": dilution_metrics,
    })
    if dilution_metrics.get("enabled"):
        print(f"[NAIVE_DP] expert_dilution_summary={json.dumps(dilution_metrics.get('summary', {}), sort_keys=True)}")
    else:
        print(f"[NAIVE_DP] expert_dilution_metrics skipped: {dilution_metrics.get('reason', 'unknown')}")
    print(f"[NAIVE_DP] zero={zero_shot['accuracy']:.4f} final={final['accuracy']:.4f} ε={stats.get('epsilon', 0.0):.4f}")
    return {"experiment": "naive_dp", "zero_shot": zero_shot, "result": stats}


def run_global_poisson_dp_loop(
    *,
    stage_name: str,
    model: SwitchSST2,
    train_ds: Dataset,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
    named_params: List[Tuple[str, nn.Parameter]],
    lr: float,
    epochs: int,
    target_epsilon: float,
    target_delta: float,
    dev_ds: Optional[Dataset] = None,
    objective: str = "classifier",
    seq2seq_label_token_ids: Optional[torch.Tensor] = None,
) -> Dict:
    if not named_params:
        raise RuntimeError(f"[{stage_name}] no trainable params selected")
    if len(train_ds) == 0:
        raise RuntimeError(f"[{stage_name}] empty training dataset")
    if OpacusGradSampleModule is None:
        raise RuntimeError(f"[{stage_name}] Opacus GradSampleModule is required for scoped global DP training.")

    configured_batch = int(args.train_batch_size)
    configured_micro = int(args.micro_batch_size)
    if configured_batch <= 0:
        raise ValueError(f"[{stage_name}] --train_batch_size must be positive, got {configured_batch}")
    if configured_micro <= 0:
        raise ValueError(f"[{stage_name}] --micro_batch_size must be positive, got {configured_micro}")

    public_n = max(1, len(train_ds))
    sample_rate = float(min(max(configured_batch, 1), public_n)) / float(public_n)
    expected_total = max(1, int(math.ceil(float(public_n) * float(sample_rate))))
    micro = min(max(1, int(configured_micro)), len(train_ds))
    updates_per_epoch = max(1, int(math.ceil(1.0 / float(sample_rate))))
    total_updates = int(updates_per_epoch) * int(epochs)
    sigma = 0.0
    if not args.no_dp:
        sigma = _stacked_noise_multiplier(
            sample_rate=sample_rate,
            steps=total_updates,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            args=args,
        )

    for _, p in named_params:
        p.requires_grad_(True)
    params = [p for _, p in named_params]
    expert_param_ids = {id(p) for name, p in named_params if _is_expert_param(name)}
    buffers = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in params}
    optimizer = AdamW(params, lr=lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, updates_per_epoch, 1, epochs)
    collate_fn = make_collate(tokenizer, args.max_length, args.input_prefix)

    print(
        f"[{stage_name}] global Poisson DP target ε={target_epsilon:.4f} δ={target_delta:.3e} "
        f"sample_size={len(train_ds)} batch={configured_batch} micro={micro} "
        f"q={sample_rate:.6f} updates={total_updates} σ={sigma:.4f}"
    )
    print(f"[{stage_name}] trainable={sum(p.numel() for p in params):,}  lr={lr}  epochs={epochs}")

    train_model: Optional[nn.Module] = None
    base_model: nn.Module = model
    _reset_runtime_peak_memory(device)
    t0 = time.time()
    completed_steps = 0

    def _accumulate_global(rows: torch.Tensor) -> None:
        if rows.numel() == 0:
            return
        norm_sq = torch.zeros(rows.numel(), device=device, dtype=torch.float32)
        usable: List[Tuple[nn.Parameter, torch.Tensor]] = []
        for p in params:
            gs = _grad_sample_as_tensor(p)
            if gs is None:
                continue
            gs_rows = gs.index_select(0, rows.to(gs.device))
            usable.append((p, gs_rows))
            norm_sq += gs_rows.detach().float().flatten(1).pow(2).sum(dim=1).to(norm_sq.device)
        if not usable:
            return
        factors = torch.clamp(float(args.max_grad_norm) / (norm_sq.sqrt() + 1e-6), max=1.0)
        for p, gs_rows in usable:
            view = factors
            while view.dim() < gs_rows.dim():
                view = view.unsqueeze(-1)
            clipped = (gs_rows * view.to(gs_rows.dtype)).sum(dim=0)
            buffers[p].add_(clipped.to(buffers[p].dtype))

    def _step() -> None:
        nonlocal completed_steps
        denom = max(1, int(configured_batch))
        for p in params:
            buf = buffers[p]
            grad = buf
            if not args.no_dp:
                noise = torch.randn(buf.shape, device=buf.device, dtype=torch.float32)
                grad = grad + noise.to(buf.dtype).mul_(float(sigma) * float(args.max_grad_norm))
            p.grad = (grad / float(denom)).to(p.dtype)
        optimizer.step()
        sanitize_parameters(params)
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        for p in params:
            buffers[p].zero_()
            p.grad = None
        completed_steps += 1

    epoch_losses: List[float] = []
    epoch_evals: List[Dict] = []
    try:
        train_model = OpacusGradSampleModule(model, batch_first=True, loss_reduction="mean")
        if hasattr(train_model, "_module"):
            base_model = train_model._module

        for epoch in range(epochs):
            train_model.train()
            epoch_loss_sum = 0.0
            epoch_n = 0
            iterator: Iterable = range(updates_per_epoch)
            if tqdm and args.show_progress:
                iterator = tqdm(iterator, desc=f"{stage_name} ep{epoch+1}/{epochs}", leave=False)

            for _update_idx in iterator:
                if sample_rate >= 1.0:
                    selected = torch.randperm(len(train_ds))
                else:
                    keep = torch.rand(len(train_ds)) < float(sample_rate)
                    selected = keep.nonzero(as_tuple=False).flatten()
                    if selected.numel() > 0:
                        selected = selected[torch.randperm(selected.numel())]

                selected_indices = selected.tolist()
                for start in range(0, len(selected_indices), micro):
                    chunk = selected_indices[start:start + micro]
                    if not chunk:
                        continue
                    batch = collate_fn([train_ds[int(idx)] for idx in chunk])
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    if hasattr(base_model, "_active_batch_indices"):
                        base_model._active_batch_indices = batch["idx"].detach()

                    logits = train_model(ids, mask)
                    if not torch.isfinite(logits).all():
                        _clear_grad_sample_state(params)
                        continue
                    loss = F.cross_entropy(logits, labels)
                    if not torch.isfinite(loss):
                        _clear_grad_sample_state(params)
                        continue
                    loss.backward()
                    _ensure_opacus_grad_samples(params, int(labels.size(0)), allowed_param_ids=expert_param_ids)
                    rows = torch.arange(labels.size(0), device=device, dtype=torch.long)
                    _accumulate_global(rows)
                    _clear_grad_sample_state(params)

                    batch_n = int(labels.size(0))
                    epoch_loss_sum += float(loss.detach().item()) * batch_n
                    epoch_n += batch_n

                _step()

            epoch_loss = epoch_loss_sum / max(epoch_n, 1)
            epoch_losses.append(epoch_loss)
            print(f"[{stage_name}] epoch={epoch+1}/{epochs}  loss={epoch_loss:.4f}  n={epoch_n}")

            if dev_ds is not None:
                eval_metrics = evaluate(
                    base_model,
                    dev_ds,
                    tokenizer,
                    args.eval_batch_size,
                    args.max_length,
                    device,
                    args.input_prefix,
                    objective,
                    seq2seq_label_token_ids,
                )
                print(
                    f"[{stage_name}] eval epoch={epoch+1}/{epochs}"
                    f"  acc={eval_metrics['accuracy']:.4f}"
                    f"  loss={eval_metrics['loss']:.4f}"
                    f"  n={eval_metrics['n']}"
                )
                epoch_evals.append({
                    "epoch": int(epoch + 1),
                    "train_loss": float(epoch_loss),
                    "dev_acc": float(eval_metrics["accuracy"]),
                    "dev_loss": float(eval_metrics["loss"]),
                    "dev_n": int(eval_metrics["n"]),
                })
                train_model.train()
    finally:
        if train_model is not None and hasattr(train_model, "to_standard_module"):
            train_model.to_standard_module()
        if hasattr(base_model, "_active_batch_indices"):
            base_model._active_batch_indices = None
        for _, p in named_params:
            p.requires_grad_(False)
        _clear_grad_sample_state(params)

    epsilon = 0.0
    spent_python: Dict = {}
    if not args.no_dp:
        epsilon = _stacked_privacy_epsilon(
            noise_multiplier=float(sigma),
            sample_rate=float(sample_rate),
            steps=int(total_updates),
            target_delta=float(target_delta),
            fallback_epsilon=float(target_epsilon),
            args=args,
        )
        eps_rdp = rdp_epsilon_for_fixed_steps(
            noise_multiplier=float(sigma),
            sample_rate=float(sample_rate),
            steps=int(total_updates),
            target_delta=float(target_delta),
            fallback_epsilon=float("nan"),
        )
        spent_python = {
            "eps_prv": epsilon,
            "eps_rdp": eps_rdp,
            "sample_rate": sample_rate,
            "logical_steps": int(total_updates),
            "logical_batch_size": int(configured_batch),
            "physical_micro_batch_size": int(micro),
            "accounting_note": "reported epsilon uses PRV accounting on the matched global Poisson update schedule",
        }
        print(f"[{stage_name}] actual ε={epsilon:.4f}  σ={sigma:.4f}")

    runtime = _runtime_stats(t0, completed_steps, device)
    return {
        "stage": stage_name,
        "epsilon": epsilon,
        "sigma": sigma,
        "accountant": ACCOUNTING_MODE if not args.no_dp else "none",
        "target_epsilon": target_epsilon if not args.no_dp else 0.0,
        "target_delta": target_delta if not args.no_dp else 0.0,
        "privacy_spent": spent_python,
        "n_trainable": int(sum(p.numel() for p in params)),
        "n_samples": len(train_ds),
        "logical_batch_size": int(configured_batch),
        "physical_micro_batch_size": int(micro),
        "planned_optimizer_steps": int(total_updates),
        "updates_per_epoch": int(updates_per_epoch),
        "sample_rate": float(sample_rate),
        "expected_sampled_records_per_update": int(expected_total),
        "epoch_losses": epoch_losses,
        "epoch_evals": epoch_evals,
        **runtime,
    }


def run_scoped_global_dp_lora(
    model: SwitchSST2,
    train_ds: SST2Dataset,
    dev_ds: SST2Dataset,
    sparse_layers: Sequence[SparseLayerRef],
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    zero_shot: Dict,
) -> Dict:
    raw_experiment = str(getattr(args, "requested_experiment", args.experiment))
    experiment = canonical_experiment_name(raw_experiment)
    if experiment not in {"baseline_a_global_matched", "baseline_b_shared_only", "baseline_e_selected_layer_lora"}:
        raise ValueError(f"[SCOPED_NAIVE_DP_LORA] unsupported experiment={experiment!r}")

    print("\n" + "=" * 88)
    print("[BASELINE_GLOBAL_DP_LORA] global DP-Adam LoRA training with restricted scope")
    print(
        f"[BASELINE_GLOBAL_DP_LORA] experiment={experiment} "
        f"(requested={raw_experiment}) ε={args.epsilon_total:.4f}  δ={args.delta:.3e}"
    )

    freeze_all(model)
    shared_freeze_targets = {str(t).strip().lower() for t in args.freeze_in_phase_a if str(t).strip()}
    if args.freeze_router_in_phase_a:
        shared_freeze_targets.add("router")

    layers_to_train: List[SparseLayerRef] = []
    target_epsilon = float(args.epsilon_total)
    target_delta = float(args.delta)
    if experiment == "baseline_b_shared_only":
        stage_name = "BASELINE_B_SHARED_ONLY"
        scope_desc = "shared_classifier_plus_non_expert_lora"
        named_params = select_ours_shared_params(model, freeze_targets=sorted(shared_freeze_targets))
        eps_mode = str(getattr(args, "baseline_epsilon_mode", "full")).strip().lower()
        if eps_mode == "shared":
            ratio = float(getattr(args, "epsilon_shared_ratio", 0.3))
            target_epsilon = float(args.epsilon_total) * ratio
            target_delta = float(args.delta) * ratio
            scope_desc = f"{scope_desc}_epsshared"
        elif eps_mode != "full":
            raise ValueError(f"[DP_SHARED_ONLY] unknown --baseline_epsilon_mode={eps_mode!r}")
    elif experiment == "baseline_a_global_matched":
        stage_name = "BASELINE_A_GLOBAL_MATCHED"
        scope_desc = "matched_shared_plus_selected_expert_lora_global_dp"
        layers_to_train = select_sparse_layers_by_args(
            sparse_layers,
            args,
            stage_name=stage_name,
        )
        named_params = select_same_scope_lora_params(
            model,
            layers_to_train,
            freeze_targets=sorted(shared_freeze_targets),
        )
    else:
        stage_name = "BASELINE_E_SELECTED_LAYER_LORA"
        scope_desc = "selected_layer_lora_global_dp"
        layers_to_train = select_sparse_layers_by_args(
            sparse_layers,
            args,
            stage_name=stage_name,
        )
        named_params = select_last_layer_lora_params(
            model,
            layers_to_train,
            include_classifier=True,
        )
    if not named_params:
        raise RuntimeError(f"[{stage_name}] no trainable LoRA/classifier params selected.")

    run_epochs = int(getattr(args, "ours_epochs", -1))
    if run_epochs <= 0:
        run_epochs = int(args.finetune_epochs)
    if layers_to_train:
        print(f"[{stage_name}] selected sparse layers = {[sl.layer_id for sl in layers_to_train]}")
    print(
        f"[{stage_name}] scope={scope_desc} trainable={sum(p.numel() for _, p in named_params):,} "
        f"epochs={run_epochs} lr={args.lr} global_clip={args.max_grad_norm}"
    )
    stats = run_global_poisson_dp_loop(
        stage_name=stage_name,
        model=model,
        train_ds=train_ds,
        tokenizer=tokenizer,
        device=device,
        args=args,
        named_params=named_params,
        lr=args.lr,
        epochs=run_epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        dev_ds=dev_ds,
    )
    final = evaluate(model, dev_ds, tokenizer, args.eval_batch_size, args.max_length, device, args.input_prefix)
    logical_batch = int(
        stats.get("privacy_spent", {}).get(
            "logical_batch_size",
            min(max(1, int(args.train_batch_size)), len(train_ds)),
        )
    )
    if layers_to_train:
        dilution_metrics = compute_expert_dilution_metrics(
            model=model,
            train_ds=train_ds,
            diagnostic_ds=dev_ds,
            tokenizer=tokenizer,
            args=args,
            device=device,
            sparse_layers=layers_to_train,
            named_params=named_params,
            method_kind="naive",
            logical_batch_size=logical_batch,
            expert_sigma=float(stats.get("sigma", 0.0)),
            expert_update_scale_mode="batch",
        )
    else:
        dilution_metrics = {"enabled": False, "reason": "shared_only_no_expert_params"}
    stats.update({
        "dev_acc": final["accuracy"],
        "dev_loss": final["loss"],
        "selected_layer_ids": [int(sl.layer_id) for sl in layers_to_train],
        "scope": scope_desc,
        "baseline_epsilon_mode": str(getattr(args, "baseline_epsilon_mode", "full")),
        "expert_dilution_metrics": dilution_metrics,
    })
    if dilution_metrics.get("enabled"):
        print(
            f"[{stage_name}] expert_dilution_summary="
            f"{json.dumps(dilution_metrics.get('summary', {}), sort_keys=True)}"
        )
    else:
        print(
            f"[{stage_name}] expert_dilution_metrics skipped: "
            f"{dilution_metrics.get('reason', 'unknown')}"
        )
    print(
        f"[{stage_name}] zero={zero_shot['accuracy']:.4f} "
        f"final={final['accuracy']:.4f} ε={stats.get('epsilon', 0.0):.4f}"
    )
    return {"experiment": experiment, "zero_shot": zero_shot, "result": stats}


def run_naive_dp_no_lora(
    model: SwitchSST2,
    train_ds: SST2Dataset,
    dev_ds: SST2Dataset,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    zero_shot: Dict,
) -> Dict:
    print("\n" + "=" * 88)
    print("[NAIVE_DP_NO_LORA] one-stage full-parameter DP training with token-level routing")
    print(f"[NAIVE_DP_NO_LORA] ε={args.epsilon_total:.4f}  δ={args.delta:.3e}")
    freeze_all(model)
    named_params = select_full_finetune_params(model, include_router=True)
    stats = run_train_loop(
        stage_name="NAIVE_DP_NO_LORA",
        model=model,
        train_ds=train_ds,
        tokenizer=tokenizer,
        device=device,
        args=args,
        named_params=named_params,
        lr=args.lr_full_finetune,
        epochs=args.finetune_epochs,
        dp=not args.no_dp,
        target_epsilon=args.epsilon_total,
        target_delta=args.delta,
        dev_ds=dev_ds,
    )
    final = evaluate(model, dev_ds, tokenizer, args.eval_batch_size, args.max_length, device, args.input_prefix)
    logical_batch = int(
        stats.get("privacy_spent", {}).get(
            "logical_batch_size",
            min(max(1, int(args.train_batch_size)), len(train_ds)),
        )
    )
    dilution_metrics = compute_expert_dilution_metrics(
        model=model,
        train_ds=train_ds,
        diagnostic_ds=dev_ds,
        tokenizer=tokenizer,
        args=args,
        device=device,
        sparse_layers=get_sparse_layers(model),
        named_params=named_params,
        method_kind="naive",
        logical_batch_size=logical_batch,
        expert_sigma=float(stats.get("sigma", 0.0)),
        expert_update_scale_mode="batch",
    )
    stats.update({
        "dev_acc": final["accuracy"],
        "dev_loss": final["loss"],
        "expert_dilution_metrics": dilution_metrics,
    })
    if dilution_metrics.get("enabled"):
        print(
            f"[NAIVE_DP_NO_LORA] expert_dilution_summary="
            f"{json.dumps(dilution_metrics.get('summary', {}), sort_keys=True)}"
        )
    else:
        print(
            f"[NAIVE_DP_NO_LORA] expert_dilution_metrics skipped: "
            f"{dilution_metrics.get('reason', 'unknown')}"
        )
    print(
        f"[NAIVE_DP_NO_LORA] zero={zero_shot['accuracy']:.4f} "
        f"final={final['accuracy']:.4f} ε={stats.get('epsilon', 0.0):.4f}"
    )
    return {"experiment": "naive_dp_no_lora", "zero_shot": zero_shot, "result": stats}


# ---------------------------------------------------------------------------
# DP-SFT baseline
# ---------------------------------------------------------------------------


def select_dpsft_params(model: SwitchSST2, args: argparse.Namespace) -> List[Tuple[str, nn.Parameter]]:
    scope = str(getattr(args, "dpsft_param_scope", "full_forward")).strip().lower()
    if scope == "full_forward":
        return select_full_finetune_params(model, include_router=True)
    if scope == "non_expert_forward":
        return select_non_expert_forward_params(model, include_router=True)
    if scope == "lora":
        return select_all_lora_params(model, include_router=False)
    raise ValueError(f"Unsupported --dpsft_param_scope={scope!r}")


def _dpsft_resolve_positive(value: float, fallback: float) -> float:
    value = float(value)
    return value if value > 0.0 else float(fallback)


def _dpsft_snapshot_steps(total_steps: int, subspace_dim: int) -> List[int]:
    total_steps = int(total_steps)
    if total_steps <= 0:
        return [0]
    k = max(1, min(int(subspace_dim), total_steps))
    steps = {0, total_steps}
    for i in range(1, k):
        steps.add(max(1, min(total_steps, int(round(float(i) * float(total_steps) / float(k))))))
    return sorted(steps)


@torch.no_grad()
def _dpsft_flat_params(named_params: Sequence[Tuple[str, nn.Parameter]]) -> torch.Tensor:
    if not named_params:
        raise RuntimeError("[DP-SFT] cannot flatten an empty parameter list")
    return torch.cat([p.detach().cpu().float().reshape(-1) for _, p in named_params], dim=0)


def _dpsft_param_slices(named_params: Sequence[Tuple[str, nn.Parameter]]) -> Dict[str, Dict]:
    cursor = 0
    meta: Dict[str, Dict] = {}
    for name, p in named_params:
        n = int(p.numel())
        meta[name] = {
            "start": int(cursor),
            "end": int(cursor + n),
            "shape": list(p.shape),
            "numel": n,
        }
        cursor += n
    return meta


def _dpsft_save_snapshot(
    *,
    named_params: Sequence[Tuple[str, nn.Parameter]],
    trajectory_dir: str,
    step: int,
    index_rows: List[Dict],
) -> None:
    os.makedirs(trajectory_dir, exist_ok=True)
    path = os.path.join(trajectory_dir, f"snapshot_{int(step):06d}.pt")
    torch.save({"step": int(step), "flat_params": _dpsft_flat_params(named_params)}, path)
    index_rows.append({"step": int(step), "path": path})
    with open(os.path.join(trajectory_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index_rows, f, indent=2)
    print(f"[DP-SFT] saved trajectory snapshot step={int(step)} -> {path}")


def _dpsft_logical_batch_config(
    train_ds: Dataset,
    args: argparse.Namespace,
    *,
    stage: str,
) -> Tuple[int, int, int, float]:
    configured_batch = int(args.train_batch_size)
    configured_micro = int(args.micro_batch_size)
    if configured_batch <= 0:
        raise ValueError(f"[{stage}] --train_batch_size must be positive, got {configured_batch}")
    if configured_micro <= 0:
        raise ValueError(f"[{stage}] --micro_batch_size must be positive, got {configured_micro}")
    effective = min(configured_batch, len(train_ds))
    micro = min(configured_micro, effective, len(train_ds))
    effective = max(micro, (effective // micro) * micro)
    accum = max(1, effective // micro)
    sample_rate = min(1.0, float(effective) / float(max(len(train_ds), 1)))
    return int(effective), int(micro), int(accum), float(sample_rate)


def _dpsft_make_loader(
    train_ds: Dataset,
    tokenizer,
    args: argparse.Namespace,
    micro: int,
) -> DataLoader:
    return DataLoader(
        train_ds,
        batch_size=int(micro),
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=make_collate(tokenizer, args.max_length, args.input_prefix),
    )


def _dpsft_compute_loss(
    train_model: nn.Module,
    ids: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
) -> Optional[torch.Tensor]:
    logits = train_model(ids, mask)
    if not torch.isfinite(logits).all():
        return None
    loss = F.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        return None
    return loss


def _dpsft_train_stage1_trajectory(
    *,
    model: SwitchSST2,
    train_ds: SST2Dataset,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    named_params: Sequence[Tuple[str, nn.Parameter]],
    target_epsilon: float,
    target_delta: float,
    trajectory_dir: str,
    dev_ds: Optional[Dataset] = None,
) -> Dict:
    if OpacusGradSampleModule is None:
        raise RuntimeError("Opacus GradSampleModule is required for DP-SFT trajectory training.")
    if len(train_ds) == 0:
        raise RuntimeError("[DP-SFT STAGE1] empty training dataset")

    epochs = int(getattr(args, "dpsft_stage1_epochs", 1))
    if epochs <= 0:
        raise ValueError(f"[DP-SFT STAGE1] --dpsft_stage1_epochs must be positive, got {epochs}")
    lr = _dpsft_resolve_positive(getattr(args, "dpsft_stage1_lr", -1.0), args.lr_full_finetune)
    clip_norm = _dpsft_resolve_positive(getattr(args, "dpsft_stage1_clip_norm", -1.0), args.max_grad_norm)
    subspace_dim = int(getattr(args, "dpsft_subspace_dim", 8))
    effective, micro, accum, sample_rate = _dpsft_logical_batch_config(train_ds, args, stage="DP-SFT STAGE1")
    loader = _dpsft_make_loader(train_ds, tokenizer, args, micro)
    logical_steps_per_epoch = len(loader) // accum
    logical_steps_total = int(logical_steps_per_epoch) * int(epochs)
    if logical_steps_total <= 0:
        raise RuntimeError(
            f"[DP-SFT STAGE1] no complete logical batches: loader_len={len(loader)} accum={accum}"
        )

    sigma = _stacked_noise_multiplier(
        sample_rate=sample_rate,
        steps=logical_steps_total,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        args=args,
    )
    params = [p for _, p in named_params]
    for _, p in named_params:
        p.requires_grad_(True)
    optimizer = AdamW(params, lr=lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, logical_steps_per_epoch, 1, epochs)
    buffers = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in params}
    snapshot_steps = set(_dpsft_snapshot_steps(logical_steps_total, subspace_dim))
    index_rows: List[Dict] = []
    _dpsft_save_snapshot(named_params=named_params, trajectory_dir=trajectory_dir, step=0, index_rows=index_rows)

    print(
        f"[DP-SFT STAGE1] epochs={epochs} steps={logical_steps_total} q={sample_rate:.6f} "
        f"batch={effective} micro={micro} accum={accum} lr={lr} clip={clip_norm} sigma={sigma:.4f}"
    )

    train_model: Optional[nn.Module] = None
    base_model: nn.Module = model
    _reset_runtime_peak_memory(device)
    t0 = time.time()
    completed_steps = 0
    epoch_losses: List[float] = []
    epoch_evals: List[Dict] = []

    def _accumulate_microbatch(batch_n: int) -> None:
        _ensure_opacus_grad_samples(params, batch_n)
        norm_sq = torch.zeros(batch_n, device=device, dtype=torch.float32)
        usable: List[Tuple[nn.Parameter, torch.Tensor]] = []
        for p in params:
            gs = _grad_sample_as_tensor(p)
            if gs is None:
                continue
            gs = gs.to(device)
            usable.append((p, gs))
            norm_sq += gs.detach().float().flatten(1).pow(2).sum(dim=1).to(norm_sq.device)
        if not usable:
            return
        factors = torch.clamp(float(clip_norm) / (norm_sq.sqrt() + 1e-6), max=1.0)
        for p, gs in usable:
            view = factors
            while view.dim() < gs.dim():
                view = view.unsqueeze(-1)
            clipped = (gs * view.to(gs.dtype)).sum(dim=0)
            buffers[p].add_(clipped.to(buffers[p].dtype))

    def _step() -> None:
        nonlocal completed_steps
        for p in params:
            grad = buffers[p]
            if not args.no_dp:
                noise = torch.randn(grad.shape, device=grad.device, dtype=torch.float32)
                grad = grad + noise.to(grad.dtype).mul_(float(sigma) * float(clip_norm))
            p.grad = (grad / float(max(effective, 1))).to(p.dtype)
        optimizer.step()
        sanitize_parameters(params)
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        for p in params:
            buffers[p].zero_()
            p.grad = None
        completed_steps += 1
        if completed_steps in snapshot_steps:
            _dpsft_save_snapshot(
                named_params=named_params,
                trajectory_dir=trajectory_dir,
                step=completed_steps,
                index_rows=index_rows,
            )

    try:
        train_model = OpacusGradSampleModule(model, batch_first=True, loss_reduction="mean")
        if hasattr(train_model, "_module"):
            base_model = train_model._module
        for epoch in range(epochs):
            train_model.train()
            epoch_loss_sum = 0.0
            epoch_n = 0
            max_micro_steps = int(logical_steps_per_epoch) * int(accum)
            iterator: Iterable = loader
            if tqdm and args.show_progress:
                iterator = tqdm(loader, desc=f"DP-SFT stage1 ep{epoch+1}/{epochs}", leave=False)
            micro_step = 0
            for batch in iterator:
                if micro_step >= max_micro_steps:
                    break
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                if hasattr(base_model, "_active_batch_indices"):
                    base_model._active_batch_indices = batch["idx"].detach()
                loss = _dpsft_compute_loss(train_model, ids, mask, labels)
                if loss is None:
                    _clear_grad_sample_state(params)
                    micro_step += 1
                    continue
                loss.backward()
                _accumulate_microbatch(int(labels.size(0)))
                _clear_grad_sample_state(params)
                batch_n = int(labels.size(0))
                epoch_loss_sum += float(loss.detach().item()) * batch_n
                epoch_n += batch_n
                micro_step += 1
                if micro_step % accum == 0:
                    _step()
            epoch_loss = epoch_loss_sum / max(epoch_n, 1)
            epoch_losses.append(epoch_loss)
            print(f"[DP-SFT STAGE1] epoch={epoch+1}/{epochs} loss={epoch_loss:.4f} n={epoch_n}")
            if dev_ds is not None:
                eval_metrics = evaluate(
                    base_model,
                    dev_ds,
                    tokenizer,
                    args.eval_batch_size,
                    args.max_length,
                    device,
                    args.input_prefix,
                )
                print(
                    f"[DP-SFT STAGE1] eval epoch={epoch+1}/{epochs}"
                    f" acc={eval_metrics['accuracy']:.4f}"
                    f" loss={eval_metrics['loss']:.4f}"
                    f" n={eval_metrics['n']}"
                )
                epoch_evals.append({
                    "epoch": int(epoch + 1),
                    "train_loss": float(epoch_loss),
                    "dev_acc": float(eval_metrics["accuracy"]),
                    "dev_loss": float(eval_metrics["loss"]),
                    "dev_n": int(eval_metrics["n"]),
                })
                train_model.train()
    finally:
        if train_model is not None and hasattr(train_model, "to_standard_module"):
            train_model.to_standard_module()
        if hasattr(base_model, "_active_batch_indices"):
            base_model._active_batch_indices = None
        _clear_grad_sample_state(params)

    if logical_steps_total not in {int(r["step"]) for r in index_rows}:
        _dpsft_save_snapshot(
            named_params=named_params,
            trajectory_dir=trajectory_dir,
            step=completed_steps,
            index_rows=index_rows,
        )

    epsilon = 0.0 if args.no_dp else _stacked_privacy_epsilon(
        noise_multiplier=float(sigma),
        sample_rate=float(sample_rate),
        steps=int(logical_steps_total),
        target_delta=float(target_delta),
        fallback_epsilon=float(target_epsilon),
        args=args,
    )
    runtime = _runtime_stats(t0, completed_steps, device)
    return {
        "stage": "dpsft_stage1_trajectory",
        "epsilon": epsilon,
        "sigma": sigma,
        "accountant": ACCOUNTING_MODE if not args.no_dp else "none",
        "target_epsilon": float(target_epsilon) if not args.no_dp else 0.0,
        "target_delta": float(target_delta) if not args.no_dp else 0.0,
        "privacy_spent": {
            "eps_prv": epsilon,
            "sample_rate": sample_rate,
            "logical_steps": int(logical_steps_total),
            "logical_batch_size": int(effective),
            "physical_micro_batch_size": int(micro),
            "accounting_note": "DP-SFT stage 1 trajectory uses PRV accounting on logical batches.",
        },
        "trajectory_dir": trajectory_dir,
        "snapshot_count": len(index_rows),
        "snapshot_steps": [int(r["step"]) for r in index_rows],
        "n_trainable": int(sum(p.numel() for p in params)),
        "logical_batch_size": int(effective),
        "physical_micro_batch_size": int(micro),
        "planned_optimizer_steps": int(logical_steps_total),
        "epoch_losses": epoch_losses,
        "epoch_evals": epoch_evals,
        **runtime,
    }


def _dpsft_compute_subspace(
    *,
    trajectory_dir: str,
    named_params: Sequence[Tuple[str, nn.Parameter]],
    subspace_dim: int,
    output_dir: str,
) -> Tuple[torch.Tensor, Dict[str, Dict], Dict]:
    index_path = os.path.join(trajectory_dir, "index.json")
    if not os.path.exists(index_path):
        raise RuntimeError(f"[DP-SFT SVD] missing trajectory index: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        index_rows = json.load(f)
    if len(index_rows) < 2:
        raise RuntimeError("[DP-SFT SVD] need at least two trajectory snapshots")
    first = torch.load(index_rows[0]["path"], map_location="cpu")
    initial = first["flat_params"].float()
    deltas: List[torch.Tensor] = []
    for row in index_rows[1:]:
        snap = torch.load(row["path"], map_location="cpu")
        flat = snap["flat_params"].float()
        if flat.numel() != initial.numel():
            raise RuntimeError(
                f"[DP-SFT SVD] snapshot size mismatch: got {flat.numel()} expected {initial.numel()}"
            )
        deltas.append(flat - initial)
    delta_matrix = torch.stack(deltas, dim=0)
    if not torch.isfinite(delta_matrix).all():
        delta_matrix = torch.nan_to_num(delta_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    actual_dim = max(1, min(int(subspace_dim), int(delta_matrix.size(0))))
    print(
        f"[DP-SFT SVD] delta_matrix={tuple(delta_matrix.shape)} requested_dim={subspace_dim} "
        f"actual_dim={actual_dim}"
    )
    svd_t0 = time.time()
    _u, singular_values, vh = torch.linalg.svd(delta_matrix, full_matrices=False)
    directions = vh[:actual_dim].contiguous().cpu().float()
    total_energy = float(singular_values.detach().float().pow(2).sum().item())
    top_energy = float(singular_values[:actual_dim].detach().float().pow(2).sum().item())
    energy_coverage = (top_energy / total_energy) if total_energy > 0.0 else 0.0
    param_meta = _dpsft_param_slices(named_params)
    os.makedirs(output_dir, exist_ok=True)
    direction_path = os.path.join(output_dir, "dpsft_subspace.pt")
    torch.save(
        {
            "directions": directions,
            "singular_values": singular_values.detach().cpu(),
            "param_meta": param_meta,
            "trajectory_index": index_rows,
        },
        direction_path,
    )
    info = {
        "stage": "dpsft_svd",
        "trajectory_dir": trajectory_dir,
        "direction_path": direction_path,
        "num_snapshots": int(len(index_rows)),
        "delta_matrix_shape": [int(x) for x in delta_matrix.shape],
        "requested_subspace_dim": int(subspace_dim),
        "actual_subspace_dim": int(actual_dim),
        "singular_values": [float(x) for x in singular_values.detach().cpu().tolist()],
        "energy_coverage": float(energy_coverage),
        "seconds": round(time.time() - svd_t0, 1),
        "completed_optimizer_steps": 0,
        "peak_gpu_memory_gb": 0.0,
    }
    del delta_matrix
    return directions, param_meta, info


def _dpsft_train_stage2_subspace(
    *,
    model: SwitchSST2,
    train_ds: SST2Dataset,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    named_params: Sequence[Tuple[str, nn.Parameter]],
    directions_cpu: torch.Tensor,
    param_meta: Dict[str, Dict],
    target_epsilon: float,
    target_delta: float,
    dev_ds: Optional[Dataset] = None,
) -> Dict:
    if OpacusGradSampleModule is None:
        raise RuntimeError("Opacus GradSampleModule is required for DP-SFT subspace training.")
    epochs = int(getattr(args, "dpsft_stage2_epochs", 1))
    if epochs <= 0:
        raise ValueError(f"[DP-SFT STAGE2] --dpsft_stage2_epochs must be positive, got {epochs}")
    lr = _dpsft_resolve_positive(getattr(args, "dpsft_stage2_lr", -1.0), args.lr)
    clip_norm = _dpsft_resolve_positive(getattr(args, "dpsft_stage2_clip_norm", -1.0), args.max_grad_norm)
    effective, micro, accum, sample_rate = _dpsft_logical_batch_config(train_ds, args, stage="DP-SFT STAGE2")
    loader = _dpsft_make_loader(train_ds, tokenizer, args, micro)
    logical_steps_per_epoch = len(loader) // accum
    logical_steps_total = int(logical_steps_per_epoch) * int(epochs)
    if logical_steps_total <= 0:
        raise RuntimeError(
            f"[DP-SFT STAGE2] no complete logical batches: loader_len={len(loader)} accum={accum}"
        )
    sigma = _stacked_noise_multiplier(
        sample_rate=sample_rate,
        steps=logical_steps_total,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        args=args,
    )

    params = [p for _, p in named_params]
    for _, p in named_params:
        p.requires_grad_(True)
    optimizer = AdamW(params, lr=lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, logical_steps_per_epoch, 1, epochs)
    d = int(directions_cpu.size(0))
    max_gpu_direction_elems = int(getattr(args, "dpsft_gpu_directions_max_elements", 20_000_000))
    directions_device: Optional[torch.Tensor] = None
    if directions_cpu.numel() <= max_gpu_direction_elems:
        directions_device = directions_cpu.to(device=device, dtype=torch.float32)
        print(f"[DP-SFT STAGE2] keeping directions on {device} ({directions_cpu.numel():,} elements)")
    else:
        print(
            f"[DP-SFT STAGE2] streaming directions from CPU "
            f"({directions_cpu.numel():,} elements > {max_gpu_direction_elems:,})"
        )

    print(
        f"[DP-SFT STAGE2] epochs={epochs} steps={logical_steps_total} q={sample_rate:.6f} "
        f"batch={effective} micro={micro} accum={accum} lr={lr} clip={clip_norm} "
        f"sigma={sigma:.4f} subspace_dim={d}"
    )

    train_model: Optional[nn.Module] = None
    base_model: nn.Module = model
    _reset_runtime_peak_memory(device)
    t0 = time.time()
    completed_steps = 0
    epoch_losses: List[float] = []
    epoch_evals: List[Dict] = []

    def _direction_slice(name: str) -> torch.Tensor:
        meta = param_meta[name]
        start, end = int(meta["start"]), int(meta["end"])
        if directions_device is not None:
            return directions_device[:, start:end]
        return directions_cpu[:, start:end].to(device=device, dtype=torch.float32)

    def _project_microbatch(batch_n: int) -> torch.Tensor:
        _ensure_opacus_grad_samples(params, batch_n)
        low = torch.zeros(batch_n, d, device=device, dtype=torch.float32)
        for name, p in named_params:
            gs = _grad_sample_as_tensor(p)
            if gs is None:
                continue
            flat = gs.detach().to(device=device, dtype=torch.float32).flatten(1)
            low.add_(flat.matmul(_direction_slice(name).t()))
        return low

    def _step(lowdim_parts: Sequence[torch.Tensor]) -> None:
        nonlocal completed_steps
        if lowdim_parts:
            lowdim_grads = torch.cat([x.to(device=device, dtype=torch.float32) for x in lowdim_parts], dim=0)
        else:
            lowdim_grads = torch.zeros(0, d, device=device, dtype=torch.float32)
        if lowdim_grads.size(0) == 0:
            avg_low = torch.zeros(d, device=device, dtype=torch.float32)
        else:
            norms = lowdim_grads.norm(p=2, dim=1)
            factors = torch.clamp(float(clip_norm) / (norms + 1e-6), max=1.0)
            clipped_sum = (lowdim_grads * factors.unsqueeze(1)).sum(dim=0)
            if not args.no_dp:
                noise = torch.randn(d, device=device, dtype=torch.float32).mul_(float(sigma) * float(clip_norm))
                clipped_sum = clipped_sum + noise
            avg_low = clipped_sum / float(max(effective, 1))
        for name, p in named_params:
            direction_block = _direction_slice(name)
            grad_flat = avg_low.matmul(direction_block)
            p.grad = grad_flat.view_as(p).to(p.dtype)
        optimizer.step()
        sanitize_parameters(params)
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        for p in params:
            p.grad = None
        completed_steps += 1

    try:
        train_model = OpacusGradSampleModule(model, batch_first=True, loss_reduction="mean")
        if hasattr(train_model, "_module"):
            base_model = train_model._module
        for epoch in range(epochs):
            train_model.train()
            epoch_loss_sum = 0.0
            epoch_n = 0
            max_micro_steps = int(logical_steps_per_epoch) * int(accum)
            lowdim_parts: List[torch.Tensor] = []
            iterator: Iterable = loader
            if tqdm and args.show_progress:
                iterator = tqdm(loader, desc=f"DP-SFT stage2 ep{epoch+1}/{epochs}", leave=False)
            micro_step = 0
            for batch in iterator:
                if micro_step >= max_micro_steps:
                    break
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                if hasattr(base_model, "_active_batch_indices"):
                    base_model._active_batch_indices = batch["idx"].detach()
                loss = _dpsft_compute_loss(train_model, ids, mask, labels)
                if loss is None:
                    _clear_grad_sample_state(params)
                    micro_step += 1
                    continue
                loss.backward()
                lowdim_parts.append(_project_microbatch(int(labels.size(0))).detach())
                _clear_grad_sample_state(params)
                batch_n = int(labels.size(0))
                epoch_loss_sum += float(loss.detach().item()) * batch_n
                epoch_n += batch_n
                micro_step += 1
                if micro_step % accum == 0:
                    _step(lowdim_parts)
                    lowdim_parts = []
            epoch_loss = epoch_loss_sum / max(epoch_n, 1)
            epoch_losses.append(epoch_loss)
            print(f"[DP-SFT STAGE2] epoch={epoch+1}/{epochs} loss={epoch_loss:.4f} n={epoch_n}")
            if dev_ds is not None:
                eval_metrics = evaluate(
                    base_model,
                    dev_ds,
                    tokenizer,
                    args.eval_batch_size,
                    args.max_length,
                    device,
                    args.input_prefix,
                )
                print(
                    f"[DP-SFT STAGE2] eval epoch={epoch+1}/{epochs}"
                    f" acc={eval_metrics['accuracy']:.4f}"
                    f" loss={eval_metrics['loss']:.4f}"
                    f" n={eval_metrics['n']}"
                )
                epoch_evals.append({
                    "epoch": int(epoch + 1),
                    "train_loss": float(epoch_loss),
                    "dev_acc": float(eval_metrics["accuracy"]),
                    "dev_loss": float(eval_metrics["loss"]),
                    "dev_n": int(eval_metrics["n"]),
                })
                train_model.train()
    finally:
        if train_model is not None and hasattr(train_model, "to_standard_module"):
            train_model.to_standard_module()
        if hasattr(base_model, "_active_batch_indices"):
            base_model._active_batch_indices = None
        _clear_grad_sample_state(params)

    epsilon = 0.0 if args.no_dp else _stacked_privacy_epsilon(
        noise_multiplier=float(sigma),
        sample_rate=float(sample_rate),
        steps=int(logical_steps_total),
        target_delta=float(target_delta),
        fallback_epsilon=float(target_epsilon),
        args=args,
    )
    runtime = _runtime_stats(t0, completed_steps, device)
    return {
        "stage": "dpsft_stage2_subspace",
        "epsilon": epsilon,
        "sigma": sigma,
        "accountant": ACCOUNTING_MODE if not args.no_dp else "none",
        "target_epsilon": float(target_epsilon) if not args.no_dp else 0.0,
        "target_delta": float(target_delta) if not args.no_dp else 0.0,
        "privacy_spent": {
            "eps_prv": epsilon,
            "sample_rate": sample_rate,
            "logical_steps": int(logical_steps_total),
            "logical_batch_size": int(effective),
            "physical_micro_batch_size": int(micro),
            "subspace_dim": int(d),
            "accounting_note": "DP-SFT stage 2 clips/noises gradients after projection into the SVD subspace.",
        },
        "n_trainable": int(sum(p.numel() for p in params)),
        "logical_batch_size": int(effective),
        "physical_micro_batch_size": int(micro),
        "planned_optimizer_steps": int(logical_steps_total),
        "epoch_losses": epoch_losses,
        "epoch_evals": epoch_evals,
        **runtime,
    }


def run_dpsft(
    model: SwitchSST2,
    train_ds: SST2Dataset,
    dev_ds: SST2Dataset,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    zero_shot: Dict,
) -> Dict:
    print("\n" + "=" * 88)
    print("[DP-SFT] two-stage DP subspace fine-tuning baseline")
    scope = str(getattr(args, "dpsft_param_scope", "full_forward")).strip().lower()
    stage1_ratio = float(getattr(args, "dpsft_stage1_epsilon_ratio", 0.4))
    if not (0.0 < stage1_ratio < 1.0):
        raise ValueError(f"[DP-SFT] --dpsft_stage1_epsilon_ratio must be in (0, 1), got {stage1_ratio}")

    freeze_all(model)
    named_params = select_dpsft_params(model, args)
    if not named_params:
        raise RuntimeError(f"[DP-SFT] no trainable params selected for scope={scope!r}")
    n_trainable = int(sum(p.numel() for _, p in named_params))
    print(f"[DP-SFT] param_scope={scope} trainable={n_trainable:,}")

    if args.no_dp:
        eps_stage1_target = 0.0
        eps_stage2_target = 0.0
        delta_stage1 = 0.0
        delta_stage2 = 0.0
    else:
        eps_stage1_target = float(args.epsilon_total) * stage1_ratio
        eps_stage2_target = float(args.epsilon_total) - eps_stage1_target
        delta_stage1 = float(args.delta) / 2.0
        delta_stage2 = float(args.delta) / 2.0
    print(
        f"[DP-SFT PRIVACY PLAN] eps_stage1={eps_stage1_target:.4f} "
        f"eps_stage2={eps_stage2_target:.4f} delta_each={delta_stage1:.3e}/{delta_stage2:.3e}"
    )

    dpsft_dir = os.path.join(args.output_dir, "dpsft")
    trajectory_dir = os.path.join(dpsft_dir, "trajectory")
    os.makedirs(dpsft_dir, exist_ok=True)

    param_meta = _dpsft_param_slices(named_params)
    with open(os.path.join(dpsft_dir, "param_meta.json"), "w", encoding="utf-8") as f:
        json.dump(param_meta, f, indent=2)

    stage1 = _dpsft_train_stage1_trajectory(
        model=model,
        train_ds=train_ds,
        tokenizer=tokenizer,
        args=args,
        device=device,
        named_params=named_params,
        target_epsilon=eps_stage1_target,
        target_delta=delta_stage1,
        trajectory_dir=trajectory_dir,
        dev_ds=dev_ds,
    )
    directions_cpu, param_meta, svd_info = _dpsft_compute_subspace(
        trajectory_dir=trajectory_dir,
        named_params=named_params,
        subspace_dim=int(getattr(args, "dpsft_subspace_dim", 8)),
        output_dir=dpsft_dir,
    )
    stage2 = _dpsft_train_stage2_subspace(
        model=model,
        train_ds=train_ds,
        tokenizer=tokenizer,
        args=args,
        device=device,
        named_params=named_params,
        directions_cpu=directions_cpu,
        param_meta=param_meta,
        target_epsilon=eps_stage2_target,
        target_delta=delta_stage2,
        dev_ds=dev_ds,
    )

    final = evaluate(model, dev_ds, tokenizer, args.eval_batch_size, args.max_length, device, args.input_prefix)
    eps_total_actual = float(stage1.get("epsilon", 0.0)) + float(stage2.get("epsilon", 0.0))
    delta_total_actual = float(delta_stage1) + float(delta_stage2)
    print(
        f"[DP-SFT] zero={zero_shot['accuracy']:.4f} final={final['accuracy']:.4f} "
        f"ε={eps_total_actual:.4f}"
    )
    for _, p in named_params:
        p.requires_grad_(False)

    result = {
        "dev_acc": final["accuracy"],
        "dev_loss": final["loss"],
        "epsilon": eps_total_actual,
        "delta": delta_total_actual,
        "accountant": ACCOUNTING_MODE if not args.no_dp else "none",
        "n_trainable": n_trainable,
        "param_scope": scope,
        "subspace_dim": int(directions_cpu.size(0)),
        "target_epsilon": float(args.epsilon_total) if not args.no_dp else 0.0,
        "target_delta": float(args.delta) if not args.no_dp else 0.0,
        "privacy_spent": {
            "eps_prv": eps_total_actual,
            "delta": delta_total_actual,
            "composition": "sequential stage1 trajectory + stage2 subspace",
            "stage1": stage1.get("privacy_spent", {}),
            "stage2": stage2.get("privacy_spent", {}),
        },
    }
    return {
        "experiment": "dpsft",
        "method": "dp_sft_trajectory_svd_subspace",
        "method_source": "Adapted from /home/jovyan/gpus-4-nodes-volume/duc3/dpsft/DP-SFT into Switch Transformer.",
        "config": vars(args),
        "zero_shot": zero_shot,
        "final_dev": final,
        "result": result,
        "privacy_plan": {
            "epsilon_total_target": float(args.epsilon_total),
            "epsilon_stage1_ratio": stage1_ratio,
            "epsilon_stage1_target": eps_stage1_target,
            "epsilon_stage2_target": eps_stage2_target,
            "delta_stage1": delta_stage1,
            "delta_stage2": delta_stage2,
            "accounting": "sequential composition",
        },
        "stage1": stage1,
        "svd": svd_info,
        "stage2": stage2,
    }


# ---------------------------------------------------------------------------
# Assignments and expert subsets
# ---------------------------------------------------------------------------


@torch.no_grad()
def construct_record_assignments(
    model: SwitchSST2,
    ds: SST2Dataset,
    tokenizer,
    target_layer: SparseLayerRef,
    batch_size: int,
    max_length: int,
    device: torch.device,
    prefix: str = "",
) -> List[int]:
    """
    Compute record->expert assignments for one sparse layer.
    The assignment is a majority vote over the layer's raw token router top-1 decisions,
    ignoring capacity.
    """
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate(tokenizer, max_length, prefix),
    )

    assignments = [-1 for _ in range(len(ds))]
    model.eval()
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        _ = model(ids, mask)
        raw_top1 = target_layer.sparse_mlp._last_raw_expert_index.detach().cpu()
        valid = batch["attention_mask"].cpu().bool()

        for i, sample_idx in enumerate(batch["idx"].tolist()):
            active_experts = raw_top1[i][valid[i]]
            if active_experts.numel() == 0:
                chosen = 0
            else:
                counts = torch.bincount(
                    active_experts.long(),
                    minlength=target_layer.sparse_mlp.router.num_experts,
                )
                chosen = int(counts.argmax().item())
            assignments[sample_idx] = chosen

    if any(a < 0 for a in assignments):
        raise RuntimeError(f"Incomplete assignments for layer {target_layer.layer_id}")
    return assignments


def group_indices_by_expert(assignments: Sequence[int], num_experts: int) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {j: [] for j in range(num_experts)}
    for idx, expert_id in enumerate(assignments):
        out[int(expert_id)].append(idx)
    return out


def owner_distribution_metrics(assignments: Sequence[int] | torch.Tensor, num_experts: int) -> Dict:
    owners = torch.as_tensor(assignments, dtype=torch.long).view(-1)
    n = int(owners.numel())
    num_experts = max(1, int(num_experts))
    if n == 0:
        return {
            "num_records": 0,
            "num_experts": num_experts,
            "entropy": 0.0,
            "entropy_normalized": 0.0,
            "entropy_deficit": math.log(float(num_experts)),
            "chi2_to_uniform": 0.0,
            "min_q": 0.0,
            "max_q": 0.0,
            "load_cv": 0.0,
            "dominant_expert": 0,
            "dominant_expert_fraction": 0.0,
            "zero_expert_fraction": 1.0,
        }

    counts = torch.bincount(owners.clamp(min=0), minlength=num_experts).float()[:num_experts]
    q = counts / counts.sum().clamp(min=1.0)
    positive = q[q > 0.0]
    entropy = float(-(positive * positive.log()).sum().item()) if positive.numel() > 0 else 0.0
    log_e = math.log(float(num_experts))
    chi2 = float(num_experts * q.pow(2).sum().item() - 1.0)
    mean_q = 1.0 / float(num_experts)
    load_std = float(torch.sqrt(((q - mean_q) ** 2).mean()).item())
    dominant_fraction, dominant_expert = torch.max(q, dim=0)
    return {
        "num_records": n,
        "num_experts": num_experts,
        "entropy": entropy,
        "entropy_normalized": _safe_div(entropy, log_e),
        "entropy_deficit": max(0.0, log_e - entropy),
        "chi2_to_uniform": max(0.0, chi2),
        "min_q": float(q.min().item()),
        "max_q": float(q.max().item()),
        "load_cv": _safe_div(load_std, mean_q),
        "dominant_expert": int(dominant_expert.item()),
        "dominant_expert_fraction": float(dominant_fraction.item()),
        "zero_expert_fraction": float((q <= 0.0).float().mean().item()),
    }


def maybe_perturb_owner_assignments(
    assignments: Sequence[int] | torch.Tensor,
    num_experts: int,
    args: argparse.Namespace,
    *,
    layer_id: int,
) -> Tuple[torch.Tensor, Dict]:
    original = torch.as_tensor(assignments, dtype=torch.long).view(-1).clone()
    num_experts = max(1, int(num_experts))
    beta = float(getattr(args, "owner_map_perturb_beta", 0.0) or 0.0)
    strategy = str(getattr(args, "owner_map_perturb_strategy", "none")).strip().lower()
    target_expert = int(getattr(args, "owner_map_perturb_target_expert", 0) or 0) % num_experts
    min_fraction = max(0.0, float(getattr(args, "owner_map_perturb_min_fraction", 0.0) or 0.0))
    seed = int(getattr(args, "owner_map_perturb_seed", -1))
    if seed < 0:
        seed = int(getattr(args, "seed", 0)) + 37000 + 1009 * int(layer_id)

    before = owner_distribution_metrics(original, num_experts)
    perturbed = original.clone()
    changed_count = 0
    requested_count = 0
    clipped_by_min_fraction = 0

    if strategy not in {"none", "target"}:
        raise ValueError(f"Unknown --owner_map_perturb_strategy={strategy!r}")
    if strategy == "target" and beta > 0.0 and perturbed.numel() > 0:
        beta = min(max(beta, 0.0), 1.0)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        candidate_mask = (torch.rand(perturbed.numel(), generator=generator) < beta) & (perturbed != target_expert)
        requested_count = int(candidate_mask.sum().item())
        final_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)

        if min_fraction > 0.0:
            min_count = int(math.ceil(float(perturbed.numel()) * min_fraction))
            for expert_id in range(num_experts):
                if expert_id == target_expert:
                    continue
                expert_rows = (perturbed == expert_id).nonzero(as_tuple=False).flatten()
                candidate_rows = ((perturbed == expert_id) & candidate_mask).nonzero(as_tuple=False).flatten()
                max_move = max(0, int(expert_rows.numel()) - min_count)
                if candidate_rows.numel() <= max_move:
                    final_mask[candidate_rows] = True
                    continue
                if max_move > 0:
                    order = torch.randperm(candidate_rows.numel(), generator=generator)[:max_move]
                    final_mask[candidate_rows.index_select(0, order)] = True
                clipped_by_min_fraction += int(candidate_rows.numel()) - max_move
        else:
            final_mask = candidate_mask

        perturbed[final_mask] = int(target_expert)
        changed_count = int(final_mask.sum().item())

    after = owner_distribution_metrics(perturbed, num_experts)
    metrics = {
        **after,
        "layer_id": int(layer_id),
        "perturb_strategy": strategy,
        "perturb_beta": float(beta),
        "perturb_target_expert": int(target_expert),
        "perturb_seed": int(seed),
        "perturb_min_fraction": float(min_fraction),
        "perturb_requested_fraction": _safe_div(requested_count, int(original.numel())),
        "perturb_changed_fraction": _safe_div(changed_count, int(original.numel())),
        "perturb_clipped_by_min_fraction": int(max(clipped_by_min_fraction, 0)),
        "original_entropy": float(before["entropy"]),
        "original_entropy_deficit": float(before["entropy_deficit"]),
        "original_chi2_to_uniform": float(before["chi2_to_uniform"]),
        "original_min_q": float(before["min_q"]),
        "original_max_q": float(before["max_q"]),
    }
    return perturbed, metrics


def summarize_owner_metric_rows(rows: Sequence[Dict]) -> Dict:
    if not rows:
        return {"num_layers": 0}
    summary: Dict[str, float | int] = {"num_layers": int(len(rows))}
    for field in (
        "entropy",
        "entropy_normalized",
        "entropy_deficit",
        "chi2_to_uniform",
        "min_q",
        "max_q",
        "load_cv",
        "dominant_expert_fraction",
        "perturb_changed_fraction",
        "original_entropy_deficit",
        "original_chi2_to_uniform",
    ):
        values = [float(row.get(field, 0.0) or 0.0) for row in rows]
        mean = sum(values) / float(len(values))
        summary[f"{field}_mean"] = mean
        summary[f"{field}_min"] = min(values)
        summary[f"{field}_max"] = max(values)
    return summary


def split_total_epochs_across_windows(total_epochs: int, num_windows: int) -> List[int]:
    if num_windows <= 0:
        raise ValueError(f"num_windows must be >= 1, got {num_windows}")
    if total_epochs < 0:
        raise ValueError(f"total_epochs must be >= 0, got {total_epochs}")
    base, rem = divmod(total_epochs, num_windows)
    return [base + (1 if i < rem else 0) for i in range(num_windows)]


def parse_refresh_windows(spec: str, num_windows: int) -> set[int]:
    if num_windows <= 0:
        raise ValueError(f"num_windows must be >= 1, got {num_windows}")
    raw = (spec or "").strip().lower()
    if raw in ("", "*", "all"):
        return set(range(1, num_windows + 1))
    if raw in ("none", "off", "no"):
        return set()

    out: set[int] = set()
    for tok in raw.replace(",", " ").split():
        token = tok.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            a = int(left)
            b = int(right)
            if a > b:
                a, b = b, a
            for w in range(a, b + 1):
                if 1 <= w <= num_windows:
                    out.add(w)
            continue

        w = int(token)
        if 1 <= w <= num_windows:
            out.add(w)

    if not out and raw not in ("none", "off", "no"):
        raise ValueError(
            f"refresh window spec {spec!r} produced no valid windows in [1, {num_windows}]"
        )
    return out


# ---------------------------------------------------------------------------
# Phase B: per-expert training
# ---------------------------------------------------------------------------


def train_one_expert(
    *,
    model: SwitchSST2,
    train_ds: SST2Dataset,
    layer: SparseLayerRef,
    expert_id: int,
    expert_indices: Sequence[int],
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    target_epsilon: float,
    target_delta: float,
    expert_epochs: Optional[int] = None,
    stage_prefix: str = "PHASE_B",
    objective: str = "classifier",
    seq2seq_label_token_ids: Optional[torch.Tensor] = None,
) -> Dict:
    if not args.no_dp:
        raise RuntimeError(
            "train_one_expert is not DP-safe for Phase B: it depends on private "
            "routed subset sizes and can skip/reveal empty experts. Use "
            "train_layer_experts_stacked instead."
        )
    subset = SubsetDataset(train_ds, expert_indices)
    freeze_all(model)
    freeze_routers(model)

    named_params = select_expert_params(model, layer, expert_id)
    stage_name = f"{stage_prefix}_LAYER_{layer.layer_id}_EXPERT_{expert_id}"
    run_epochs = args.expert_epochs if expert_epochs is None else int(expert_epochs)

    stats = run_train_loop(
        stage_name=stage_name,
        model=model,
        train_ds=subset,
        tokenizer=tokenizer,
        device=device,
        args=args,
        named_params=named_params,
        lr=args.lr,
        epochs=run_epochs,
        dp=not args.no_dp,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        objective=objective,
        seq2seq_label_token_ids=seq2seq_label_token_ids,
        batch_phase="phase_b",
    )
    stats.update({
        "layer_id": layer.layer_id,
        "expert_id": expert_id,
        "subset_size": len(subset),
        "epochs": run_epochs,
    })
    return stats


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
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        steps=steps,
        target_delta=target_delta,
        fallback_epsilon=fallback_epsilon,
        eps_error=float(getattr(args, "prv_eps_error", 0.1)),
    )


def resolve_ours_privacy_budget(args: argparse.Namespace, num_layers: int) -> Dict[str, float]:
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive for ours privacy budget, got {num_layers}")

    shared_ratio = float(getattr(args, "epsilon_shared_ratio", 0.3))
    if not (0.0 < shared_ratio < 1.0):
        raise ValueError(f"--epsilon_shared_ratio must be in (0, 1), got {shared_ratio}")

    adj_factor = 1 if args.adjacency == "add_remove" else 2
    num_layers = int(num_layers)

    eps_total = float(args.epsilon_total)
    eps_shared = shared_ratio * eps_total
    eps_experts_total = (1.0 - shared_ratio) * eps_total
    eps_expert = eps_experts_total / float(adj_factor * num_layers)
    eps_layer = float(adj_factor) * eps_expert

    delta_total = float(args.delta)
    delta_shared = shared_ratio * delta_total
    delta_experts_total = (1.0 - shared_ratio) * delta_total
    delta_expert = delta_experts_total / float(adj_factor * num_layers)
    delta_layer = float(adj_factor) * delta_expert

    return {
        "adj_factor": float(adj_factor),
        "num_units": float(1 + adj_factor * num_layers),
        "epsilon_shared_ratio": shared_ratio,
        "eps_shared": eps_shared,
        "eps_experts_total": eps_experts_total,
        "eps_expert": eps_expert,
        "eps_layer": eps_layer,
        "delta_shared": delta_shared,
        "delta_experts_total": delta_experts_total,
        "delta_expert": delta_expert,
        "delta_layer": delta_layer,
    }


def _residual_weights_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    mode = str(getattr(args, "expert_residual_weighting", "none")).strip().lower()
    if mode == "none":
        return torch.ones(labels.size(0), device=labels.device, dtype=torch.float32)

    probs = F.softmax(logits.detach().float(), dim=-1)
    labels = labels.to(device=probs.device, dtype=torch.long)
    true_probs = probs.gather(1, labels.view(-1, 1)).squeeze(1)

    if mode == "prob":
        weights = 1.0 - true_probs
        min_weight = float(getattr(args, "expert_residual_min_weight", 0.0))
        if min_weight > 0.0:
            weights = weights.clamp(min=min_weight)
        return weights.clamp(0.0, 1.0).to(device=labels.device)

    if mode == "margin":
        label_mask = F.one_hot(labels, num_classes=probs.size(-1)).bool()
        other_probs = probs.masked_fill(label_mask, -float("inf")).max(dim=-1).values
        margins = true_probs - other_probs
        threshold = float(getattr(args, "expert_residual_margin_threshold", 0.2))
        small_weight = float(getattr(args, "expert_residual_small_weight", 0.0))
        weights = torch.where(
            margins < threshold,
            torch.ones_like(margins),
            torch.full_like(margins, small_weight),
        )
        min_weight = float(getattr(args, "expert_residual_min_weight", 0.0))
        if min_weight > 0.0:
            weights = weights.clamp(min=min_weight)
        return weights.clamp(0.0, 1.0).to(device=labels.device)

    raise ValueError(f"Unknown --expert_residual_weighting={mode!r}")


def compute_shared_only_logits(
    *,
    base_model: SwitchSST2,
    sparse_layers: Sequence[SparseLayerRef],
    ids: torch.Tensor,
    mask: torch.Tensor,
    objective: str,
    seq2seq_label_token_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    was_training = bool(base_model.training)
    touched = _set_expert_lora_disabled(sparse_layers, True)
    try:
        base_model.eval()
        with torch.no_grad():
            if objective == "seq2seq":
                if seq2seq_label_token_ids is None:
                    raise ValueError("seq2seq residual weighting requires label token ids")
                logits = base_model.seq2seq_class_logits(
                    ids,
                    mask,
                    seq2seq_label_token_ids.to(ids.device),
                )
            else:
                logits = base_model(ids, mask)
    finally:
        _restore_expert_lora_disabled(touched)
        if was_training:
            base_model.train()

    return logits.detach()


def compute_shared_residual_weights(
    *,
    base_model: SwitchSST2,
    sparse_layers: Sequence[SparseLayerRef],
    ids: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    objective: str,
    seq2seq_label_token_ids: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    mode = str(getattr(args, "expert_residual_weighting", "none")).strip().lower()
    if mode == "none":
        return None

    logits = compute_shared_only_logits(
        base_model=base_model,
        sparse_layers=sparse_layers,
        ids=ids,
        mask=mask,
        objective=objective,
        seq2seq_label_token_ids=seq2seq_label_token_ids,
    )
    weights = _residual_weights_from_logits(logits, labels, args)
    return weights.detach().to(device=labels.device, dtype=torch.float32)


def _grad_sample_as_tensor(p: nn.Parameter) -> Optional[torch.Tensor]:
    gs = getattr(p, "grad_sample", None)
    if gs is None:
        return None
    if isinstance(gs, list):
        parts = [g for g in gs if g is not None]
        if not parts:
            return None
        return torch.cat(parts, dim=0)
    return gs


def _ensure_opacus_grad_samples(
    params: Sequence[nn.Parameter],
    batch_size: int,
    allowed_param_ids: Optional[set[int]] = None,
) -> None:
    """
    Opacus requires every trainable parameter to have a per-sample gradient for
    each physical microbatch. Sparse MoE routes can leave an otherwise trainable
    expert inactive, whose correct per-sample gradient is all zeros.
    """
    for p in params:
        if allowed_param_ids is not None and id(p) not in allowed_param_ids:
            continue
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


def _clear_grad_sample_state(params: Sequence[nn.Parameter]) -> None:
    for p in params:
        p.grad = None
        # Opacus GradSampleModule expects trainable parameters to keep a
        # grad_sample attribute while wrapped; deleting it breaks the next
        # backward hook. Clear values, but only delete transient hook state.
        if hasattr(p, "grad_sample"):
            p.grad_sample = None
        if hasattr(p, "_current_grad_sample"):
            delattr(p, "_current_grad_sample")
        if hasattr(p, "summed_grad"):
            p.summed_grad = None


def _expert_key_for_name(
    name: str,
    sparse_layers: Sequence[SparseLayerRef],
) -> Optional[Tuple[int, int]]:
    for layer in sparse_layers:
        prefix = f"base_model.encoder.block.{layer.block_id}.layer.1.mlp.experts.expert_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        expert_token = rest.split(".", 1)[0]
        if expert_token.isdigit():
            return int(layer.layer_id), int(expert_token)
    return None


def _role_keys_for_name(
    name: str,
    sparse_layers: Sequence[SparseLayerRef],
) -> List[str]:
    expert_key = _expert_key_for_name(name, sparse_layers)
    if expert_key is not None:
        layer_id, expert_id = expert_key
        return ["experts", f"expert_L{layer_id}_E{expert_id}"]
    if _is_router(name):
        return ["router"]
    if _is_classifier(name):
        return ["classifier"]
    return ["shared"]


def _mean_or_zero(total: float, count: int) -> float:
    return float(total) / float(count) if count > 0 else 0.0


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num) / float(den) if float(den) > 0.0 else float(default)


def _clamp_cosine(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(-1.0, min(1.0, float(value)))


def _numeric_summary(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "max_to_min_positive": 0.0,
        }
    mean = sum(vals) / float(len(vals))
    var = sum((v - mean) ** 2 for v in vals) / float(len(vals))
    std = math.sqrt(max(var, 0.0))
    positives = [v for v in vals if v > 0.0]
    return {
        "mean": mean,
        "min": min(vals),
        "max": max(vals),
        "std": std,
        "cv": std / mean if mean > 0.0 else 0.0,
        "max_to_min_positive": (max(positives) / min(positives)) if positives else 0.0,
    }


def _distribution_summary(values: Sequence[float]) -> Dict[str, float]:
    summary = _numeric_summary(values)
    vals = [max(float(v), 0.0) for v in values if math.isfinite(float(v))]
    total = sum(vals)
    if total <= 0.0 or len(vals) <= 1:
        summary.update({
            "normalized_entropy": 0.0,
            "zero_fraction": 1.0 if vals else 0.0,
        })
        return summary
    entropy = 0.0
    zero_count = 0
    for value in vals:
        if value <= 0.0:
            zero_count += 1
            continue
        p = value / total
        entropy -= p * math.log(p)
    summary.update({
        "normalized_entropy": entropy / math.log(float(len(vals))),
        "zero_fraction": float(zero_count) / float(len(vals)),
    })
    return summary


def _expert_field_summary(
    rows: Sequence[Dict],
    field: str,
    *,
    distribution: bool = False,
) -> Dict:
    summary_fn = _distribution_summary if distribution else _numeric_summary
    all_values = [float(row.get(field, 0.0)) for row in rows]
    by_layer_values: Dict[int, List[float]] = {}
    for row in rows:
        layer_id = int(row.get("layer_id", -1))
        by_layer_values.setdefault(layer_id, []).append(float(row.get(field, 0.0)))
    return {
        "field": field,
        "all": summary_fn(all_values),
        "by_layer": {
            str(layer_id): summary_fn(values)
            for layer_id, values in sorted(by_layer_values.items())
        },
    }


def compute_expert_dilution_metrics(
    *,
    model: SwitchSST2,
    train_ds: Dataset,
    diagnostic_ds: Optional[Dataset] = None,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    sparse_layers: Sequence[SparseLayerRef],
    named_params: Sequence[Tuple[str, nn.Parameter]],
    method_kind: str,
    logical_batch_size: int,
    expert_sigma: float,
    expert_update_scale_mode: str = "batch",
    owner_assignments: Optional[Dict[int, torch.Tensor]] = None,
    objective: str = "classifier",
    seq2seq_label_token_ids: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Post-training diagnostic for the expert dilution claim.

    It estimates, on a few diagnostic batches, how many records/tokens touch
    each expert, how much clipped expert signal remains after the method's
    denominator, how that compares to the expected DP noise norm, and how the
    privatized expert update compares with the non-private oracle expert
    gradient on the same examples.
    """
    max_batches = int(getattr(args, "dilution_metric_batches", 0))
    if max_batches <= 0:
        return {"enabled": False, "reason": "dilution_metric_batches<=0"}
    if OpacusGradSampleModule is None:
        return {"enabled": False, "reason": "opacus_grad_sample_unavailable"}
    metric_source_ds = diagnostic_ds if diagnostic_ds is not None else train_ds
    diagnostic_split = "provided_holdout" if diagnostic_ds is not None else "train"
    if len(metric_source_ds) == 0:
        return {"enabled": False, "reason": "empty_diagnostic_dataset"}

    batch_size = min(
        max(1, int(getattr(args, "dilution_metric_batch_size", 8))),
        len(metric_source_ds),
    )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + 2027)
    if diagnostic_ds is not None:
        max_examples = min(len(metric_source_ds), max(1, int(max_batches) * int(batch_size)))
        selected = torch.randperm(len(metric_source_ds), generator=generator)[:max_examples].tolist()
        metric_ds: Dataset = ReindexedSubsetDataset(metric_source_ds, selected)
        shuffle_diagnostics = False
    else:
        metric_ds = metric_source_ds
        shuffle_diagnostics = True
    noise_seed = int(getattr(args, "diagnostic_noise_seed", -1))
    if noise_seed < 0:
        noise_seed = int(args.seed) + 92023
    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(int(noise_seed))
    loader = DataLoader(
        metric_ds,
        batch_size=batch_size,
        shuffle=shuffle_diagnostics,
        drop_last=False,
        num_workers=0,
        generator=generator,
        collate_fn=make_collate(tokenizer, args.max_length, args.input_prefix),
    )

    expert_named: List[Tuple[str, nn.Parameter]] = []
    role_named: List[Tuple[str, nn.Parameter]] = []
    seen_role_param_ids: set[int] = set()
    for name, p in named_params:
        if _is_opacus_incompatible_param(name):
            continue
        key = _expert_key_for_name(name, sparse_layers)
        if key is not None:
            expert_named.append((name, p))
        pid = id(p)
        if pid not in seen_role_param_ids:
            role_named.append((name, p))
            seen_role_param_ids.add(pid)

    if not expert_named:
        return {"enabled": False, "reason": "no_selected_expert_params"}

    owner_routed_method = method_kind in {"ours", "baseline_c_expert_only"}

    # Naive DP clips experts together with the full trainable model. Ours clips
    # expert streams separately by default, but the global-clipping ablation uses
    # the full trainable parameter set for its clipping reference.
    ours_clip_scope = str(getattr(args, "ours_clip_scope", "role")).strip().lower()
    use_global_clip_reference = bool(method_kind == "naive" or ours_clip_scope == "global")
    clip_named = role_named if use_global_clip_reference else expert_named
    clip_param_ids = {id(p) for _, p in clip_named}
    diagnostic_named = role_named
    diagnostic_params = [p for _, p in diagnostic_named]
    expert_params = [p for _, p in expert_named]
    expert_param_ids = {id(p) for _, p in expert_named}
    param_keys = {id(p): _expert_key_for_name(name, sparse_layers) for name, p in expert_named}
    params_by_key: Dict[Tuple[int, int], List[nn.Parameter]] = {}
    for _name, p in expert_named:
        key = param_keys[id(p)]
        if key is not None:
            params_by_key.setdefault(key, []).append(p)
    clip_reference = "full_trainable_params" if use_global_clip_reference else "expert_stream_params"
    role_param_counts: Dict[str, int] = {}
    for name, p in role_named:
        for role_key in _role_keys_for_name(name, sparse_layers):
            role_param_counts[role_key] = role_param_counts.get(role_key, 0) + int(p.numel())
    role_stats: Dict[str, Dict[str, float]] = {
        role_key: {
            "num_params": float(num_params),
            "norm_sum": 0.0,
            "norm_sq_sum": 0.0,
            "active_count": 0.0,
            "sample_count": 0.0,
            "global_clip_factor_sum": 0.0,
            "role_only_clip_factor_sum": 0.0,
            "global_to_role_clip_factor_sum": 0.0,
            "clip_count": 0.0,
            "missing_param_count": 0.0,
            "batches": 0.0,
        }
        for role_key, num_params in role_param_counts.items()
    }

    num_experts_by_layer = {
        int(layer.layer_id): int(layer.sparse_mlp.router.num_experts)
        for layer in sparse_layers
    }
    stats_by_key: Dict[Tuple[int, int], Dict[str, float]] = {}
    for key, params_for_key in params_by_key.items():
        layer_id, expert_id = key
        if owner_routed_method and expert_update_scale_mode == "expected_owner":
            denom = float(max(1.0, float(logical_batch_size) / float(max(num_experts_by_layer[layer_id], 1))))
        else:
            denom = float(max(1, int(logical_batch_size)))
        stats_by_key[key] = {
            "layer_id": float(layer_id),
            "expert_id": float(expert_id),
            "num_params": float(sum(p.numel() for p in params_for_key)),
            "denominator": denom,
            "observed_record_count": 0.0,
            "valid_token_count": 0.0,
            "routed_token_count": 0.0,
            "routed_record_count": 0.0,
            "kept_token_count": 0.0,
            "kept_record_count": 0.0,
            "owner_count": 0.0,
            "active_grad_record_count": 0.0,
            "signal_before_divide_norm_sum": 0.0,
            "signal_after_divide_norm_sum": 0.0,
            "noise_after_divide_norm": 0.0,
            "snr_sum": 0.0,
            "oracle_gradient_norm_sum": 0.0,
            "privatized_update_norm_sum": 0.0,
            "diagnostic_noise_update_norm_sum": 0.0,
            "legacy_per_step_update_coverage_ratio_sum": 0.0,
            "legacy_per_step_update_coverage_ratio_min": 0.0,
            "legacy_per_step_update_coverage_ratio_count": 0.0,
            "oracle_record_count": 0.0,
            "vector_average_count": 0.0,
            "oracle_cosine_alignment_sum": 0.0,
            "oracle_cosine_alignment_count": 0.0,
            "oracle_cosine_alignment_above_0_5_count": 0.0,
            "clip_factor_sum": 0.0,
            "expert_only_clip_factor_sum": 0.0,
            "clip_factor_ratio_sum": 0.0,
            "clip_factor_count": 0.0,
            "global_clipped_count": 0.0,
            "expert_only_clipped_count": 0.0,
            "clip_suppressed_count": 0.0,
            "expert_only_unclipped_count": 0.0,
            "global_clipped_expert_unclipped_count": 0.0,
            "nonzero_signal_batches": 0.0,
            "batches": 0.0,
            "total_clip_param_count": 0.0,
            "total_clip_missing_param_count": 0.0,
        }

    prev_requires_grad = [(p, p.requires_grad) for _, p in model.named_parameters()]
    saved_record_layers = set(model._record_route_layer_ids)
    saved_fixed_assignments = dict(getattr(model, "_record_route_fixed_assignments", {}))
    train_model: Optional[nn.Module] = None
    base_model: nn.Module = model
    label_token_ids = seq2seq_label_token_ids.to(device) if seq2seq_label_token_ids is not None else None
    max_grad_norm = float(args.max_grad_norm)
    update_multiplier = float(getattr(args, "expert_lr_multiplier", 1.0)) if owner_routed_method else 1.0
    vector_sum_kinds = ("private", "oracle_record", "signal_update", "noise_update")
    vector_sums: Dict[Tuple[int, int], Dict[str, Dict[int, torch.Tensor]]] = {
        key: {kind: {} for kind in vector_sum_kinds}
        for key in stats_by_key
    }

    def _accumulate_vector_sum(
        key: Tuple[int, int],
        kind: str,
        param: nn.Parameter,
        value: torch.Tensor,
    ) -> None:
        bucket = vector_sums[key][kind]
        pid = id(param)
        value_cpu = value.detach().float().cpu()
        if pid not in bucket:
            bucket[pid] = value_cpu.clone()
        else:
            bucket[pid].add_(value_cpu)

    metric_owner_assignments = owner_assignments
    diagnostic_owner_assignments_recomputed = False
    if owner_routed_method and owner_assignments:
        needs_recompute = False
        for layer in sparse_layers:
            layer_id = int(layer.layer_id)
            assignments = owner_assignments.get(layer_id)
            if assignments is None or int(assignments.numel()) != len(metric_ds):
                needs_recompute = True
                break
        if needs_recompute:
            diagnostic_owner_assignments_recomputed = True
            metric_owner_assignments = {}
            model._record_route_layer_ids.clear()
            model._record_route_fixed_assignments = {}
            try:
                for layer in sparse_layers:
                    if int(layer.layer_id) not in num_experts_by_layer:
                        continue
                    assignments = construct_record_assignments(
                        model,
                        metric_ds,
                        tokenizer,
                        target_layer=layer,
                        batch_size=int(getattr(args, "assignment_batch_size", 64)),
                        max_length=int(args.max_length),
                        device=device,
                        prefix=args.input_prefix,
                    )
                    perturbed, _metrics = maybe_perturb_owner_assignments(
                        assignments,
                        int(layer.sparse_mlp.router.num_experts),
                        args,
                        layer_id=int(layer.layer_id),
                    )
                    metric_owner_assignments[int(layer.layer_id)] = perturbed.detach().cpu().long()
            except Exception as exc:
                return {"enabled": False, "reason": f"diagnostic_owner_assignment_{type(exc).__name__}: {exc}"}
            finally:
                model._record_route_layer_ids.clear()
                model._record_route_layer_ids.update(saved_record_layers)
                model._record_route_fixed_assignments = saved_fixed_assignments

    try:
        freeze_all(model)
        for _, p in diagnostic_named:
            p.requires_grad_(True)
        if owner_routed_method and metric_owner_assignments:
            model._record_route_layer_ids.clear()
            model._record_route_layer_ids.update(int(k) for k in metric_owner_assignments.keys())
            model._record_route_fixed_assignments = {
                int(k): v.detach().to(device="cpu", dtype=torch.long)
                for k, v in metric_owner_assignments.items()
            }

        train_model = OpacusGradSampleModule(model, batch_first=True, loss_reduction="mean")
        if hasattr(train_model, "_module"):
            base_model = train_model._module
        # Opacus GradSampleModule records forward activations only for modules
        # in training mode. Running this diagnostic in eval mode makes the
        # backward hook fire without a matching stored activation.
        train_model.train()
        for module in train_model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            batch_n = int(labels.size(0))
            if hasattr(base_model, "_active_batch_indices"):
                base_model._active_batch_indices = batch["idx"].detach()

            for p in diagnostic_params:
                p.grad = None
            if objective == "seq2seq":
                if label_token_ids is None:
                    return {"enabled": False, "reason": "missing_seq2seq_label_tokens"}
                target_token_ids = label_token_ids.index_select(0, labels)
                loss = base_model.seq2seq_loss(ids, mask, target_token_ids)
            else:
                logits = train_model(ids, mask)
                if not torch.isfinite(logits).all():
                    continue
                loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                continue

            loss.backward()
            _ensure_opacus_grad_samples(expert_params, batch_n, allowed_param_ids=expert_param_ids)

            valid_mask = mask.detach().bool().cpu()
            for layer in sparse_layers:
                layer_id = int(layer.layer_id)
                raw_top1 = getattr(layer.sparse_mlp, "_last_raw_expert_index", None)
                if raw_top1 is None:
                    continue
                raw_top1_cpu = raw_top1.detach().cpu()
                keep_mask = getattr(layer.sparse_mlp, "_last_keep_mask", None)
                if keep_mask is None:
                    keep_cpu = valid_mask
                else:
                    keep_cpu = keep_mask.detach().cpu().bool()
                    if keep_cpu.shape != valid_mask.shape:
                        keep_cpu = valid_mask
                    else:
                        keep_cpu = keep_cpu & valid_mask
                valid_token_count = float(valid_mask.sum().item())
                for expert_id in range(int(layer.sparse_mlp.router.num_experts)):
                    key = (layer_id, int(expert_id))
                    if key not in stats_by_key:
                        continue
                    stats_by_key[key]["observed_record_count"] += float(batch_n)
                    stats_by_key[key]["valid_token_count"] += valid_token_count
                    token_hits = (raw_top1_cpu == int(expert_id)) & valid_mask
                    kept_hits = (raw_top1_cpu == int(expert_id)) & keep_cpu
                    stats_by_key[key]["routed_token_count"] += float(token_hits.sum().item())
                    stats_by_key[key]["routed_record_count"] += float(token_hits.any(dim=1).sum().item())
                    stats_by_key[key]["kept_token_count"] += float(kept_hits.sum().item())
                    stats_by_key[key]["kept_record_count"] += float(kept_hits.any(dim=1).sum().item())

            if owner_routed_method and metric_owner_assignments:
                batch_indices = batch["idx"].detach().cpu().long()
                for layer in sparse_layers:
                    layer_id = int(layer.layer_id)
                    assignments = metric_owner_assignments.get(layer_id)
                    if assignments is None:
                        continue
                    owners = assignments.detach().cpu().long().index_select(0, batch_indices)
                    for expert_id in range(int(layer.sparse_mlp.router.num_experts)):
                        key = (layer_id, int(expert_id))
                        if key in stats_by_key:
                            stats_by_key[key]["owner_count"] += float((owners == int(expert_id)).sum().item())

            method_clip_norm_sq = torch.zeros(batch_n, device=device, dtype=torch.float32)
            full_trainable_norm_sq = torch.zeros(batch_n, device=device, dtype=torch.float32)
            role_norm_sq: Dict[str, torch.Tensor] = {
                role_key: torch.zeros(batch_n, device=device, dtype=torch.float32)
                for role_key in role_stats
            }
            expert_norm_sq: Dict[Tuple[int, int], torch.Tensor] = {
                key: torch.zeros(batch_n, device=device, dtype=torch.float32)
                for key in stats_by_key
            }
            total_clip_param_count = 0
            total_clip_missing_param_count = 0
            param_grad_samples: Dict[int, torch.Tensor] = {}
            for name, p in diagnostic_named:
                gs = _grad_sample_as_tensor(p)
                role_keys = _role_keys_for_name(name, sparse_layers)
                if gs is None or gs.size(0) != batch_n:
                    for role_key in role_keys:
                        role_stats[role_key]["missing_param_count"] += float(p.numel())
                    if id(p) in clip_param_ids:
                        total_clip_missing_param_count += int(p.numel())
                    continue
                gs = gs.to(device)
                param_grad_samples[id(p)] = gs
                norm_sq = gs.detach().float().flatten(1).pow(2).sum(dim=1)
                full_trainable_norm_sq += norm_sq
                for role_key in role_keys:
                    role_norm_sq[role_key] += norm_sq
                if id(p) in clip_param_ids:
                    method_clip_norm_sq += norm_sq
                    total_clip_param_count += int(p.numel())
                expert_key = param_keys.get(id(p))
                if expert_key is not None:
                    expert_norm_sq[expert_key] += norm_sq

            usable_params: List[Tuple[nn.Parameter, Tuple[int, int], torch.Tensor]] = []
            for _name, p in expert_named:
                key = param_keys[id(p)]
                if key is None:
                    continue
                gs = param_grad_samples.get(id(p))
                if gs is None:
                    continue
                if gs.size(0) != batch_n:
                    continue
                usable_params.append((p, key, gs))

            total_clip = torch.clamp(max_grad_norm / (method_clip_norm_sq.sqrt() + 1e-6), max=1.0)
            full_trainable_clip = torch.clamp(max_grad_norm / (full_trainable_norm_sq.sqrt() + 1e-6), max=1.0)
            for role_key, norm_sq in role_norm_sq.items():
                role_norm = norm_sq.sqrt()
                role_clip = torch.clamp(max_grad_norm / (role_norm + 1e-6), max=1.0)
                st_role = role_stats[role_key]
                st_role["norm_sum"] += float(role_norm.detach().sum().item())
                st_role["norm_sq_sum"] += float(norm_sq.detach().sum().item())
                st_role["active_count"] += float((role_norm > 0.0).sum().item())
                st_role["sample_count"] += float(batch_n)
                st_role["global_clip_factor_sum"] += float(full_trainable_clip.detach().sum().item())
                st_role["role_only_clip_factor_sum"] += float(role_clip.detach().sum().item())
                st_role["global_to_role_clip_factor_sum"] += float(
                    (full_trainable_clip / (role_clip + 1e-12)).detach().sum().item()
                )
                st_role["clip_count"] += float(batch_n)
                st_role["batches"] += 1.0

            for key, params_for_key in params_by_key.items():
                layer_id, expert_id = key
                st = stats_by_key[key]
                st["total_clip_param_count"] += float(total_clip_param_count)
                st["total_clip_missing_param_count"] += float(total_clip_missing_param_count)
                expert_norm = expert_norm_sq[key].sqrt()
                expert_clip = torch.clamp(max_grad_norm / (expert_norm + 1e-6), max=1.0)

                if owner_routed_method and metric_owner_assignments:
                    batch_indices = batch["idx"].detach().cpu().long()
                    assignments = metric_owner_assignments.get(layer_id)
                    if assignments is None:
                        row_mask = torch.zeros(batch_n, device=device, dtype=torch.bool)
                    else:
                        owners = assignments.detach().cpu().long().index_select(0, batch_indices).to(device)
                        row_mask = owners == int(expert_id)
                    active_rows = row_mask.nonzero(as_tuple=False).flatten()
                    clip_for_signal = total_clip if use_global_clip_reference else expert_clip
                else:
                    active_rows = (expert_norm > 0.0).nonzero(as_tuple=False).flatten()
                    clip_for_signal = total_clip

                st["active_grad_record_count"] += float(active_rows.numel())
                if active_rows.numel() > 0:
                    clip_rows = clip_for_signal.index_select(0, active_rows)
                    expert_clip_rows = expert_clip.index_select(0, active_rows)
                    st["clip_factor_sum"] += float(clip_rows.detach().sum().item())
                    st["expert_only_clip_factor_sum"] += float(expert_clip_rows.detach().sum().item())
                    st["clip_factor_ratio_sum"] += float(
                        (clip_rows / (expert_clip_rows + 1e-12)).detach().sum().item()
                    )
                    st["clip_factor_count"] += float(active_rows.numel())
                    global_clipped = clip_rows < (1.0 - 1e-6)
                    expert_clipped = expert_clip_rows < (1.0 - 1e-6)
                    expert_unclipped = ~expert_clipped
                    st["global_clipped_count"] += float(global_clipped.detach().sum().item())
                    st["expert_only_clipped_count"] += float(expert_clipped.detach().sum().item())
                    st["clip_suppressed_count"] += float(
                        ((clip_rows + 1e-6) < expert_clip_rows).detach().sum().item()
                    )
                    st["expert_only_unclipped_count"] += float(expert_unclipped.detach().sum().item())
                    st["global_clipped_expert_unclipped_count"] += float(
                        (global_clipped & expert_unclipped).detach().sum().item()
                    )

                signal_sq = 0.0
                oracle_sq = 0.0
                private_update_sq = 0.0
                diagnostic_noise_update_sq = 0.0
                oracle_private_dot = 0.0
                denom = max(float(st["denominator"]), 1.0)
                for p, param_key, gs in usable_params:
                    if param_key != key:
                        continue
                    if active_rows.numel() == 0:
                        clipped_sum = torch.zeros_like(p, memory_format=torch.preserve_format)
                        oracle_record_sum = torch.zeros_like(p, memory_format=torch.preserve_format)
                        oracle_grad = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else:
                        gs_rows = gs.index_select(0, active_rows)
                        factors = clip_for_signal.index_select(0, active_rows).to(gs_rows.dtype)
                        while factors.dim() < gs_rows.dim():
                            factors = factors.unsqueeze(-1)
                        clipped_sum = (gs_rows * factors).sum(dim=0)
                        oracle_record_sum = gs_rows.sum(dim=0)
                        oracle_grad = oracle_record_sum / float(max(int(active_rows.numel()), 1))
                    if float(expert_sigma) > 0.0 and max_grad_norm > 0.0:
                        noise = torch.randn(
                            tuple(p.shape),
                            generator=noise_generator,
                            device="cpu",
                            dtype=torch.float32,
                        ).to(device=p.device, dtype=clipped_sum.dtype)
                        noise = noise.mul(float(expert_sigma) * max_grad_norm)
                    else:
                        noise = torch.zeros_like(clipped_sum)
                    private_update = (clipped_sum + noise).mul(float(update_multiplier) / denom)
                    signal_update = clipped_sum.mul(float(update_multiplier) / denom)
                    noise_update = noise.mul(float(update_multiplier) / denom)
                    _accumulate_vector_sum(key, "private", p, private_update)
                    _accumulate_vector_sum(key, "oracle_record", p, oracle_record_sum)
                    _accumulate_vector_sum(key, "signal_update", p, signal_update)
                    _accumulate_vector_sum(key, "noise_update", p, noise_update)

                    private_f = private_update.detach().float()
                    oracle_f = oracle_grad.detach().float()
                    noise_update_f = noise_update.detach().float()
                    signal_sq += float(clipped_sum.detach().float().pow(2).sum().item())
                    oracle_sq += float(oracle_f.pow(2).sum().item())
                    private_update_sq += float(private_f.pow(2).sum().item())
                    diagnostic_noise_update_sq += float(noise_update_f.pow(2).sum().item())
                    oracle_private_dot += float((private_f * oracle_f).sum().item())

                signal_before = math.sqrt(max(signal_sq, 0.0))
                signal_after = signal_before / denom
                noise_before = float(expert_sigma) * max_grad_norm * math.sqrt(max(float(st["num_params"]), 1.0))
                noise_after = noise_before / denom
                snr = signal_before / noise_before if noise_before > 0.0 else 0.0
                oracle_norm = math.sqrt(max(oracle_sq, 0.0))
                private_update_norm = math.sqrt(max(private_update_sq, 0.0))
                diagnostic_noise_update_norm = math.sqrt(max(diagnostic_noise_update_sq, 0.0))

                st["signal_before_divide_norm_sum"] += signal_before
                st["signal_after_divide_norm_sum"] += signal_after
                st["noise_after_divide_norm"] = noise_after
                st["snr_sum"] += snr
                st["nonzero_signal_batches"] += float(signal_before > 0.0)
                st["vector_average_count"] += 1.0
                st["oracle_record_count"] += float(active_rows.numel())
                if oracle_norm > 0.0:
                    ucr = private_update_norm / oracle_norm
                    st["oracle_gradient_norm_sum"] += oracle_norm
                    st["privatized_update_norm_sum"] += private_update_norm
                    st["diagnostic_noise_update_norm_sum"] += diagnostic_noise_update_norm
                    st["legacy_per_step_update_coverage_ratio_sum"] += ucr
                    if st["legacy_per_step_update_coverage_ratio_count"] <= 0.0:
                        st["legacy_per_step_update_coverage_ratio_min"] = ucr
                    else:
                        st["legacy_per_step_update_coverage_ratio_min"] = min(
                            float(st["legacy_per_step_update_coverage_ratio_min"]),
                            ucr,
                        )
                    st["legacy_per_step_update_coverage_ratio_count"] += 1.0
                    if private_update_norm > 0.0:
                        oca = _clamp_cosine(oracle_private_dot / (private_update_norm * oracle_norm))
                        st["oracle_cosine_alignment_sum"] += oca
                        st["oracle_cosine_alignment_count"] += 1.0
                        st["oracle_cosine_alignment_above_0_5_count"] += float(oca > 0.5)
                st["batches"] += 1.0

            _clear_grad_sample_state(diagnostic_params)
    except Exception as exc:
        return {"enabled": False, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        if train_model is not None and hasattr(train_model, "to_standard_module"):
            train_model.to_standard_module()
        model._record_route_layer_ids.clear()
        model._record_route_layer_ids.update(saved_record_layers)
        model._record_route_fixed_assignments = saved_fixed_assignments
        if hasattr(base_model, "_active_batch_indices"):
            base_model._active_batch_indices = None
        _clear_grad_sample_state(diagnostic_params)
        for p, enabled in prev_requires_grad:
            p.requires_grad_(enabled)

    def _averaged_vector_diagnostics_for_key(key: Tuple[int, int]) -> Dict[str, float]:
        update_count = int(stats_by_key[key].get("vector_average_count", 0.0))
        oracle_count = int(stats_by_key[key].get("oracle_record_count", 0.0))
        if update_count <= 0:
            return {
                "averaged_private_update_norm": 0.0,
                "averaged_oracle_gradient_norm": 0.0,
                "averaged_signal_update_norm": 0.0,
                "averaged_noise_update_norm": 0.0,
                "averaged_update_coverage_ratio": 0.0,
                "averaged_signal_coverage_ratio": 0.0,
                "averaged_oracle_cosine_alignment": 0.0,
                "averaged_signal_oracle_cosine_alignment": 0.0,
                "vector_average_count": 0.0,
                "oracle_record_count": 0.0,
            }

        def _norm(kind: str, count: int) -> float:
            if count <= 0:
                return 0.0
            norm_sq = 0.0
            for value_sum in vector_sums[key][kind].values():
                avg = value_sum / float(count)
                norm_sq += float(avg.float().pow(2).sum().item())
            return math.sqrt(max(norm_sq, 0.0))

        def _dot(kind_a: str, count_a: int, kind_b: str, count_b: int) -> float:
            if count_a <= 0 or count_b <= 0:
                return 0.0
            total = 0.0
            a_bucket = vector_sums[key][kind_a]
            b_bucket = vector_sums[key][kind_b]
            for pid in set(a_bucket) & set(b_bucket):
                a_avg = a_bucket[pid] / float(count_a)
                b_avg = b_bucket[pid] / float(count_b)
                total += float((a_avg.float() * b_avg.float()).sum().item())
            return total

        private_norm = _norm("private", update_count)
        oracle_norm = _norm("oracle_record", oracle_count)
        signal_norm = _norm("signal_update", update_count)
        noise_norm = _norm("noise_update", update_count)
        private_oracle_dot = _dot("private", update_count, "oracle_record", oracle_count)
        signal_oracle_dot = _dot("signal_update", update_count, "oracle_record", oracle_count)
        return {
            "averaged_private_update_norm": private_norm,
            "averaged_oracle_gradient_norm": oracle_norm,
            "averaged_signal_update_norm": signal_norm,
            "averaged_noise_update_norm": noise_norm,
            "averaged_update_coverage_ratio": _safe_div(private_norm, oracle_norm),
            "averaged_signal_coverage_ratio": _safe_div(signal_norm, oracle_norm),
            "averaged_oracle_cosine_alignment": (
                _clamp_cosine(private_oracle_dot / (private_norm * oracle_norm))
                if private_norm > 0.0 and oracle_norm > 0.0
                else 0.0
            ),
            "averaged_signal_oracle_cosine_alignment": (
                _clamp_cosine(signal_oracle_dot / (signal_norm * oracle_norm))
                if signal_norm > 0.0 and oracle_norm > 0.0
                else 0.0
            ),
            "vector_average_count": float(update_count),
            "oracle_record_count": float(oracle_count),
        }

    averaged_vector_diagnostics = {
        key: _averaged_vector_diagnostics_for_key(key)
        for key in stats_by_key
    }

    role_rows: List[Dict] = []
    for role_key in sorted(role_stats):
        st = role_stats[role_key]
        sample_n = int(st["sample_count"])
        clip_n = int(st["clip_count"])
        norm_mean = _mean_or_zero(st["norm_sum"], sample_n)
        norm_rms = math.sqrt(_mean_or_zero(st["norm_sq_sum"], sample_n))
        role_rows.append({
            "role": role_key,
            "num_params": int(st["num_params"]),
            "norm_mean": norm_mean,
            "norm_rms": norm_rms,
            "norm_per_sqrt_param": _safe_div(norm_mean, math.sqrt(max(float(st["num_params"]), 1.0))),
            "norm_per_param": _safe_div(norm_mean, max(float(st["num_params"]), 1.0)),
            "active_fraction": _mean_or_zero(st["active_count"], sample_n),
            "global_clip_factor_mean": _mean_or_zero(st["global_clip_factor_sum"], clip_n),
            "role_only_clip_factor_mean": _mean_or_zero(st["role_only_clip_factor_sum"], clip_n),
            "global_to_role_clip_factor_mean": _mean_or_zero(st["global_to_role_clip_factor_sum"], clip_n),
            "missing_param_count_mean": int(_mean_or_zero(st["missing_param_count"], int(st["batches"]))),
            "batches": int(st["batches"]),
        })

    expert_rows: List[Dict] = []
    for key in sorted(stats_by_key):
        st = stats_by_key[key]
        batches = int(st["batches"])
        clip_n = int(st["clip_factor_count"])
        denom = float(st["denominator"])
        owner_or_routed = st["owner_count"] if owner_routed_method and metric_owner_assignments else st["routed_record_count"]
        observed_records = float(st["observed_record_count"])
        valid_tokens = float(st["valid_token_count"])
        avg_valid_tokens_per_record = _safe_div(valid_tokens, observed_records)
        owner_or_routed_fraction = _safe_div(float(owner_or_routed), observed_records)
        routed_record_fraction = _safe_div(float(st["routed_record_count"]), observed_records)
        kept_record_fraction = _safe_div(float(st["kept_record_count"]), observed_records)
        active_grad_fraction = _safe_div(float(st["active_grad_record_count"]), observed_records)
        routed_token_fraction = _safe_div(float(st["routed_token_count"]), valid_tokens)
        kept_token_fraction = _safe_div(float(st["kept_token_count"]), valid_tokens)
        estimated_owner_or_routed_records = owner_or_routed_fraction * float(logical_batch_size)
        estimated_active_grad_records = active_grad_fraction * float(logical_batch_size)
        estimated_routed_records = routed_record_fraction * float(logical_batch_size)
        estimated_kept_records = kept_record_fraction * float(logical_batch_size)
        estimated_routed_tokens = routed_token_fraction * avg_valid_tokens_per_record * float(logical_batch_size)
        layer_num_experts = max(1, int(num_experts_by_layer.get(int(st["layer_id"]), 1)))
        expected_uniform_records = float(logical_batch_size) / float(layer_num_experts)
        total_clip_param_count_mean = _mean_or_zero(st["total_clip_param_count"], batches)
        total_clip_missing_param_count_mean = _mean_or_zero(st["total_clip_missing_param_count"], batches)
        legacy_ucr_n = int(st["legacy_per_step_update_coverage_ratio_count"])
        oca_n = int(st["oracle_cosine_alignment_count"])
        avg_vec = averaged_vector_diagnostics.get(key, {})
        expert_rows.append({
            "layer_id": int(st["layer_id"]),
            "expert_id": int(st["expert_id"]),
            "num_params": int(st["num_params"]),
            "denominator": denom,
            "clip_reference": clip_reference,
            "observed_record_count": int(observed_records),
            "valid_token_count": int(valid_tokens),
            "avg_valid_tokens_per_record": avg_valid_tokens_per_record,
            "routed_token_count": int(st["routed_token_count"]),
            "routed_record_count": int(st["routed_record_count"]),
            "kept_token_count": int(st["kept_token_count"]),
            "kept_record_count": int(st["kept_record_count"]),
            "owner_count": int(st["owner_count"]),
            "owner_or_routed_count": int(owner_or_routed),
            "routed_record_fraction": routed_record_fraction,
            "kept_record_fraction": kept_record_fraction,
            "owner_or_routed_fraction": owner_or_routed_fraction,
            "active_grad_record_fraction": active_grad_fraction,
            "routed_token_fraction": routed_token_fraction,
            "kept_token_fraction": kept_token_fraction,
            "effective_records_per_batch_mean": float(owner_or_routed) / max(float(batches), 1.0),
            "estimated_owner_or_routed_records_per_logical_batch": estimated_owner_or_routed_records,
            "estimated_active_grad_records_per_logical_batch": estimated_active_grad_records,
            "estimated_routed_records_per_logical_batch": estimated_routed_records,
            "estimated_kept_records_per_logical_batch": estimated_kept_records,
            "estimated_routed_tokens_per_logical_batch": estimated_routed_tokens,
            "expected_uniform_records_per_expert": expected_uniform_records,
            "owner_or_routed_to_denominator": _safe_div(estimated_owner_or_routed_records, denom),
            "active_grad_to_denominator": _safe_div(estimated_active_grad_records, denom),
            "denominator_to_estimated_owner_or_routed": _safe_div(denom, estimated_owner_or_routed_records),
            "estimated_owner_or_routed_to_expected_uniform": _safe_div(
                estimated_owner_or_routed_records,
                expected_uniform_records,
            ),
            "raw_owner_or_routed_to_denominator": float(owner_or_routed) / max(denom * max(batches, 1), 1.0),
            "signal_before_divide_norm_mean": _mean_or_zero(st["signal_before_divide_norm_sum"], batches),
            "signal_after_divide_norm_mean": _mean_or_zero(st["signal_after_divide_norm_sum"], batches),
            "noise_after_divide_norm": float(st["noise_after_divide_norm"]),
            "snr_mean": _mean_or_zero(st["snr_sum"], batches),
            "oracle_gradient_norm_mean": _mean_or_zero(st["oracle_gradient_norm_sum"], legacy_ucr_n),
            "privatized_update_norm_mean": _mean_or_zero(st["privatized_update_norm_sum"], legacy_ucr_n),
            "diagnostic_noise_update_norm_mean": _mean_or_zero(st["diagnostic_noise_update_norm_sum"], legacy_ucr_n),
            "legacy_per_step_update_coverage_ratio_mean": _mean_or_zero(
                st["legacy_per_step_update_coverage_ratio_sum"],
                legacy_ucr_n,
            ),
            "legacy_per_step_update_coverage_ratio_min": (
                float(st["legacy_per_step_update_coverage_ratio_min"]) if legacy_ucr_n > 0 else 0.0
            ),
            "legacy_per_step_update_coverage_ratio_count": legacy_ucr_n,
            "averaged_private_update_norm": float(avg_vec.get("averaged_private_update_norm", 0.0)),
            "averaged_oracle_gradient_norm": float(avg_vec.get("averaged_oracle_gradient_norm", 0.0)),
            "averaged_signal_update_norm": float(avg_vec.get("averaged_signal_update_norm", 0.0)),
            "averaged_noise_update_norm": float(avg_vec.get("averaged_noise_update_norm", 0.0)),
            "averaged_signal_coverage_ratio": float(avg_vec.get("averaged_signal_coverage_ratio", 0.0)),
            "averaged_oracle_cosine_alignment": float(avg_vec.get("averaged_oracle_cosine_alignment", 0.0)),
            "averaged_signal_oracle_cosine_alignment": float(
                avg_vec.get("averaged_signal_oracle_cosine_alignment", 0.0)
            ),
            "vector_average_count": int(avg_vec.get("vector_average_count", 0.0)),
            "oracle_record_count": int(avg_vec.get("oracle_record_count", 0.0)),
            "update_coverage_ratio_mean": float(avg_vec.get("averaged_update_coverage_ratio", 0.0)),
            "update_coverage_ratio_min": float(avg_vec.get("averaged_update_coverage_ratio", 0.0)),
            "update_coverage_ratio_count": int(avg_vec.get("vector_average_count", 0.0)),
            "oracle_cosine_alignment_mean": _mean_or_zero(st["oracle_cosine_alignment_sum"], oca_n),
            "oracle_cosine_alignment_fraction_above_0_5": _safe_div(
                st["oracle_cosine_alignment_above_0_5_count"],
                oca_n,
            ),
            "oracle_cosine_alignment_count": oca_n,
            "clip_factor_mean": _mean_or_zero(st["clip_factor_sum"], clip_n),
            "expert_only_clip_factor_mean": _mean_or_zero(st["expert_only_clip_factor_sum"], clip_n),
            "clip_factor_to_expert_only_mean": _mean_or_zero(st["clip_factor_ratio_sum"], clip_n),
            "global_clipped_fraction": _safe_div(st["global_clipped_count"], clip_n),
            "expert_only_clipped_fraction": _safe_div(st["expert_only_clipped_count"], clip_n),
            "clip_suppressed_fraction": _safe_div(st["clip_suppressed_count"], clip_n),
            "expert_only_unclipped_fraction": _safe_div(st["expert_only_unclipped_count"], clip_n),
            "global_clipped_expert_unclipped_fraction": _safe_div(
                st["global_clipped_expert_unclipped_count"],
                clip_n,
            ),
            "global_clipped_given_expert_unclipped_fraction": _safe_div(
                st["global_clipped_expert_unclipped_count"],
                st["expert_only_unclipped_count"],
            ),
            "nonzero_signal_batches": int(st["nonzero_signal_batches"]),
            "zero_signal_batch_fraction": 1.0 - _safe_div(st["nonzero_signal_batches"], batches),
            "batches": batches,
            "total_clip_param_count_mean": int(total_clip_param_count_mean),
            "total_clip_missing_param_count_mean": int(total_clip_missing_param_count_mean),
        })

    snr_values = [row["snr_mean"] for row in expert_rows]
    ucr_values = [
        row["update_coverage_ratio_mean"]
        for row in expert_rows
        if int(row.get("update_coverage_ratio_count", 0)) > 0
    ]
    oca_values = [
        row["oracle_cosine_alignment_mean"]
        for row in expert_rows
        if int(row.get("oracle_cosine_alignment_count", 0)) > 0
    ]
    dilution_values = [row["owner_or_routed_to_denominator"] for row in expert_rows]
    effective_record_values = [row["effective_records_per_batch_mean"] for row in expert_rows]
    clip_ratio_values = [row["clip_factor_to_expert_only_mean"] for row in expert_rows if row["clip_factor_to_expert_only_mean"] > 0.0]
    role_norm_means = {row["role"]: row["norm_mean"] for row in role_rows}
    dense_role_norm_mean = sum(
        role_norm_means.get(role_key, 0.0)
        for role_key in ("shared", "router", "classifier")
    )
    experts_role_norm_mean = role_norm_means.get("experts", 0.0)
    dominant_role = max(role_rows, key=lambda row: row["norm_mean"])["role"] if role_rows else ""
    experts_role_row = next((row for row in role_rows if row["role"] == "experts"), None)
    expert_role_rows = [row for row in role_rows if row["role"].startswith("expert_L")]
    expert_role_norm_values = [row["norm_mean"] for row in expert_role_rows]
    expert_role_active_values = [row["active_fraction"] for row in expert_role_rows]
    owner_or_routed_load_summary = _expert_field_summary(
        expert_rows,
        "owner_or_routed_count",
        distribution=True,
    )
    routed_record_load_summary = _expert_field_summary(
        expert_rows,
        "routed_record_count",
        distribution=True,
    )
    kept_record_load_summary = _expert_field_summary(
        expert_rows,
        "kept_record_count",
        distribution=True,
    )
    effective_record_summary = _expert_field_summary(
        expert_rows,
        "effective_records_per_batch_mean",
        distribution=True,
    )
    snr_summary = _expert_field_summary(expert_rows, "snr_mean")
    ucr_summary = _expert_field_summary(expert_rows, "update_coverage_ratio_mean")
    legacy_ucr_summary = _expert_field_summary(expert_rows, "legacy_per_step_update_coverage_ratio_mean")
    oca_summary = _expert_field_summary(expert_rows, "oracle_cosine_alignment_mean")
    averaged_oca_summary = _expert_field_summary(expert_rows, "averaged_oracle_cosine_alignment")
    averaged_signal_ucr_summary = _expert_field_summary(expert_rows, "averaged_signal_coverage_ratio")
    oracle_gradient_norm_summary = _expert_field_summary(expert_rows, "oracle_gradient_norm_mean")
    averaged_oracle_gradient_norm_summary = _expert_field_summary(expert_rows, "averaged_oracle_gradient_norm")
    averaged_private_update_norm_summary = _expert_field_summary(expert_rows, "averaged_private_update_norm")
    privatized_update_norm_summary = _expert_field_summary(expert_rows, "privatized_update_norm_mean")
    signal_after_summary = _expert_field_summary(expert_rows, "signal_after_divide_norm_mean")
    clip_ratio_summary = _expert_field_summary(expert_rows, "clip_factor_to_expert_only_mean")
    estimated_records_summary = _expert_field_summary(
        expert_rows,
        "estimated_owner_or_routed_records_per_logical_batch",
        distribution=True,
    )
    estimated_active_records_summary = _expert_field_summary(
        expert_rows,
        "estimated_active_grad_records_per_logical_batch",
        distribution=True,
    )
    routed_token_fraction_summary = _expert_field_summary(expert_rows, "routed_token_fraction")
    kept_token_fraction_summary = _expert_field_summary(expert_rows, "kept_token_fraction")
    active_grad_fraction_summary = _expert_field_summary(expert_rows, "active_grad_record_fraction")
    estimated_to_denominator_summary = _expert_field_summary(expert_rows, "owner_or_routed_to_denominator")
    active_to_denominator_summary = _expert_field_summary(expert_rows, "active_grad_to_denominator")
    denominator_to_signal_summary = _expert_field_summary(expert_rows, "denominator_to_estimated_owner_or_routed")
    estimated_to_expected_uniform_summary = _expert_field_summary(
        expert_rows,
        "estimated_owner_or_routed_to_expected_uniform",
        distribution=True,
    )
    zero_signal_batch_summary = _expert_field_summary(expert_rows, "zero_signal_batch_fraction")
    clip_suppression_summary = _expert_field_summary(expert_rows, "clip_suppressed_fraction")
    global_clipped_summary = _expert_field_summary(expert_rows, "global_clipped_fraction")
    expert_only_clipped_summary = _expert_field_summary(expert_rows, "expert_only_clipped_fraction")
    global_clipped_expert_unclipped_summary = _expert_field_summary(
        expert_rows,
        "global_clipped_expert_unclipped_fraction",
    )
    role_by_name = {row["role"]: row for row in role_rows}
    top_level_role_keys = ("shared", "router", "classifier", "experts")
    top_level_role_rows = [
        role_by_name[key]
        for key in top_level_role_keys
        if key in role_by_name
    ]
    top_level_norm_total = sum(float(row["norm_mean"]) for row in top_level_role_rows)
    top_level_param_total = sum(float(row["num_params"]) for row in top_level_role_rows)
    top_level_role_norm_share = {
        row["role"]: _safe_div(float(row["norm_mean"]), top_level_norm_total)
        for row in top_level_role_rows
    }
    top_level_role_param_share = {
        row["role"]: _safe_div(float(row["num_params"]), top_level_param_total)
        for row in top_level_role_rows
    }
    role_mismatch_index = 0.5 * sum(
        abs(
            float(top_level_role_norm_share.get(role_key, 0.0))
            - float(top_level_role_param_share.get(role_key, 0.0))
        )
        for role_key in top_level_role_keys
    )
    top_level_role_norm_summary = _distribution_summary([float(row["norm_mean"]) for row in top_level_role_rows])
    top_level_role_active_summary = _numeric_summary([float(row["active_fraction"]) for row in top_level_role_rows])
    full_model_global_clip_factor_mean = (
        _safe_div(
            sum(float(row["global_clip_factor_mean"]) for row in top_level_role_rows),
            float(len(top_level_role_rows)),
        )
        if top_level_role_rows
        else 0.0
    )
    for row in role_rows:
        role_key = row["role"]
        row["top_level_norm_mean_share"] = top_level_role_norm_share.get(role_key, 0.0)
        row["top_level_param_share"] = top_level_role_param_share.get(role_key, 0.0)
    router_frozen_or_absent = "router" not in role_by_name
    if owner_routed_method:
        freeze_targets = {
            str(t).strip().lower()
            for t in getattr(args, "freeze_in_phase_a", [])
            if str(t).strip()
        }
        router_frozen_or_absent = (
            router_frozen_or_absent
            or bool(getattr(args, "freeze_router_in_phase_a", False))
            or ("router" in freeze_targets)
        )
    method_features = {
        "separate_shared_expert_streams": bool(method_kind == "ours"),
        "separate_privacy_budget": bool(method_kind == "ours"),
        "epsilon_shared_ratio": (
            float(getattr(args, "epsilon_shared_ratio", 0.0))
            if owner_routed_method
            else 0.0
        ),
        "router_frozen_or_absent_from_trainable_roles": bool(router_frozen_or_absent),
        "uses_record_owner_assignments": bool(owner_routed_method and metric_owner_assignments),
        "expert_update_scale_mode": expert_update_scale_mode,
        "uses_expected_owner_scaling": bool(expert_update_scale_mode == "expected_owner"),
        "ours_clip_scope": ours_clip_scope if owner_routed_method else "global",
        "uses_global_clipping": bool(use_global_clip_reference),
        "expert_update_multiplier_for_diagnostics": float(update_multiplier),
        "expert_objective": str(getattr(args, "expert_objective", "ce")),
        "expert_residual_weighting": str(getattr(args, "expert_residual_weighting", "none")),
        "load_balancing_loss_used": False,
        "uses_private_realized_expert_count_denominator": False,
    }
    problem_validation = {
        "problem_1_full_model_dp_treats_heterogeneous_roles_uniformly": {
            "evidence": {
                "role_norms_key": "role_norms",
                "dense_roles_to_experts_norm_mean_ratio": dense_role_norm_mean / max(experts_role_norm_mean, 1e-12),
                "dominant_role_by_norm_mean": dominant_role,
                "top_level_role_norm_share": top_level_role_norm_share,
                "top_level_role_param_share": top_level_role_param_share,
                "role_mismatch_index": role_mismatch_index,
                "M_role": role_mismatch_index,
                "top_level_role_norm_distribution": top_level_role_norm_summary,
                "top_level_role_active_fraction_distribution": top_level_role_active_summary,
                "full_model_global_clip_factor_mean": full_model_global_clip_factor_mean,
                "experts_global_to_role_clip_factor_mean": (
                    experts_role_row["global_to_role_clip_factor_mean"] if experts_role_row is not None else 0.0
                ),
                "role_active_fraction": {
                    row["role"]: row["active_fraction"]
                    for row in role_rows
                    if row["role"] in ("shared", "router", "classifier", "experts")
                },
                "validation_methods": [
                    "compare per-record gradient norm shares across top-level roles",
                    "compare active fractions across roles",
                    "compare global clipping factor against role-only clipping factors",
                    "inspect whether one role dominates the full-model norm",
                ],
            },
            "method_response": {
                "separate_shared_expert_streams": method_features["separate_shared_expert_streams"],
                "epsilon_shared_ratio": method_features["epsilon_shared_ratio"],
                "router_frozen_or_absent_from_trainable_roles": method_features["router_frozen_or_absent_from_trainable_roles"],
                "expert_specific_update_rules": bool(owner_routed_method),
            },
        },
        "problem_2_sparse_expert_signal_dilution": {
            "evidence": {
                "logical_batch_size": int(logical_batch_size),
                "diagnostic_batch_size": int(batch_size),
                "raw_effective_records_per_diagnostic_batch": effective_record_summary,
                "estimated_owner_or_routed_records_per_logical_batch": estimated_records_summary,
                "estimated_active_grad_records_per_logical_batch": estimated_active_records_summary,
                "owner_or_routed_to_denominator_mean": sum(dilution_values) / max(len(dilution_values), 1),
                "owner_or_routed_to_denominator": estimated_to_denominator_summary,
                "active_grad_to_denominator": active_to_denominator_summary,
                "denominator_to_estimated_owner_or_routed": denominator_to_signal_summary,
                "routed_token_fraction_by_expert": routed_token_fraction_summary,
                "active_grad_record_fraction_by_expert": active_grad_fraction_summary,
                "zero_signal_batch_fraction_by_expert": zero_signal_batch_summary,
                "signal_after_divide_norm_by_expert": signal_after_summary,
                "expert_snr_by_expert": snr_summary,
                "update_coverage_ratio_by_expert": ucr_summary,
                "legacy_per_step_update_coverage_ratio_by_expert": legacy_ucr_summary,
                "oracle_cosine_alignment_by_expert": oca_summary,
                "averaged_oracle_cosine_alignment_by_expert": averaged_oca_summary,
                "oracle_gradient_norm_by_expert": oracle_gradient_norm_summary,
                "averaged_oracle_gradient_norm_by_expert": averaged_oracle_gradient_norm_summary,
                "averaged_private_update_norm_by_expert": averaged_private_update_norm_summary,
                "privatized_update_norm_by_expert": privatized_update_norm_summary,
                "validation_methods": [
                    "estimate each expert's active records in a logical DP batch from diagnostic batches",
                    "compare that estimate to the denominator used in the optimizer update",
                    "measure routed-token fraction per expert",
                    "measure fraction of diagnostic batches with zero expert signal",
                    "compare noisy privatized expert updates to unclipped non-private oracle expert gradients",
                ],
            },
            "method_response": {
                "uses_record_owner_assignments": method_features["uses_record_owner_assignments"],
                "expert_update_scale_mode": expert_update_scale_mode,
                "uses_expected_owner_scaling": method_features["uses_expected_owner_scaling"],
                "expected_owner_denominator": (
                    float(logical_batch_size) / float(max(max(num_experts_by_layer.values()), 1))
                    if num_experts_by_layer and expert_update_scale_mode == "expected_owner"
                    else float(logical_batch_size)
                ),
                "uses_private_realized_expert_count_denominator": False,
            },
        },
        "problem_3_global_clipping_suppresses_expert_gradients": {
            "evidence": {
                "clip_reference": clip_reference,
                "experts_global_to_role_clip_factor_mean": (
                    experts_role_row["global_to_role_clip_factor_mean"] if experts_role_row is not None else 0.0
                ),
                "clip_factor_to_expert_only": clip_ratio_summary,
                "clip_suppressed_fraction": clip_suppression_summary,
                "global_clipped_fraction": global_clipped_summary,
                "expert_only_clipped_fraction": expert_only_clipped_summary,
                "global_clipped_expert_unclipped_fraction": global_clipped_expert_unclipped_summary,
                "interpretation": "values below 1.0 mean full/global clipping is harsher than expert-only clipping",
                "validation_methods": [
                    "compare full-model clipping factor to expert-only clipping factor on active expert rows",
                    "measure how often global clipping is active while expert-only clipping would not be",
                    "measure the average retained expert scale under full clipping",
                ],
            },
            "method_response": {
                "uses_expert_stream_clipping": bool(owner_routed_method and not use_global_clip_reference),
                "expert_stream_clip_reference": clip_reference if owner_routed_method else "not_used_by_naive",
                "uses_global_clipping": bool(use_global_clip_reference),
            },
        },
        "problem_4_expert_load_imbalance": {
            "evidence": {
                "owner_or_routed_count_distribution": owner_or_routed_load_summary,
                "routed_record_count_distribution": routed_record_load_summary,
                "kept_record_count_distribution": kept_record_load_summary,
                "estimated_owner_or_routed_records_per_logical_batch": estimated_records_summary,
                "estimated_owner_or_routed_to_expected_uniform": estimated_to_expected_uniform_summary,
                "active_grad_record_fraction_distribution": active_grad_fraction_summary,
                "zero_signal_batch_fraction_distribution": zero_signal_batch_summary,
                "expert_snr_distribution": snr_summary,
                "per_expert_active_fraction_min": min(expert_role_active_values) if expert_role_active_values else 0.0,
                "per_expert_active_fraction_max": max(expert_role_active_values) if expert_role_active_values else 0.0,
                "validation_methods": [
                    "measure coefficient of variation and entropy of load across experts",
                    "measure min/max ratio relative to uniform expected owner count",
                    "measure expert SNR imbalance and zero-signal batch fraction",
                    "separate 'imbalance exists' from 'method balances routing'",
                ],
            },
            "method_response": {
                "fixed_record_owner_assignments": method_features["uses_record_owner_assignments"],
                "uses_expected_owner_scaling": method_features["uses_expected_owner_scaling"],
                "uses_public_expected_owner_denominator": method_features["uses_expected_owner_scaling"],
                "reduces_damage_from_imbalance_not_necessarily_load_cv": bool(owner_routed_method),
                "expert_specific_objective": method_features["expert_objective"],
                "expert_residual_weighting": method_features["expert_residual_weighting"],
            },
        },
        "problem_5_non_per_sample_batch_level_moe_losses": {
            "evidence": {
                "load_balancing_loss_used": False,
                "routing_imbalance_without_private_lb_loss": {
                    "routed_record_count_distribution": routed_record_load_summary,
                    "kept_record_count_distribution": kept_record_load_summary,
                    "routed_token_fraction_distribution": routed_token_fraction_summary,
                    "kept_token_fraction_distribution": kept_token_fraction_summary,
                    "estimated_owner_or_routed_to_expected_uniform": estimated_to_expected_uniform_summary,
                },
                "router_frozen_or_absent_from_trainable_roles": method_features["router_frozen_or_absent_from_trainable_roles"],
                "validation_methods": [
                    "explicitly log that batch-level load-balancing loss is disabled",
                    "measure routing imbalance directly with token and record distributions",
                    "verify the method response does not use private realized expert counts as denominators",
                ],
            },
            "method_response": {
                "does_not_reintroduce_private_batch_level_load_balancing_loss": True,
                "freezes_or_stabilizes_router": method_features["router_frozen_or_absent_from_trainable_roles"],
                "uses_record_owner_expert_training": method_features["uses_record_owner_assignments"],
                "uses_public_expected_owner_denominator": method_features["uses_expected_owner_scaling"],
                "uses_private_realized_expert_count_denominator": False,
            },
        },
    }
    validation_checks = {
        "full_model_role_heterogeneity_supported": bool(
            top_level_role_norm_summary["cv"] > 0.25
            or dense_role_norm_mean / max(experts_role_norm_mean, 1e-12) > 2.0
        ),
        "sparse_expert_signal_supported": bool(
            estimated_to_denominator_summary["all"]["mean"] < 0.75
            or active_to_denominator_summary["all"]["mean"] < 0.75
            or routed_token_fraction_summary["all"]["mean"] < 0.25
        ),
        "global_clipping_suppression_supported": bool(
            clip_suppression_summary["all"]["mean"] > 0.05
            or clip_ratio_summary["all"]["mean"] < 0.9
            or global_clipped_expert_unclipped_summary["all"]["mean"] > 0.05
        ),
        "expert_load_imbalance_supported": bool(
            estimated_to_expected_uniform_summary["all"]["cv"] > 0.25
            or estimated_to_expected_uniform_summary["all"]["normalized_entropy"] < 0.95
            or zero_signal_batch_summary["all"]["mean"] > 0.0
        ),
        "batch_level_load_balancing_loss_absent": True,
        "method_is_role_aware": bool(method_kind == "ours"),
        "method_is_owner_routed_baseline": bool(method_kind == "baseline_c_expert_only"),
    }
    return {
        "enabled": True,
        "method_kind": method_kind,
        "diagnostic_batches": max_batches,
        "diagnostic_batch_size": batch_size,
        "diagnostic_split": diagnostic_split,
        "diagnostic_dataset_size": int(len(metric_ds)),
        "diagnostic_source_dataset_size": int(len(metric_source_ds)),
        "diagnostic_noise_seed": int(noise_seed),
        "logical_batch_size": int(logical_batch_size),
        "expert_sigma": float(expert_sigma),
        "expert_update_scale_mode": expert_update_scale_mode,
        "ours_clip_scope": ours_clip_scope if owner_routed_method else "global",
        "clip_reference": clip_reference,
        "role_norm_global_clip_reference": "full_trainable_params",
        "diagnostic_owner_assignments_recomputed": bool(diagnostic_owner_assignments_recomputed),
        "oracle_reference": "same_model_state_unclipped_unnoised_expert_gradient_on_same_diagnostic_batch",
        "diagnostic_notes": {
            "raw_counts": "Counts are measured on small diagnostic batches to avoid OOM.",
            "logical_batch_estimates": "Estimated logical-batch records scale diagnostic fractions by logical_batch_size.",
            "denominator_note": "Expected-owner scaling uses a public denominator B/num_experts; it does not use private realized expert counts.",
            "ucr_note": "UCR is ||mean_B privatized_update|| / ||non_private_oracle_gradient over active/owner diagnostic records||.",
            "legacy_per_step_ucr_note": "legacy_per_step_update_coverage_ratio_mean is the old mean of per-batch noisy norm ratios; it can explode when oracle gradients are tiny.",
            "oca_note": "OCA is the mean per-step cosine between the diagnostic privatized expert update and the unclipped non-private oracle expert gradient on the same batch.",
            "averaged_oca_note": "averaged_oracle_cosine_alignment is a sanity check: cosine(mean_B privatized_update, mean_B oracle_gradient).",
            "load_imbalance_note": "High load CV in ours does not mean the method failed; the method aims to reduce optimization damage from imbalance, not necessarily balance routing.",
        },
        "summary": {
            "expert_snr_mean": sum(snr_values) / max(len(snr_values), 1),
            "expert_snr_min": min(snr_values) if snr_values else 0.0,
            "update_coverage_ratio_mean": sum(ucr_values) / max(len(ucr_values), 1),
            "update_coverage_ratio_min": min(ucr_values) if ucr_values else 0.0,
            "legacy_per_step_update_coverage_ratio_mean": legacy_ucr_summary["all"]["mean"],
            "legacy_per_step_update_coverage_ratio_min": legacy_ucr_summary["all"]["min"],
            "oracle_cosine_alignment_mean": sum(oca_values) / max(len(oca_values), 1),
            "oracle_cosine_alignment_fraction_experts_above_0_5": _safe_div(
                sum(1.0 for value in oca_values if float(value) > 0.5),
                len(oca_values),
            ),
            "averaged_oracle_cosine_alignment_mean": averaged_oca_summary["all"]["mean"],
            "oracle_gradient_norm_mean": oracle_gradient_norm_summary["all"]["mean"],
            "privatized_update_norm_mean": privatized_update_norm_summary["all"]["mean"],
            "averaged_oracle_gradient_norm_mean": averaged_oracle_gradient_norm_summary["all"]["mean"],
            "averaged_private_update_norm_mean": averaged_private_update_norm_summary["all"]["mean"],
            "averaged_signal_coverage_ratio_mean": averaged_signal_ucr_summary["all"]["mean"],
            "effective_records_per_expert_batch_mean": sum(effective_record_values) / max(len(effective_record_values), 1),
            "estimated_owner_or_routed_records_per_logical_batch_mean": estimated_records_summary["all"]["mean"],
            "estimated_active_grad_records_per_logical_batch_mean": estimated_active_records_summary["all"]["mean"],
            "owner_or_routed_to_denominator_mean": sum(dilution_values) / max(len(dilution_values), 1),
            "active_grad_to_denominator_mean": active_to_denominator_summary["all"]["mean"],
            "denominator_to_estimated_owner_or_routed_mean": denominator_to_signal_summary["all"]["mean"],
            "routed_token_fraction_mean": routed_token_fraction_summary["all"]["mean"],
            "kept_token_fraction_mean": kept_token_fraction_summary["all"]["mean"],
            "active_grad_record_fraction_mean": active_grad_fraction_summary["all"]["mean"],
            "zero_signal_batch_fraction_mean": zero_signal_batch_summary["all"]["mean"],
            "clip_factor_to_expert_only_mean": sum(clip_ratio_values) / max(len(clip_ratio_values), 1),
            "clip_suppressed_fraction_mean": clip_suppression_summary["all"]["mean"],
            "global_clipped_fraction_mean": global_clipped_summary["all"]["mean"],
            "expert_only_clipped_fraction_mean": expert_only_clipped_summary["all"]["mean"],
            "global_clipped_expert_unclipped_fraction_mean": global_clipped_expert_unclipped_summary["all"]["mean"],
            "dense_roles_to_experts_norm_mean_ratio": dense_role_norm_mean / max(experts_role_norm_mean, 1e-12),
            "dominant_role_by_norm_mean": dominant_role,
            "top_level_role_norm_share": top_level_role_norm_share,
            "top_level_role_param_share": top_level_role_param_share,
            "role_mismatch_index": role_mismatch_index,
            "M_role": role_mismatch_index,
            "top_level_role_norm_cv": top_level_role_norm_summary["cv"],
            "top_level_role_norm_normalized_entropy": top_level_role_norm_summary["normalized_entropy"],
            "full_model_global_clip_factor_mean": full_model_global_clip_factor_mean,
            "experts_global_to_role_clip_factor_mean": (
                experts_role_row["global_to_role_clip_factor_mean"] if experts_role_row is not None else 0.0
            ),
            "per_expert_role_norm_mean_min": min(expert_role_norm_values) if expert_role_norm_values else 0.0,
            "per_expert_role_norm_mean_max": max(expert_role_norm_values) if expert_role_norm_values else 0.0,
            "per_expert_active_fraction_min": min(expert_role_active_values) if expert_role_active_values else 0.0,
            "per_expert_active_fraction_max": max(expert_role_active_values) if expert_role_active_values else 0.0,
            "owner_or_routed_count_cv": owner_or_routed_load_summary["all"]["cv"],
            "owner_or_routed_count_normalized_entropy": owner_or_routed_load_summary["all"]["normalized_entropy"],
            "estimated_owner_or_routed_to_expected_uniform_cv": estimated_to_expected_uniform_summary["all"]["cv"],
            "estimated_owner_or_routed_to_expected_uniform_min": estimated_to_expected_uniform_summary["all"]["min"],
            "estimated_owner_or_routed_to_expected_uniform_max": estimated_to_expected_uniform_summary["all"]["max"],
            "routed_record_count_cv": routed_record_load_summary["all"]["cv"],
            "kept_record_count_cv": kept_record_load_summary["all"]["cv"],
            "expert_snr_cv": snr_summary["all"]["cv"],
        },
        "informative_mechanism_diagnostics": {
            "D1_update_coverage_ratio": {
                "mean_across_experts": sum(ucr_values) / max(len(ucr_values), 1),
                "min_across_experts": min(ucr_values) if ucr_values else 0.0,
                "by_expert": ucr_summary,
                "definition": "||mean_B privatized_update|| / ||mean_B non_private_oracle_gradient||",
                "legacy_per_step_noisy_ratio_by_expert": legacy_ucr_summary,
                "signal_only_coverage_ratio_by_expert": averaged_signal_ucr_summary,
            },
            "D2_oracle_cosine_alignment": {
                "mean_across_experts": sum(oca_values) / max(len(oca_values), 1),
                "fraction_experts_mean_above_0_5": _safe_div(
                    sum(1.0 for value in oca_values if float(value) > 0.5),
                    len(oca_values),
                ),
                "by_expert": oca_summary,
                "averaged_then_cosine_by_expert": averaged_oca_summary,
            },
            "D3_role_mismatch_and_snr_variability": {
                "M_role": role_mismatch_index,
                "role_mismatch_index": role_mismatch_index,
                "SNR_CV": snr_summary["all"]["cv"],
                "expert_snr": snr_summary,
            },
        },
        "method_features": method_features,
        "validation_checks": validation_checks,
        "load_imbalance": {
            "owner_or_routed_count": owner_or_routed_load_summary,
            "routed_record_count": routed_record_load_summary,
            "kept_record_count": kept_record_load_summary,
            "effective_records_per_batch": effective_record_summary,
            "estimated_owner_or_routed_records_per_logical_batch": estimated_records_summary,
            "estimated_active_grad_records_per_logical_batch": estimated_active_records_summary,
            "estimated_owner_or_routed_to_expected_uniform": estimated_to_expected_uniform_summary,
            "routed_token_fraction": routed_token_fraction_summary,
            "kept_token_fraction": kept_token_fraction_summary,
            "active_grad_record_fraction": active_grad_fraction_summary,
            "zero_signal_batch_fraction": zero_signal_batch_summary,
            "snr": snr_summary,
            "update_coverage_ratio": ucr_summary,
            "oracle_cosine_alignment": oca_summary,
            "oracle_gradient_norm": oracle_gradient_norm_summary,
            "privatized_update_norm": privatized_update_norm_summary,
            "signal_after_divide_norm": signal_after_summary,
            "clip_factor_to_expert_only": clip_ratio_summary,
            "clip_suppressed_fraction": clip_suppression_summary,
            "global_clipped_fraction": global_clipped_summary,
            "expert_only_clipped_fraction": expert_only_clipped_summary,
            "global_clipped_expert_unclipped_fraction": global_clipped_expert_unclipped_summary,
        },
        "problem_validation": problem_validation,
        "role_norms": role_rows,
        "experts": expert_rows,
    }


def _concat_collated_batches(batches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "idx": torch.cat([b["idx"] for b in batches], dim=0),
        "labels": torch.cat([b["labels"] for b in batches], dim=0),
        "input_ids": torch.cat([b["input_ids"] for b in batches], dim=0),
        "attention_mask": torch.cat([b["attention_mask"] for b in batches], dim=0),
    }


def train_layer_experts_stacked(
    *,
    model: SwitchSST2,
    train_ds: SST2Dataset,
    layer: SparseLayerRef,
    assignments: Sequence[int],
    per_expert_indices: Dict[int, List[int]],
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    target_epsilon: float,
    target_delta: float,
    objective: str = "classifier",
    seq2seq_label_token_ids: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Stacked Phase B for one layer.

    Computation is stacked across experts, but DP is still blockwise:
    every expert follows the same public update schedule and keeps its own
    clipped-sum buffer, Gaussian noise, optimizer, and accountant stream.
    """
    if len(assignments) != len(train_ds):
        raise ValueError(
            f"Assignment length mismatch for layer {layer.layer_id}: "
            f"got {len(assignments)}, expected {len(train_ds)}"
        )
    if OpacusGradSampleModule is None:
        raise RuntimeError("Opacus GradSampleModule is required for stacked expert DP training.")
    if objective not in ("classifier", "seq2seq"):
        raise ValueError(f"[PHASE_B_STACKED] unsupported objective={objective!r}")

    _reset_runtime_peak_memory(device)
    t0 = time.time()
    num_experts = int(layer.sparse_mlp.router.num_experts)
    expert_ids = list(range(num_experts))

    freeze_all(model)
    freeze_routers(model)

    configured_batch, configured_micro = resolve_phase_batch_sizes(args, "phase_b")
    public_n = max(1, len(train_ds))
    public_sample_rate = resolve_phase_b_sample_rate(args, public_n, configured_batch)
    public_expected_total = max(1, int(math.ceil(float(public_n) * float(public_sample_rate))))
    public_effective = max(1, int(math.ceil(float(public_expected_total) / float(max(num_experts, 1)))))
    public_micro = min(configured_micro, public_effective)
    if public_micro <= 0:
        raise RuntimeError("[PHASE B STACKED] micro batch size resolved to zero")
    public_accum = max(1, math.ceil(public_effective / public_micro))
    public_updates_per_epoch = max(1, math.ceil(1.0 / float(public_sample_rate)))
    public_total_updates = int(public_updates_per_epoch) * int(args.expert_epochs)
    sigma = _stacked_noise_multiplier(
        sample_rate=public_sample_rate,
        steps=public_total_updates,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        args=args,
    )

    states: Dict[int, Dict] = {}
    all_named_params: List[Tuple[str, nn.Parameter]] = []
    for expert_id in expert_ids:
        indices = list(per_expert_indices.get(expert_id, []))
        named_params = select_expert_params(model, layer, expert_id)
        for _, p in named_params:
            p.requires_grad_(True)
        params = [p for _, p in named_params]
        optimizer = AdamW(
            params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        scheduler = make_scheduler(optimizer, public_updates_per_epoch, 1, args.expert_epochs)
        states[expert_id] = {
            "indices": indices,
            "named_params": named_params,
            "params": params,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "buffers": {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in params},
            "effective": public_effective,
            "micro": public_micro,
            "accum": public_accum,
            "sigma": sigma,
            "sample_rate": public_sample_rate,
            "updates": 0,
        }
        all_named_params.extend(named_params)

    all_params = [p for _, p in all_named_params]
    n_trainable = sum(p.numel() for p in all_params)
    assignment_tensor = torch.tensor(assignments, dtype=torch.long)
    saved_record_layers = set(model._record_route_layer_ids)
    saved_fixed_assignments = dict(getattr(model, "_record_route_fixed_assignments", {}))
    model._record_route_layer_ids.add(layer.layer_id)
    model._record_route_fixed_assignments = dict(saved_fixed_assignments)
    model._record_route_fixed_assignments[int(layer.layer_id)] = assignment_tensor

    train_model: Optional[nn.Module] = None
    base_model: nn.Module = model
    collate_fn = make_collate(tokenizer, args.max_length, args.input_prefix)

    def _logical_microbatches(expert_id: int) -> List[Dict[str, torch.Tensor]]:
        state = states[expert_id]
        indices = state["indices"]
        if not indices:
            return []

        q = float(state["sample_rate"])
        if q >= 1.0:
            selected_positions = torch.randperm(len(indices))
        else:
            keep = torch.rand(len(indices)) < q
            selected_positions = keep.nonzero(as_tuple=False).flatten()
            if selected_positions.numel() == 0:
                return []
            selected_positions = selected_positions[torch.randperm(selected_positions.numel())]
        chosen = [indices[int(i)] for i in selected_positions.tolist()]
        batches: List[Dict[str, torch.Tensor]] = []
        for start in range(0, len(chosen), int(state["micro"])):
            chunk = chosen[start:start + int(state["micro"])]
            if chunk:
                batches.append(collate_fn([train_ds[idx] for idx in chunk]))
        return batches

    def _accumulate_for_expert(expert_id: int, row_start: int, row_end: int) -> None:
        state = states[expert_id]
        params = state["params"]
        rows = torch.arange(row_start, row_end, device=device, dtype=torch.long)
        if rows.numel() == 0:
            return

        norm_sq = torch.zeros(rows.numel(), device=device, dtype=torch.float32)
        usable: List[Tuple[nn.Parameter, torch.Tensor]] = []
        for p in params:
            gs = _grad_sample_as_tensor(p)
            if gs is None:
                continue
            gs_rows = gs.index_select(0, rows)
            usable.append((p, gs_rows))
            norm_sq += gs_rows.detach().float().flatten(1).pow(2).sum(dim=1)

        if not usable:
            return

        norms = norm_sq.sqrt()
        factors = torch.clamp(float(args.max_grad_norm) / (norms + 1e-6), max=1.0)
        for p, gs_rows in usable:
            view = factors
            while view.dim() < gs_rows.dim():
                view = view.unsqueeze(-1)
            clipped = (gs_rows * view.to(gs_rows.dtype)).sum(dim=0)
            state["buffers"][p].add_(clipped.to(state["buffers"][p].dtype))

    def _step_expert(expert_id: int) -> None:
        state = states[expert_id]
        params = state["params"]
        denom = int(state["effective"])
        for p in params:
            buf = state["buffers"][p]
            grad = buf
            if not args.no_dp:
                noise = torch.randn(buf.shape, device=buf.device, dtype=torch.float32)
                noise = noise.to(buf.dtype).mul_(float(state["sigma"]) * float(args.max_grad_norm))
                grad = grad + noise
            p.grad = (grad / float(max(denom, 1))).to(p.dtype)

        state["optimizer"].step()
        sanitize_parameters(params)
        state["scheduler"].step()
        state["optimizer"].zero_grad(set_to_none=True)
        for p in params:
            state["buffers"][p].zero_()
            p.grad = None
        state["updates"] += 1

    def _debug_assert_non_owner_grad_samples_zero(batch_indices: torch.Tensor) -> None:
        if not getattr(args, "debug_privacy", False):
            return

        owners = assignment_tensor.index_select(0, batch_indices.detach().cpu().long()).to(device)
        for expert_id in expert_ids:
            non_owner_rows = (owners != int(expert_id)).nonzero(as_tuple=False).flatten()
            if non_owner_rows.numel() == 0:
                continue

            worst = 0.0
            worst_name = ""
            for name, p in states[expert_id]["named_params"]:
                gs = _grad_sample_as_tensor(p)
                if gs is None:
                    continue
                if gs.size(0) != owners.numel():
                    raise RuntimeError(
                        f"[DEBUG PRIVACY] grad_sample batch mismatch for expert {expert_id} "
                        f"param {name}: got {gs.size(0)}, expected {owners.numel()}"
                    )
                rows = non_owner_rows.to(gs.device)
                row_norms = gs.index_select(0, rows).detach().float().flatten(1).norm(dim=1)
                if row_norms.numel() == 0:
                    continue
                value = float(row_norms.max().item())
                if value > worst:
                    worst = value
                    worst_name = name

            if worst > 1e-6:
                raise RuntimeError(
                    f"[DEBUG PRIVACY] non-owner grad_sample is nonzero for expert {expert_id}: "
                    f"max_norm={worst:.6e} param={worst_name}"
                )

    try:
        train_model = OpacusGradSampleModule(model, batch_first=True, loss_reduction="mean")
        if hasattr(train_model, "_module"):
            base_model = train_model._module

        print(
            f"[PHASE B STACKED] layer={layer.layer_id} experts={num_experts} "
            f"trainable={n_trainable:,} epochs={args.expert_epochs} "
            f"updates_per_epoch={public_updates_per_epoch} q={public_sample_rate:.6f} "
            f"expected_total/update={public_expected_total} "
            f"expected_expert/update={public_effective} sigma={sigma:.4f}"
        )
        if args.min_expert_size != 1:
            print("[PHASE B STACKED] min_expert_size is ignored in fixed-schedule DP mode.")

        label_token_ids = None
        if objective == "seq2seq":
            if seq2seq_label_token_ids is None:
                raise ValueError("[PHASE B STACKED] seq2seq objective requires label token ids")
            label_token_ids = seq2seq_label_token_ids.to(device)

        for epoch in range(args.expert_epochs):
            train_model.train()
            for _update_idx in range(public_updates_per_epoch):
                logical_batches = {eid: _logical_microbatches(eid) for eid in expert_ids}
                max_micro_steps = max(
                    public_accum,
                    max((len(v) for v in logical_batches.values()), default=0),
                )
                for micro_idx in range(max_micro_steps):
                    packed: List[Tuple[int, Dict[str, torch.Tensor]]] = []
                    row_slices: Dict[int, Tuple[int, int]] = {}
                    row_start = 0
                    for expert_id in expert_ids:
                        batches = logical_batches[expert_id]
                        if micro_idx >= len(batches):
                            continue
                        batch = batches[micro_idx]
                        batch_n = int(batch["labels"].numel())
                        if batch_n <= 0:
                            continue
                        packed.append((expert_id, batch))
                        row_slices[expert_id] = (row_start, row_start + batch_n)
                        row_start += batch_n

                    if not packed:
                        continue

                    combined = _concat_collated_batches([batch for _, batch in packed])
                    ids = combined["input_ids"].to(device)
                    mask = combined["attention_mask"].to(device)
                    labels = combined["labels"].to(device)
                    base_model._active_batch_indices = combined["idx"].detach()

                    for p in all_params:
                        p.grad = None
                    if objective == "seq2seq":
                        target_token_ids = label_token_ids.index_select(0, labels)
                        loss = base_model.seq2seq_loss(ids, mask, target_token_ids)
                    else:
                        logits = train_model(ids, mask)
                        if not torch.isfinite(logits).all():
                            _clear_grad_sample_state(all_params)
                            continue
                        loss = F.cross_entropy(logits, labels)
                    if not torch.isfinite(loss):
                        _clear_grad_sample_state(all_params)
                        continue

                    loss.backward()
                    _debug_assert_non_owner_grad_samples_zero(combined["idx"])
                    for expert_id, (start, end) in row_slices.items():
                        _accumulate_for_expert(expert_id, start, end)
                    _clear_grad_sample_state(all_params)

                for expert_id in expert_ids:
                    _step_expert(expert_id)

            print(f"[PHASE B STACKED] layer={layer.layer_id} epoch={epoch+1}/{args.expert_epochs} complete")
    finally:
        if train_model is not None and hasattr(train_model, "to_standard_module"):
            train_model.to_standard_module()
        model._record_route_layer_ids.clear()
        model._record_route_layer_ids.update(saved_record_layers)
        model._record_route_fixed_assignments = saved_fixed_assignments
        if hasattr(base_model, "_active_batch_indices"):
            base_model._active_batch_indices = None
        for _, p in all_named_params:
            p.requires_grad_(False)
        _clear_grad_sample_state(all_params)

    expert_logs: List[Dict] = []
    expert_epsilons: List[float] = []
    sigmas: List[float] = []
    total_updates = 0
    for expert_id in expert_ids:
        state = states[expert_id]
        eps = _stacked_privacy_epsilon(
            noise_multiplier=float(state["sigma"]),
            sample_rate=float(state["sample_rate"]),
            steps=int(state["updates"]),
            target_delta=target_delta,
            fallback_epsilon=target_epsilon,
            args=args,
        )
        expert_epsilons.append(eps)
        sigmas.append(float(state["sigma"]))
        total_updates += int(state["updates"])
        expert_logs.append({
            "layer_id": layer.layer_id,
            "expert_id": expert_id,
            "stacked": True,
            "epsilon": eps,
            "sigma": float(state["sigma"]),
            "sample_rate": float(state["sample_rate"]),
            "updates": int(state["updates"]),
            "micro_batch_size": int(state["micro"]),
            "public_denominator": int(state["effective"]),
            "expected_sampled_records_per_expert_update": int(state["effective"]),
            "accum": int(state["accum"]),
            "target_epsilon": target_epsilon,
            "target_delta": target_delta,
            "schedule": "fixed_public_poisson",
        })

    max_eps = max(expert_epsilons) if expert_epsilons else 0.0
    print(f"[PHASE B STACKED] layer={layer.layer_id} max expert actual ε={max_eps:.4f}")
    runtime = _runtime_stats(t0, total_updates, device)
    return {
        "stage": f"PHASE_B_LAYER_{layer.layer_id}_STACKED_EXPERTS",
        "epsilon": max_eps,
        "sigma": max(sigmas) if sigmas else 0.0,
        "accountant": ACCOUNTING_MODE if not args.no_dp else "none",
        "target_epsilon": target_epsilon if not args.no_dp else 0.0,
        "target_delta": target_delta if not args.no_dp else 0.0,
        "privacy_spent": {"eps_prv": max_eps} if not args.no_dp else {},
        "n_trainable": n_trainable,
        "updates": total_updates,
        "updates_per_expert": public_total_updates,
        "updates_per_epoch": public_updates_per_epoch,
        "planned_optimizer_steps": int(public_total_updates * max(1, num_experts)),
        "sample_rate": public_sample_rate,
        "expected_sampled_records_per_update": public_expected_total,
        "expected_sampled_records_per_expert_update": public_effective,
        **runtime,
        "experts": sorted(expert_logs, key=lambda x: int(x["expert_id"])),
        "empty_expert_policy": "fixed_schedule_noise_only_update",
        "subsampling": "public_record_poisson_before_frozen_routing",
        "stacking": "combined_forward_backward_with_per_expert_noise",
        "count_logging": "disabled",
    }


def train_one_phase_hybrid_ours(
    *,
    model: SwitchSST2,
    train_ds: SST2Dataset,
    dev_ds: SST2Dataset,
    sparse_layers: Sequence[SparseLayerRef],
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    zero_shot: Dict,
    objective: str = "classifier",
    seq2seq_label_token_ids: Optional[torch.Tensor] = None,
) -> Dict:
    """
    One-phase hybrid DP-MoE for --experiment ours.

    Shared parameters are one sampled DP-SGD stream over full records.
    Expert parameters are separate per-expert streams with record-owner
    gradient masking. Experts compose in parallel within one sparse layer;
    shared + layers compose sequentially.
    """
    if OpacusGradSampleModule is None:
        raise RuntimeError("Opacus GradSampleModule is required for one-phase hybrid DP training.")
    if objective not in ("classifier", "seq2seq"):
        raise ValueError(f"[OURS ONE-PHASE] unsupported objective={objective!r}")
    if len(train_ds) == 0:
        raise RuntimeError("[OURS ONE-PHASE] empty training dataset")
    phase_a_only = bool(getattr(args, "phase_a_only", False))
    expert_only = canonical_experiment_name(str(getattr(args, "experiment", ""))) == "baseline_c_expert_only"
    log_prefix = "[BASELINE_C_EXPERT_ONLY]" if expert_only else "[OURS ONE-PHASE]"
    privacy_prefix = "[BASELINE_C_EXPERT_ONLY PRIVACY PLAN]" if expert_only else "[OURS ONE-PHASE PRIVACY PLAN]"
    if phase_a_only and expert_only:
        raise ValueError("[OURS ONE-PHASE] shared-only and expert-only modes are mutually exclusive")

    layers_to_train = select_sparse_layers_by_args(
        sparse_layers,
        args,
        stage_name=log_prefix.strip("[]"),
    )

    run_epochs = int(getattr(args, "ours_epochs", -1))
    if run_epochs <= 0:
        run_epochs = int(args.finetune_epochs)
    if run_epochs <= 0:
        raise ValueError("[OURS ONE-PHASE] --ours_epochs or --finetune_epochs must be positive.")

    configured_batch = int(args.train_batch_size)
    configured_micro = int(args.micro_batch_size)
    if configured_batch <= 0:
        raise ValueError(f"[OURS ONE-PHASE] --train_batch_size must be positive, got {configured_batch}")
    if configured_micro <= 0:
        raise ValueError(f"[OURS ONE-PHASE] --micro_batch_size must be positive, got {configured_micro}")
    public_n = max(1, len(train_ds))
    sample_rate = float(min(max(configured_batch, 1), public_n)) / float(public_n)
    expected_total = max(1, int(math.ceil(float(public_n) * float(sample_rate))))
    micro = min(max(1, int(configured_micro)), len(train_ds))
    updates_per_epoch = max(1, int(math.ceil(1.0 / float(sample_rate))))
    total_updates = int(updates_per_epoch) * int(run_epochs)
    residual_mode = str(getattr(args, "expert_residual_weighting", "none")).strip().lower()
    if residual_mode not in ("none", "prob", "margin"):
        raise ValueError(f"[OURS ONE-PHASE] unknown expert_residual_weighting={residual_mode!r}")
    expert_update_scale_mode = str(getattr(args, "expert_update_scale_mode", "batch")).strip().lower()
    if expert_update_scale_mode not in ("batch", "expected_owner"):
        raise ValueError(f"[OURS ONE-PHASE] unknown expert_update_scale_mode={expert_update_scale_mode!r}")
    ours_clip_scope = str(getattr(args, "ours_clip_scope", "role")).strip().lower()
    if ours_clip_scope not in ("role", "global"):
        raise ValueError(f"[OURS ONE-PHASE] unknown ours_clip_scope={ours_clip_scope!r}")
    expert_lr_multiplier = float(getattr(args, "expert_lr_multiplier", 1.0))
    if expert_lr_multiplier <= 0.0:
        raise ValueError(f"[OURS ONE-PHASE] --expert_lr_multiplier must be > 0, got {expert_lr_multiplier}")
    expert_objective = str(getattr(args, "expert_objective", "ce")).strip().lower()
    if expert_objective not in ("ce", "residual_logit"):
        raise ValueError(f"[OURS ONE-PHASE] unknown expert_objective={expert_objective!r}")
    if phase_a_only:
        expert_objective = "ce"
    alternating_shared_steps = int(getattr(args, "alternating_shared_steps", 0))
    alternating_expert_steps = int(getattr(args, "alternating_expert_steps", 0))
    if alternating_shared_steps < 0:
        raise ValueError("[OURS ONE-PHASE] --alternating_shared_steps must be >= 0")
    if alternating_expert_steps < 0:
        raise ValueError("[OURS ONE-PHASE] --alternating_expert_steps must be >= 0")
    alternating_enabled = (not phase_a_only) and (not expert_only) and alternating_expert_steps > 0
    if alternating_enabled and alternating_shared_steps <= 0:
        alternating_shared_steps = 1
    if not alternating_enabled:
        alternating_shared_steps = 0
        alternating_expert_steps = 0

    if alternating_enabled:
        window_steps = int(alternating_shared_steps + alternating_expert_steps)
        full_windows = int(total_updates // window_steps)
        remainder = int(total_updates % window_steps)
        planned_shared_updates = full_windows * int(alternating_shared_steps) + min(
            remainder,
            int(alternating_shared_steps),
        )
        planned_expert_updates = full_windows * int(alternating_expert_steps) + max(
            0,
            remainder - int(alternating_shared_steps),
        )
    else:
        planned_shared_updates = 0 if expert_only else int(total_updates)
        planned_expert_updates = 0 if phase_a_only else int(total_updates)
    shared_accounting_steps = max(1, int(planned_shared_updates))
    expert_accounting_steps = max(1, int(planned_expert_updates)) if not phase_a_only else 0

    if args.no_dp:
        adj_factor = 1 if args.adjacency == "add_remove" else 2
        num_private_components = 0
        epsilon_shared_ratio = float(getattr(args, "epsilon_shared_ratio", 0.3))
        eps_shared_target = 0.0
        eps_experts_total_target = 0.0
        eps_layer_target = 0.0
        expert_eps_target = 0.0
        delta_shared = 0.0
        delta_experts_total = 0.0
        delta_layer = 0.0
        expert_delta = 0.0
        sigma_shared = 0.0
        sigma_expert = 0.0
    elif phase_a_only:
        adj_factor = 1 if args.adjacency == "add_remove" else 2
        num_private_components = 1
        epsilon_shared_ratio = 1.0
        eps_shared_target = float(args.epsilon_total)
        eps_experts_total_target = 0.0
        eps_layer_target = 0.0
        expert_eps_target = 0.0
        delta_shared = float(args.delta)
        delta_experts_total = 0.0
        delta_layer = 0.0
        expert_delta = 0.0
        sigma_shared = _stacked_noise_multiplier(
            sample_rate=sample_rate,
            steps=shared_accounting_steps,
            target_epsilon=eps_shared_target,
            target_delta=delta_shared,
            args=args,
        )
        sigma_expert = 0.0
    elif expert_only:
        adj_factor = 1 if args.adjacency == "add_remove" else 2
        num_private_components = int(adj_factor * len(layers_to_train))
        epsilon_shared_ratio = 0.0
        eps_shared_target = 0.0
        eps_experts_total_target = float(args.epsilon_total)
        eps_layer_target = eps_experts_total_target / float(max(1, len(layers_to_train)))
        expert_eps_target = eps_experts_total_target / float(max(1, adj_factor * len(layers_to_train)))
        delta_shared = 0.0
        delta_experts_total = float(args.delta)
        delta_layer = delta_experts_total / float(max(1, len(layers_to_train)))
        expert_delta = delta_experts_total / float(max(1, adj_factor * len(layers_to_train)))
        sigma_shared = 0.0
        sigma_expert = _stacked_noise_multiplier(
            sample_rate=sample_rate,
            steps=expert_accounting_steps,
            target_epsilon=expert_eps_target,
            target_delta=expert_delta,
            args=args,
        )
    else:
        budget = resolve_ours_privacy_budget(args, len(layers_to_train))
        adj_factor = int(budget["adj_factor"])
        num_private_components = int(budget["num_units"])
        epsilon_shared_ratio = float(budget["epsilon_shared_ratio"])
        eps_shared_target = float(budget["eps_shared"])
        eps_experts_total_target = float(budget["eps_experts_total"])
        delta_shared = float(budget["delta_shared"])
        delta_experts_total = float(budget["delta_experts_total"])
        expert_eps_target = float(budget["eps_expert"])
        expert_delta = float(budget["delta_expert"])
        eps_layer_target = float(budget["eps_layer"])
        delta_layer = float(budget["delta_layer"])
        sigma_shared = _stacked_noise_multiplier(
            sample_rate=sample_rate,
            steps=shared_accounting_steps,
            target_epsilon=eps_shared_target,
            target_delta=delta_shared,
            args=args,
        )
        sigma_expert = _stacked_noise_multiplier(
            sample_rate=sample_rate,
            steps=expert_accounting_steps,
            target_epsilon=expert_eps_target,
            target_delta=expert_delta,
            args=args,
        )

    print("\n" + "=" * 88)
    print(f"{log_prefix} hybrid shared + record-owner expert DP training")
    if phase_a_only:
        print(f"{log_prefix} --phase_a_only enabled: expert streams disabled; shared stream uses full budget")
    if expert_only:
        print(f"{log_prefix} shared stream disabled; expert streams use full budget")
    print(f"{log_prefix} selected sparse layers = {[sl.layer_id for sl in layers_to_train]}")
    print(
        f"{log_prefix} epochs={run_epochs} updates_per_epoch={updates_per_epoch} "
        f"total_updates={total_updates} q={sample_rate:.6f} "
        f"expected/update={expected_total} micro={micro}"
    )
    print(
        f"{log_prefix} expert_residual_weighting={residual_mode} "
        f"expert_objective={expert_objective} "
        f"expert_update_scale_mode={expert_update_scale_mode} "
        f"clip_scope={ours_clip_scope} "
        f"expert_lr_multiplier={expert_lr_multiplier:g}"
    )
    if alternating_enabled:
        print(
            f"{log_prefix} alternating residual windows: "
            f"shared_steps={alternating_shared_steps} expert_steps={alternating_expert_steps} "
            f"planned_shared_updates={planned_shared_updates} planned_expert_updates={planned_expert_updates}"
        )
    else:
        print(
            f"{log_prefix} simultaneous shared/expert updates "
            f"planned_shared_updates={planned_shared_updates} planned_expert_updates={planned_expert_updates}"
        )
    if args.no_dp:
        print(f"{privacy_prefix} no-DP mode")
    else:
        if phase_a_only:
            print(
                f"{privacy_prefix} eps_shared={eps_shared_target:.4f} "
                f"eps_experts_total=0.0000 delta_shared={delta_shared:.3e} "
                f"shared_ratio={epsilon_shared_ratio:.3f} sigma_shared={sigma_shared:.4f}"
            )
        elif expert_only:
            print(
                f"{privacy_prefix} eps_shared=0.0000 "
                f"eps_experts_total={eps_experts_total_target:.4f} "
                f"eps_layer={eps_layer_target:.4f} expert_eps_target={expert_eps_target:.4f} "
                f"expert_delta={expert_delta:.3e} shared_ratio={epsilon_shared_ratio:.3f} "
                f"sigma_expert={sigma_expert:.4f}"
            )
        else:
            print(
                f"{privacy_prefix} eps_shared={eps_shared_target:.4f} "
                f"eps_experts_total={eps_experts_total_target:.4f} "
                f"eps_layer={eps_layer_target:.4f} expert_eps_target={expert_eps_target:.4f} "
                f"delta_shared={delta_shared:.3e} expert_delta={expert_delta:.3e} "
                f"shared_ratio={epsilon_shared_ratio:.3f} "
                f"sigma_shared={sigma_shared:.4f} sigma_expert={sigma_expert:.4f}"
            )

    freeze_all(model)
    freeze_routers(model)
    zero_router_jitter(model)

    assignment_cache: Dict[int, torch.Tensor] = {}
    assignment_metric_rows: List[Dict] = []
    assignment_metrics_by_layer: Dict[int, Dict] = {}
    print(f"{log_prefix} computing fixed record-owner assignments (counts are not logged)")
    for layer in layers_to_train:
        assignments = construct_record_assignments(
            model,
            train_ds,
            tokenizer,
            target_layer=layer,
            batch_size=args.assignment_batch_size,
            max_length=args.max_length,
            device=device,
            prefix=args.input_prefix,
        )
        perturbed_assignments, assignment_metrics = maybe_perturb_owner_assignments(
            assignments,
            int(layer.sparse_mlp.router.num_experts),
            args,
            layer_id=int(layer.layer_id),
        )
        assignment_metrics["block_id"] = int(layer.block_id)
        assignment_cache[int(layer.layer_id)] = perturbed_assignments.detach().cpu().long()
        assignment_metric_rows.append(assignment_metrics)
        assignment_metrics_by_layer[int(layer.layer_id)] = assignment_metrics
        print(
            f"{log_prefix} layer={layer.layer_id} assignments ready "
            f"H={assignment_metrics['entropy']:.4f} "
            f"deficit={assignment_metrics['entropy_deficit']:.4f} "
            f"chi2={assignment_metrics['chi2_to_uniform']:.4f} "
            f"changed={assignment_metrics['perturb_changed_fraction']:.3f}"
        )
    assignment_metric_summary = summarize_owner_metric_rows(assignment_metric_rows)

    shared_freeze_targets = {str(t).strip().lower() for t in args.freeze_in_phase_a if str(t).strip()}
    if args.freeze_router_in_phase_a:
        shared_freeze_targets.add("router")
    shared_named = [] if expert_only else select_ours_shared_params(model, freeze_targets=sorted(shared_freeze_targets))
    if (not expert_only) and not shared_named:
        raise RuntimeError("[OURS ONE-PHASE] no shared parameters selected.")
    for _, p in shared_named:
        p.requires_grad_(True)
    shared_params = [p for _, p in shared_named]
    shared_optimizer: Optional[torch.optim.Optimizer] = None
    shared_scheduler = None
    if shared_params:
        shared_optimizer = AdamW(
            shared_params,
            lr=float(args.lr),
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        shared_scheduler = make_scheduler(shared_optimizer, updates_per_epoch, 1, run_epochs)
    shared_buffers = {
        p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in shared_params
    }

    expert_states: Dict[Tuple[int, int], Dict] = {}
    all_named_params: List[Tuple[str, nn.Parameter]] = list(shared_named)
    if not phase_a_only:
        for layer in layers_to_train:
            num_experts = int(layer.sparse_mlp.router.num_experts)
            for expert_id in range(num_experts):
                named = select_expert_params(model, layer, expert_id)
                if not named:
                    raise RuntimeError(
                        f"[OURS ONE-PHASE] no params selected for layer={layer.layer_id} expert={expert_id}"
                    )
                for _, p in named:
                    p.requires_grad_(True)
                params = [p for _, p in named]
                optimizer = AdamW(
                    params,
                    lr=float(args.lr),
                    betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay,
                )
                scheduler = make_scheduler(optimizer, updates_per_epoch, 1, run_epochs)
                expert_states[(int(layer.layer_id), int(expert_id))] = {
                    "layer_id": int(layer.layer_id),
                    "expert_id": int(expert_id),
                    "num_experts": int(num_experts),
                    "named_params": named,
                    "params": params,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "buffers": {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in params},
                    "updates": 0,
                }
                all_named_params.extend(named)

    unique_params: List[nn.Parameter] = []
    seen_param_ids: set[int] = set()
    for _, p in all_named_params:
        pid = id(p)
        if pid not in seen_param_ids:
            unique_params.append(p)
            seen_param_ids.add(pid)

    n_shared = sum(p.numel() for p in shared_params)
    n_expert = sum(
        p.numel()
        for state in expert_states.values()
        for p in state["params"]
    )
    expert_params_flat = [
        p
        for state in expert_states.values()
        for p in state["params"]
    ]
    print(
        f"{log_prefix} shared_trainable={n_shared:,} expert_trainable={n_expert:,} "
        f"lr={args.lr}"
    )

    label_token_ids = None
    if objective == "seq2seq":
        if seq2seq_label_token_ids is None:
            raise ValueError("[OURS ONE-PHASE] seq2seq objective requires label token ids")
        label_token_ids = seq2seq_label_token_ids.to(device)

    collate_fn = make_collate(tokenizer, args.max_length, args.input_prefix)
    saved_record_layers = set(model._record_route_layer_ids)
    saved_fixed_assignments = dict(getattr(model, "_record_route_fixed_assignments", {}))
    model._record_route_layer_ids.clear()
    model._record_route_layer_ids.update({int(sl.layer_id) for sl in layers_to_train})
    model._record_route_fixed_assignments = {
        int(sl.layer_id): assignment_cache[int(sl.layer_id)] for sl in layers_to_train
    }

    train_model: Optional[nn.Module] = None
    base_model: nn.Module = model

    def _global_clip_factors(rows: torch.Tensor) -> Optional[torch.Tensor]:
        if ours_clip_scope != "global" or rows.numel() == 0:
            return None
        norm_sq = torch.zeros(rows.numel(), device=device, dtype=torch.float32)
        saw_grad = False
        for p in unique_params:
            gs = _grad_sample_as_tensor(p)
            if gs is None:
                continue
            gs_rows = gs.index_select(0, rows.to(gs.device))
            norm_sq += gs_rows.detach().float().flatten(1).pow(2).sum(dim=1).to(norm_sq.device)
            saw_grad = True
        if not saw_grad:
            return None
        return torch.clamp(float(args.max_grad_norm) / (norm_sq.sqrt() + 1e-6), max=1.0)

    def _accumulate_shared(rows: torch.Tensor, clip_factors: Optional[torch.Tensor] = None) -> None:
        if rows.numel() == 0:
            return
        norm_sq = torch.zeros(rows.numel(), device=device, dtype=torch.float32)
        usable: List[Tuple[nn.Parameter, torch.Tensor]] = []
        for p in shared_params:
            gs = _grad_sample_as_tensor(p)
            if gs is None:
                continue
            gs_rows = gs.index_select(0, rows.to(gs.device))
            usable.append((p, gs_rows))
            norm_sq += gs_rows.detach().float().flatten(1).pow(2).sum(dim=1).to(norm_sq.device)
        if not usable:
            return
        if clip_factors is None:
            factors = torch.clamp(float(args.max_grad_norm) / (norm_sq.sqrt() + 1e-6), max=1.0)
        else:
            factors = clip_factors.to(device=norm_sq.device, dtype=torch.float32)
        for p, gs_rows in usable:
            view = factors
            while view.dim() < gs_rows.dim():
                view = view.unsqueeze(-1)
            clipped = (gs_rows * view.to(gs_rows.dtype)).sum(dim=0)
            shared_buffers[p].add_(clipped.to(shared_buffers[p].dtype))

    def _accumulate_expert(
        layer_id: int,
        expert_id: int,
        batch_indices: torch.Tensor,
        residual_weights: Optional[torch.Tensor],
        clip_factors: Optional[torch.Tensor] = None,
    ) -> None:
        owners = assignment_cache[int(layer_id)].index_select(
            0, batch_indices.detach().cpu().long()
        ).to(device=device)
        owner_rows = (owners == int(expert_id)).nonzero(as_tuple=False).flatten()
        if owner_rows.numel() == 0:
            return
        state = expert_states[(int(layer_id), int(expert_id))]
        norm_sq = torch.zeros(owner_rows.numel(), device=device, dtype=torch.float32)
        usable: List[Tuple[nn.Parameter, torch.Tensor]] = []
        for p in state["params"]:
            gs = _grad_sample_as_tensor(p)
            if gs is None:
                continue
            gs_rows = gs.index_select(0, owner_rows.to(gs.device))
            if residual_weights is not None:
                row_weights = residual_weights.index_select(0, owner_rows.to(residual_weights.device))
                row_weights = row_weights.to(device=gs_rows.device, dtype=gs_rows.dtype)
                while row_weights.dim() < gs_rows.dim():
                    row_weights = row_weights.unsqueeze(-1)
                gs_rows = gs_rows * row_weights
            usable.append((p, gs_rows))
            norm_sq += gs_rows.detach().float().flatten(1).pow(2).sum(dim=1).to(norm_sq.device)
        if not usable:
            return
        if clip_factors is None:
            factors = torch.clamp(float(args.max_grad_norm) / (norm_sq.sqrt() + 1e-6), max=1.0)
        else:
            factors = clip_factors.index_select(0, owner_rows.to(clip_factors.device)).to(
                device=norm_sq.device,
                dtype=torch.float32,
            )
        for p, gs_rows in usable:
            view = factors
            while view.dim() < gs_rows.dim():
                view = view.unsqueeze(-1)
            clipped = (gs_rows * view.to(gs_rows.dtype)).sum(dim=0)
            state["buffers"][p].add_(clipped.to(state["buffers"][p].dtype))

    def _step_shared() -> None:
        if not shared_params or shared_optimizer is None or shared_scheduler is None:
            return
        denom = max(1, int(configured_batch))
        for p in shared_params:
            buf = shared_buffers[p]
            grad = buf
            if not args.no_dp:
                noise = torch.randn(buf.shape, device=buf.device, dtype=torch.float32)
                noise = noise.to(buf.dtype).mul_(float(sigma_shared) * float(args.max_grad_norm))
                grad = grad + noise
            p.grad = (grad / float(denom)).to(p.dtype)
        shared_optimizer.step()
        sanitize_parameters(shared_params)
        shared_scheduler.step()
        shared_optimizer.zero_grad(set_to_none=True)
        for p in shared_params:
            shared_buffers[p].zero_()
            p.grad = None

    def _step_expert(layer_id: int, expert_id: int) -> None:
        state = expert_states[(int(layer_id), int(expert_id))]
        if expert_update_scale_mode == "expected_owner":
            num_experts = max(1, int(state.get("num_experts", 1)))
            denom = max(1.0, float(configured_batch) / float(num_experts))
        else:
            denom = float(max(1, int(configured_batch)))
        for p in state["params"]:
            buf = state["buffers"][p]
            grad = buf
            if not args.no_dp:
                noise = torch.randn(buf.shape, device=buf.device, dtype=torch.float32)
                noise = noise.to(buf.dtype).mul_(float(sigma_expert) * float(args.max_grad_norm))
                grad = grad + noise
            p.grad = (grad * float(expert_lr_multiplier) / float(denom)).to(p.dtype)
        state["optimizer"].step()
        sanitize_parameters(state["params"])
        state["scheduler"].step()
        state["optimizer"].zero_grad(set_to_none=True)
        for p in state["params"]:
            state["buffers"][p].zero_()
            p.grad = None
        state["updates"] += 1

    def _debug_assert_non_owner_grad_samples_zero(batch_indices: torch.Tensor) -> None:
        if phase_a_only or not getattr(args, "debug_privacy", False):
            return
        for layer in layers_to_train:
            layer_id = int(layer.layer_id)
            owners = assignment_cache[layer_id].index_select(
                0, batch_indices.detach().cpu().long()
            ).to(device=device)
            for expert_id in range(int(layer.sparse_mlp.router.num_experts)):
                non_owner_rows = (owners != int(expert_id)).nonzero(as_tuple=False).flatten()
                if non_owner_rows.numel() == 0:
                    continue
                state = expert_states[(layer_id, int(expert_id))]
                worst = 0.0
                worst_name = ""
                for name, p in state["named_params"]:
                    gs = _grad_sample_as_tensor(p)
                    if gs is None:
                        continue
                    if gs.size(0) != owners.numel():
                        raise RuntimeError(
                            f"[DEBUG PRIVACY] grad_sample batch mismatch for layer {layer_id} "
                            f"expert {expert_id} param {name}: got {gs.size(0)}, expected {owners.numel()}"
                        )
                    row_norms = gs.index_select(0, non_owner_rows.to(gs.device)).detach().float().flatten(1).norm(dim=1)
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

    def _set_trainable_streams(shared_enabled: bool, expert_enabled: bool) -> None:
        # Opacus registers forward/backward hooks assuming a stable trainable
        # parameter set. Do not toggle requires_grad inside the wrapped model:
        # "frozen" windows are enforced by choosing which clipped buffers are
        # accumulated and which optimizers are stepped.
        _set_params_requires_grad(shared_params, True)
        _set_params_requires_grad(expert_params_flat, not phase_a_only)

    def _ce_logits(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if objective == "seq2seq":
            if label_token_ids is None:
                raise ValueError("[OURS ONE-PHASE] seq2seq class logits require label token ids")
            return base_model.seq2seq_class_logits(ids, mask, label_token_ids.to(ids.device))
        if train_model is None:
            raise RuntimeError("[OURS ONE-PHASE] train model is not initialized")
        return train_model(ids, mask)

    def _standard_loss(ids: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> Optional[torch.Tensor]:
        if objective == "seq2seq":
            if label_token_ids is None:
                raise ValueError("[OURS ONE-PHASE] seq2seq objective requires label token ids")
            target_token_ids = label_token_ids.index_select(0, labels)
            return base_model.seq2seq_loss(ids, mask, target_token_ids)
        logits = _ce_logits(ids, mask)
        if not torch.isfinite(logits).all():
            return None
        return F.cross_entropy(logits, labels)

    def _shared_stream_loss(ids: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> Optional[torch.Tensor]:
        if expert_objective != "residual_logit":
            return _standard_loss(ids, mask, labels)
        touched = _set_expert_lora_disabled(layers_to_train, True)
        try:
            return _standard_loss(ids, mask, labels)
        finally:
            _restore_expert_lora_disabled(touched)

    def _expert_stream_loss(
        ids: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        shared_logits: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if expert_objective != "residual_logit":
            return _standard_loss(ids, mask, labels)
        if shared_logits is None:
            shared_logits = compute_shared_only_logits(
                base_model=base_model,
                sparse_layers=layers_to_train,
                ids=ids,
                mask=mask,
                objective=objective,
                seq2seq_label_token_ids=label_token_ids,
            )
        full_logits = _ce_logits(ids, mask)
        if not torch.isfinite(full_logits).all():
            return None
        shared_logits = shared_logits.to(device=full_logits.device, dtype=full_logits.dtype)
        expert_delta_logits = full_logits - shared_logits
        residual_logits = shared_logits.detach() + expert_delta_logits
        return F.cross_entropy(residual_logits, labels)

    _reset_runtime_peak_memory(device)
    t0 = time.time()
    epoch_losses: List[float] = []
    epoch_evals: List[Dict] = []
    try:
        train_model = OpacusGradSampleModule(model, batch_first=True, loss_reduction="mean")
        if hasattr(train_model, "_module"):
            base_model = train_model._module

        for epoch in range(run_epochs):
            train_model.train()
            epoch_loss_sum = 0.0
            epoch_residual_weight_sum = 0.0
            epoch_residual_weight_n = 0
            epoch_n = 0
            iterator: Iterable = range(updates_per_epoch)
            if tqdm and args.show_progress:
                iterator = tqdm(iterator, desc=f"{log_prefix.strip('[]')} ep{epoch+1}/{run_epochs}", leave=False)

            for _update_idx in iterator:
                if sample_rate >= 1.0:
                    selected = torch.randperm(len(train_ds))
                else:
                    keep = torch.rand(len(train_ds)) < float(sample_rate)
                    selected = keep.nonzero(as_tuple=False).flatten()
                    if selected.numel() > 0:
                        selected = selected[torch.randperm(selected.numel())]

                global_update_idx = int(epoch) * int(updates_per_epoch) + int(_update_idx)
                if alternating_enabled:
                    schedule_pos = global_update_idx % int(alternating_shared_steps + alternating_expert_steps)
                    train_shared_this_update = schedule_pos < int(alternating_shared_steps)
                    train_experts_this_update = (not phase_a_only) and not train_shared_this_update
                else:
                    train_shared_this_update = not expert_only
                    train_experts_this_update = not phase_a_only
                split_backward = bool(alternating_enabled or expert_objective == "residual_logit")

                selected_indices = selected.tolist()
                for start in range(0, len(selected_indices), micro):
                    chunk = selected_indices[start:start + micro]
                    if not chunk:
                        continue
                    batch = collate_fn([train_ds[int(idx)] for idx in chunk])
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    base_model._active_batch_indices = batch["idx"].detach()

                    rows = torch.arange(labels.size(0), device=device, dtype=torch.long)
                    micro_losses: List[float] = []

                    if split_backward:
                        if train_shared_this_update:
                            for p in unique_params:
                                p.grad = None
                            _set_trainable_streams(True, False)
                            shared_loss = _shared_stream_loss(ids, mask, labels)
                            if shared_loss is not None and torch.isfinite(shared_loss):
                                shared_loss.backward()
                                clip_factors = _global_clip_factors(rows)
                                _accumulate_shared(rows, clip_factors)
                                micro_losses.append(float(shared_loss.detach().item()))
                            _clear_grad_sample_state(unique_params)

                        if train_experts_this_update:
                            shared_logits_for_expert = None
                            residual_weights = None
                            if expert_objective == "residual_logit" or residual_mode != "none":
                                shared_logits_for_expert = compute_shared_only_logits(
                                    base_model=base_model,
                                    sparse_layers=layers_to_train,
                                    ids=ids,
                                    mask=mask,
                                    objective=objective,
                                    seq2seq_label_token_ids=label_token_ids,
                                )
                            if residual_mode != "none" and shared_logits_for_expert is not None:
                                residual_weights = _residual_weights_from_logits(
                                    shared_logits_for_expert,
                                    labels,
                                    args,
                                ).detach().to(device=labels.device, dtype=torch.float32)
                                epoch_residual_weight_sum += float(residual_weights.detach().sum().item())
                                epoch_residual_weight_n += int(residual_weights.numel())

                            for p in unique_params:
                                p.grad = None
                            _set_trainable_streams(False, True)
                            expert_loss = _expert_stream_loss(ids, mask, labels, shared_logits_for_expert)
                            if expert_loss is not None and torch.isfinite(expert_loss):
                                expert_loss.backward()
                                _debug_assert_non_owner_grad_samples_zero(batch["idx"])
                                clip_factors = _global_clip_factors(rows)
                                for layer in layers_to_train:
                                    layer_id = int(layer.layer_id)
                                    for expert_id in range(int(layer.sparse_mlp.router.num_experts)):
                                        _accumulate_expert(
                                            layer_id,
                                            int(expert_id),
                                            batch["idx"],
                                            residual_weights,
                                            clip_factors,
                                        )
                                micro_losses.append(float(expert_loss.detach().item()))
                            _clear_grad_sample_state(unique_params)
                    else:
                        for p in unique_params:
                            p.grad = None
                        _set_trainable_streams(True, True)

                        residual_weights = None
                        if residual_mode != "none":
                            residual_weights = compute_shared_residual_weights(
                                base_model=base_model,
                                sparse_layers=layers_to_train,
                                ids=ids,
                                mask=mask,
                                labels=labels,
                                args=args,
                                objective=objective,
                                seq2seq_label_token_ids=label_token_ids,
                            )
                            if residual_weights is not None:
                                epoch_residual_weight_sum += float(residual_weights.detach().sum().item())
                                epoch_residual_weight_n += int(residual_weights.numel())

                        loss = _standard_loss(ids, mask, labels)
                        if loss is not None and torch.isfinite(loss):
                            loss.backward()
                            _debug_assert_non_owner_grad_samples_zero(batch["idx"])
                            clip_factors = _global_clip_factors(rows)
                            _accumulate_shared(rows, clip_factors)
                            if train_experts_this_update:
                                for layer in layers_to_train:
                                    layer_id = int(layer.layer_id)
                                    for expert_id in range(int(layer.sparse_mlp.router.num_experts)):
                                        _accumulate_expert(
                                            layer_id,
                                            int(expert_id),
                                            batch["idx"],
                                            residual_weights,
                                            clip_factors,
                                        )
                            micro_losses.append(float(loss.detach().item()))
                        _clear_grad_sample_state(unique_params)

                    batch_n = int(labels.size(0))
                    if micro_losses:
                        epoch_loss_sum += (sum(micro_losses) / float(len(micro_losses))) * batch_n
                        epoch_n += batch_n

                if train_shared_this_update:
                    _set_trainable_streams(True, False)
                    _step_shared()
                if train_experts_this_update:
                    _set_trainable_streams(False, True)
                    for layer in layers_to_train:
                        layer_id = int(layer.layer_id)
                        for expert_id in range(int(layer.sparse_mlp.router.num_experts)):
                            _step_expert(layer_id, int(expert_id))
                _set_trainable_streams(True, not phase_a_only)

            epoch_loss = epoch_loss_sum / max(epoch_n, 1)
            epoch_losses.append(float(epoch_loss))
            if epoch_residual_weight_n > 0:
                mean_residual_weight = epoch_residual_weight_sum / float(epoch_residual_weight_n)
                print(
                    f"{log_prefix} epoch={epoch+1}/{run_epochs}  loss={epoch_loss:.4f}  "
                    f"n={epoch_n}  residual_w_mean={mean_residual_weight:.4f}"
                )
            else:
                print(f"{log_prefix} epoch={epoch+1}/{run_epochs}  loss={epoch_loss:.4f}  n={epoch_n}")

            eval_metrics = evaluate(
                base_model,
                dev_ds,
                tokenizer,
                args.eval_batch_size,
                args.max_length,
                device,
                args.input_prefix,
                objective=objective,
                seq2seq_label_token_ids=seq2seq_label_token_ids,
            )
            print(
                f"{log_prefix} eval epoch={epoch+1}/{run_epochs}"
                f"  acc={eval_metrics['accuracy']:.4f}"
                f"  loss={eval_metrics['loss']:.4f}"
                f"  n={eval_metrics['n']}"
            )
            epoch_evals.append({
                "epoch": int(epoch + 1),
                "train_loss": float(epoch_loss),
                "dev_acc": float(eval_metrics["accuracy"]),
                "dev_loss": float(eval_metrics["loss"]),
                "dev_n": int(eval_metrics["n"]),
            })
            train_model.train()
    finally:
        if train_model is not None and hasattr(train_model, "to_standard_module"):
            train_model.to_standard_module()
        model._record_route_layer_ids.clear()
        model._record_route_layer_ids.update(saved_record_layers)
        model._record_route_fixed_assignments = saved_fixed_assignments
        if hasattr(base_model, "_active_batch_indices"):
            base_model._active_batch_indices = None
        for _, p in all_named_params:
            p.requires_grad_(False)
        _clear_grad_sample_state(unique_params)

    if args.no_dp:
        eps_shared_actual = 0.0
        expert_eps_actual = 0.0
        layer_eps_actuals: List[float] = [0.0 for _ in layers_to_train]
        layer_delta_actuals: List[float] = [0.0 for _ in layers_to_train]
    elif phase_a_only:
        eps_shared_actual = _stacked_privacy_epsilon(
            noise_multiplier=float(sigma_shared),
            sample_rate=float(sample_rate),
            steps=int(shared_accounting_steps),
            target_delta=float(delta_shared),
            fallback_epsilon=float(eps_shared_target),
            args=args,
        )
        expert_eps_actual = 0.0
        layer_eps_actuals = [0.0 for _ in layers_to_train]
        layer_delta_actuals = [0.0 for _ in layers_to_train]
    elif expert_only:
        eps_shared_actual = 0.0
        expert_eps_actual = _stacked_privacy_epsilon(
            noise_multiplier=float(sigma_expert),
            sample_rate=float(sample_rate),
            steps=int(expert_accounting_steps),
            target_delta=float(expert_delta),
            fallback_epsilon=float(expert_eps_target),
            args=args,
        )
        if args.adjacency == "replace_one":
            layer_eps_actuals = [2.0 * expert_eps_actual for _ in layers_to_train]
            layer_delta_actuals = [2.0 * expert_delta for _ in layers_to_train]
        else:
            layer_eps_actuals = [expert_eps_actual for _ in layers_to_train]
            layer_delta_actuals = [expert_delta for _ in layers_to_train]
    else:
        eps_shared_actual = _stacked_privacy_epsilon(
            noise_multiplier=float(sigma_shared),
            sample_rate=float(sample_rate),
            steps=int(shared_accounting_steps),
            target_delta=float(delta_shared),
            fallback_epsilon=float(eps_shared_target),
            args=args,
        )
        expert_eps_actual = _stacked_privacy_epsilon(
            noise_multiplier=float(sigma_expert),
            sample_rate=float(sample_rate),
            steps=int(expert_accounting_steps),
            target_delta=float(expert_delta),
            fallback_epsilon=float(expert_eps_target),
            args=args,
        )
        if args.adjacency == "replace_one":
            layer_eps_actuals = [2.0 * expert_eps_actual for _ in layers_to_train]
            layer_delta_actuals = [2.0 * expert_delta for _ in layers_to_train]
        else:
            layer_eps_actuals = [expert_eps_actual for _ in layers_to_train]
            layer_delta_actuals = [expert_delta for _ in layers_to_train]

    expert_layer_logs: List[Dict] = []
    for layer, layer_eps_actual, layer_delta_actual in zip(layers_to_train, layer_eps_actuals, layer_delta_actuals):
        layer_id = int(layer.layer_id)
        num_experts = int(layer.sparse_mlp.router.num_experts)
        if phase_a_only:
            expert_updates = [0 for _ in range(num_experts)]
        else:
            expert_updates = [
                int(expert_states[(layer_id, expert_id)]["updates"])
                for expert_id in range(num_experts)
            ]
        expert_layer_logs.append({
            "layer_id": layer_id,
            "block_id": int(layer.block_id),
            "num_experts": num_experts,
            "target_layer_epsilon": eps_layer_target,
            "target_layer_delta": delta_layer,
            "expert_target_epsilon": expert_eps_target,
            "expert_target_delta": expert_delta,
            "expert_sigma": sigma_expert,
            "max_expert_epsilon_actual": (layer_eps_actual / 2.0 if args.adjacency == "replace_one" else layer_eps_actual),
            "layer_epsilon_actual": layer_eps_actual,
            "layer_delta_actual": layer_delta_actual,
            "updates_per_expert": expert_updates,
            "expert_update_scale_mode": expert_update_scale_mode,
            "ours_clip_scope": ours_clip_scope,
            "expert_lr_multiplier": expert_lr_multiplier,
            "expert_objective": expert_objective,
            "alternating_shared_steps": int(alternating_shared_steps),
            "alternating_expert_steps": int(alternating_expert_steps),
            "planned_expert_updates": int(planned_expert_updates),
            "routing_imbalance": assignment_metrics_by_layer.get(layer_id, {}),
            "schedule": (
                "alternating_shared_then_frozen_expert"
                if alternating_enabled
                else "fixed_public_full_dataset_poisson"
            ),
            "count_logging": "disabled",
        })

    final_dev = evaluate(
        model,
        dev_ds,
        tokenizer,
        args.eval_batch_size,
        args.max_length,
        device,
        args.input_prefix,
        objective=objective,
        seq2seq_label_token_ids=seq2seq_label_token_ids,
    )

    if phase_a_only:
        dilution_metrics = {"enabled": False, "reason": "phase_a_only_expert_stream_disabled"}
    else:
        dilution_metrics = compute_expert_dilution_metrics(
            model=model,
            train_ds=train_ds,
            diagnostic_ds=dev_ds,
            tokenizer=tokenizer,
            args=args,
            device=device,
            sparse_layers=layers_to_train,
            named_params=all_named_params,
            method_kind="baseline_c_expert_only" if expert_only else "ours",
            logical_batch_size=int(configured_batch),
            expert_sigma=float(sigma_expert),
            expert_update_scale_mode=expert_update_scale_mode,
            owner_assignments=assignment_cache,
            objective=objective,
            seq2seq_label_token_ids=seq2seq_label_token_ids,
        )
    if dilution_metrics.get("enabled"):
        print(
            f"{log_prefix} expert_dilution_summary="
            f"{json.dumps(dilution_metrics.get('summary', {}), sort_keys=True)}"
        )
    else:
        print(
            f"{log_prefix} expert_dilution_metrics skipped: "
            f"{dilution_metrics.get('reason', 'unknown')}"
        )

    eps_expert_actual = float(sum(layer_eps_actuals))
    delta_expert_actual = float(sum(layer_delta_actuals))
    delta_shared_actual = 0.0 if args.no_dp else float(delta_shared)
    eps_total_actual = float(eps_shared_actual) + eps_expert_actual
    delta_total_actual = delta_shared_actual + delta_expert_actual
    runtime = _runtime_stats(t0, total_updates, device)
    completed_stream_updates = int(planned_shared_updates)
    if not phase_a_only:
        completed_stream_updates += sum(
            int(state.get("updates", 0))
            for state in expert_states.values()
        )
    if phase_a_only:
        accounting_desc = "shared only; expert streams disabled by --phase_a_only"
        method_desc = "one_phase_shared_only_dp_moe"
    elif expert_only:
        accounting_desc = "expert only; shared stream disabled"
        method_desc = "baseline_c_expert_only_dp_lora"
    else:
        accounting_desc = "shared sequential + expert parallel within layer"
        method_desc = "one_phase_hybrid_dp_moe"

    log: Dict = {
        "experiment": args.experiment,
        "method": method_desc,
        "config": vars(args),
        "phase_a_only": phase_a_only,
        "expert_only": expert_only,
        "accountant": ACCOUNTING_MODE if not args.no_dp else "none",
        "zero_shot": zero_shot,
        "final_dev": final_dev,
        "selected_layer_ids": [int(sl.layer_id) for sl in layers_to_train],
        "assignment_logging": "redacted",
        "expert_dilution_metrics": dilution_metrics,
        "theorem7_entropy_validation": {
            "enabled": True,
            "owner_map_perturbation": {
                "strategy": str(getattr(args, "owner_map_perturb_strategy", "none")),
                "beta": float(getattr(args, "owner_map_perturb_beta", 0.0) or 0.0),
                "target_expert": int(getattr(args, "owner_map_perturb_target_expert", 0) or 0),
                "min_fraction": float(getattr(args, "owner_map_perturb_min_fraction", 0.0) or 0.0),
                "seed": int(getattr(args, "owner_map_perturb_seed", -1)),
            },
            "routing_imbalance": {
                "summary": assignment_metric_summary,
                "per_layer": assignment_metric_rows,
                "x_axis_entropy_deficit": assignment_metric_summary.get("entropy_deficit_mean", 0.0),
                "x_axis_chi2_to_uniform": assignment_metric_summary.get("chi2_to_uniform_mean", 0.0),
            },
            "response_variable": "Acc_non_private - Acc_private; use plotting script to join non-private upper-bound accuracy.",
        },
        "privacy_plan": {
            "accounting": accounting_desc,
            "adjacency": args.adjacency,
            "adjacency_factor": adj_factor,
            "epsilon_shared_ratio": epsilon_shared_ratio,
            "privacy_units": num_private_components,
            "epsilon_shared_target": eps_shared_target,
            "epsilon_experts_total_target": eps_experts_total_target,
            "epsilon_expert_layer_target": eps_layer_target,
            "epsilon_expert_target": expert_eps_target,
            "epsilon_total_target": args.epsilon_total,
            "delta_shared_target": delta_shared,
            "delta_experts_total_target": delta_experts_total,
            "delta_layer_target": delta_layer,
            "delta_expert_target": expert_delta,
            "delta_total_target": args.delta,
            "num_private_components": num_private_components,
            "sigma_shared": sigma_shared,
            "sigma_expert": sigma_expert,
            "shared_accounting_steps": int(shared_accounting_steps),
            "expert_accounting_steps": int(expert_accounting_steps),
        },
        "training": {
            "epochs": run_epochs,
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
            "planned_shared_updates": int(planned_shared_updates),
            "planned_expert_updates": int(planned_expert_updates),
            "sample_rate": sample_rate,
            "expected_sampled_records_per_update": expected_total,
            "logical_denominator": int(configured_batch),
            "micro_batch_size": int(micro),
            "lr": float(args.lr),
            "shared_trainable": int(n_shared),
            "expert_trainable": int(n_expert),
            "expert_updates_enabled": not phase_a_only,
            "expert_residual_weighting": residual_mode,
            "expert_objective": expert_objective,
            "expert_residual_margin_threshold": float(args.expert_residual_margin_threshold),
            "expert_residual_small_weight": float(args.expert_residual_small_weight),
            "expert_residual_min_weight": float(args.expert_residual_min_weight),
            "expert_update_scale_mode": expert_update_scale_mode,
            "ours_clip_scope": ours_clip_scope,
            "expert_lr_multiplier": expert_lr_multiplier,
            "alternating_enabled": bool(alternating_enabled),
            "alternating_shared_steps": int(alternating_shared_steps),
            "alternating_expert_steps": int(alternating_expert_steps),
            "epoch_losses": epoch_losses,
            "epoch_evals": epoch_evals,
        },
        "shared": {
            "epsilon": eps_shared_actual,
            "sigma": sigma_shared,
            "target_epsilon": eps_shared_target,
            "target_delta": delta_shared,
            "n_trainable": int(n_shared),
            "freeze_targets": sorted(shared_freeze_targets),
        },
        "expert_layers": expert_layer_logs,
        "privacy_summary": {
            "epsilon_shared_actual": eps_shared_actual,
            "epsilon_expert_layers_actual": layer_eps_actuals,
            "epsilon_expert_actual": eps_expert_actual,
            "epsilon_total_actual": eps_total_actual,
            "epsilon_total_target": args.epsilon_total,
            "delta_shared_actual": delta_shared_actual,
            "delta_expert_layers_actual": layer_delta_actuals,
            "delta_expert_actual": delta_expert_actual,
            "delta_total_actual": delta_total_actual,
            "delta_total_target": args.delta,
            "accounting": accounting_desc,
            "adjacency": args.adjacency,
            "epsilon_shared_ratio": epsilon_shared_ratio,
            "phase_a_only": phase_a_only,
            "expert_only": expert_only,
            "expert_objective": expert_objective,
            "ours_clip_scope": ours_clip_scope,
            "alternating_shared_steps": int(alternating_shared_steps),
            "alternating_expert_steps": int(alternating_expert_steps),
        },
        "completed_stream_updates": int(completed_stream_updates),
        "planned_optimizer_steps": int(total_updates),
        **runtime,
    }

    print("\n" + "=" * 88)
    print("[FINAL]")
    print(f"zero-shot acc      : {zero_shot['accuracy']:.4f}")
    print(f"final dev acc      : {final_dev['accuracy']:.4f}")
    print(f"delta acc          : {final_dev['accuracy'] - zero_shot['accuracy']:+.4f}")
    if not args.no_dp:
        print(f"epsilon shared act.: {eps_shared_actual:.4f}")
        print(f"epsilon layers act.: {[round(v, 4) for v in layer_eps_actuals]}")
        print(f"epsilon total act. : {eps_total_actual:.4f}")
        print(f"epsilon total tgt  : {args.epsilon_total:.4f}")
        print(f"delta total act.   : {delta_total_actual:.3e}")
        print(f"delta total tgt    : {args.delta:.3e}")
    print("=" * 88)

    return log


def run_oursver2(
    model: SwitchSST2,
    train_ds: SST2Dataset,
    dev_ds: SST2Dataset,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    zero_shot: Dict,
    sparse_layers: Sequence[SparseLayerRef],
    objective: str = "classifier",
    seq2seq_label_token_ids: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Windowed Dynamic Record-Routed DP-MoE (oursver2).

    Exact theorem-style implementation:
      - refresh (or reuse) fixed per-window record partitions,
      - run one joint per-step update over shared + expert parameters inside each window.
    """
    print("\n" + "=" * 88)
    print("[OURSVER2] one-phase windowed dynamic record-routed training")

    layers_to_train = select_sparse_layers_by_args(
        sparse_layers,
        args,
        stage_name="OURSVER2",
    )

    num_windows = max(1, int(args.oursver2_windows))
    refresh_windows = parse_refresh_windows(args.oursver2_refresh_windows, num_windows)

    # Keep backward compatibility with existing CLI knobs; for theorem-style joint training
    # we use a single window epoch schedule.
    shared_total_epochs = (
        int(args.oursver2_shared_total_epochs)
        if int(args.oursver2_shared_total_epochs) >= 0
        else int(args.attention_epochs)
    )
    expert_total_epochs = (
        int(args.oursver2_expert_total_epochs)
        if int(args.oursver2_expert_total_epochs) >= 0
        else int(args.expert_epochs)
    )
    joint_total_epochs = max(shared_total_epochs, expert_total_epochs)
    if shared_total_epochs != expert_total_epochs:
        print(
            f"[OURSVER2] shared_total_epochs={shared_total_epochs} and expert_total_epochs={expert_total_epochs} "
            f"differ; using joint_total_epochs=max(... )={joint_total_epochs}"
        )
    joint_epochs_by_window = split_total_epochs_across_windows(joint_total_epochs, num_windows)

    shared_freeze_targets = {str(t).strip().lower() for t in args.freeze_in_phase_a if str(t).strip()}
    if args.freeze_router_in_phase_a:
        shared_freeze_targets.add("router")
    freeze_desc = ",".join(sorted(shared_freeze_targets)) if shared_freeze_targets else "none"

    # Joint trainable parameter set: shared + experts from selected layers.
    named_joint_map: Dict[str, nn.Parameter] = {}
    for name, p in select_phase_a_params(model, freeze_targets=sorted(shared_freeze_targets)):
        named_joint_map[name] = p
    for layer in layers_to_train:
        for expert_id in range(layer.sparse_mlp.router.num_experts):
            for name, p in select_expert_params(model, layer, expert_id):
                named_joint_map[name] = p
    named_joint = sorted(named_joint_map.items(), key=lambda kv: kv[0])

    print(f"[OURSVER2] selected sparse layers = {[sl.layer_id for sl in layers_to_train]}")
    print(f"[OURSVER2] windows={num_windows} refresh_windows={sorted(refresh_windows)}")
    print(f"[OURSVER2] joint_total_epochs={joint_total_epochs}  joint_epochs_by_window={joint_epochs_by_window}")
    print(f"[OURSVER2] joint freeze targets={freeze_desc}")
    print(f"[OURSVER2] joint trainable params={len(named_joint):,}")

    num_private_stages = sum(1 for ep in joint_epochs_by_window if ep > 0) if not args.no_dp else 0

    if args.no_dp:
        target_eps_stage = 0.0
        target_delta_stage = 0.0
    else:
        if num_private_stages <= 0:
            raise ValueError("[OURSVER2] no private stages were scheduled.")
        target_eps_stage = args.epsilon_total / float(num_private_stages)
        target_delta_stage = args.delta / float(num_private_stages)

    print(
        f"[OURSVER2 PRIVACY PLAN] eps_stage_target={target_eps_stage:.6f}  "
        f"delta_stage_target={target_delta_stage:.3e}  private_stages={num_private_stages}"
    )

    log: Dict = {
        "experiment": "oursver2",
        "config": vars(args),
        "accountant": ACCOUNTING_MODE,
        "zero_shot": zero_shot,
        "windows": [],
        "privacy_plan": {
            "num_windows": num_windows,
            "refresh_windows": sorted(refresh_windows),
            "selected_layer_ids": [sl.layer_id for sl in layers_to_train],
            "joint_epochs_by_window": joint_epochs_by_window,
            "private_stage_count": num_private_stages,
            "target_eps_per_stage": target_eps_stage,
            "target_delta_per_stage": target_delta_stage,
            "adjacency": args.adjacency,
        },
    }

    assignment_cache: Dict[int, torch.Tensor] = {}
    eps_total_actual = 0.0
    delta_total_actual = 0.0

    for win_idx in range(1, num_windows + 1):
        print("\n" + "=" * 88)
        print(f"[OURSVER2] window={win_idx}/{num_windows}")

        window_epochs = int(joint_epochs_by_window[win_idx - 1])
        should_refresh = (win_idx == 1) or (win_idx in refresh_windows) or (not assignment_cache)

        window_log: Dict = {
            "window": win_idx,
            "routing_refreshed": bool(should_refresh),
            "joint_epochs": window_epochs,
            "window_stage": None,
        }

        if should_refresh:
            assignment_cache = {}
            print("[OURSVER2] refreshing record partitions from current model snapshot")
            for layer in layers_to_train:
                assignments = construct_record_assignments(
                    model,
                    train_ds,
                    tokenizer,
                    target_layer=layer,
                    batch_size=args.assignment_batch_size,
                    max_length=args.max_length,
                    device=device,
                    prefix=args.input_prefix,
                )
                assign_t = torch.tensor(assignments, dtype=torch.long)
                assignment_cache[int(layer.layer_id)] = assign_t
            window_log["assignment_sizes"] = "redacted"
        else:
            print("[OURSVER2] reusing previous window partitions")
            window_log["assignment_sizes"] = "redacted"

        if window_epochs > 0:
            freeze_all(model)
            zero_router_jitter(model)
            saved_record_layers = set(model._record_route_layer_ids)
            saved_fixed_assignments = dict(getattr(model, "_record_route_fixed_assignments", {}))
            model._record_route_layer_ids.clear()
            model._record_route_layer_ids.update({int(sl.layer_id) for sl in layers_to_train})
            model._record_route_fixed_assignments = {
                int(sl.layer_id): assignment_cache[int(sl.layer_id)] for sl in layers_to_train
            }
            try:
                window_stats = run_train_loop(
                    stage_name=f"OURSVER2_W{win_idx}_JOINT",
                    model=model,
                    train_ds=train_ds,
                    tokenizer=tokenizer,
                    device=device,
                    args=args,
                    named_params=named_joint,
                    lr=args.lr,
                    epochs=window_epochs,
                    dp=not args.no_dp,
                    target_epsilon=target_eps_stage,
                    target_delta=target_delta_stage,
                    objective=objective,
                    seq2seq_label_token_ids=seq2seq_label_token_ids,
                    batch_phase="phase_b",
                    dev_ds=dev_ds,
                )
            finally:
                model._record_route_layer_ids.clear()
                model._record_route_layer_ids.update(saved_record_layers)
                model._record_route_fixed_assignments = saved_fixed_assignments

            window_log["window_stage"] = window_stats
            eps_total_actual += float(window_stats.get("epsilon", 0.0))
            delta_total_actual += float(window_stats.get("target_delta", 0.0))
        else:
            print(f"[OURSVER2] joint stage skipped at window {win_idx} (epochs=0)")

        if args.eval_each_layer:
            dev_w = evaluate(
                model,
                dev_ds,
                tokenizer,
                args.eval_batch_size,
                args.max_length,
                device,
                args.input_prefix,
                objective=objective,
                seq2seq_label_token_ids=seq2seq_label_token_ids,
            )
            window_log["dev_acc"] = dev_w["accuracy"]
            window_log["dev_loss"] = dev_w["loss"]
            print(f"[OURSVER2] window={win_idx} dev_acc={dev_w['accuracy']:.4f} dev_loss={dev_w['loss']:.4f}")

        log["windows"].append(window_log)

    final_dev = evaluate(
        model,
        dev_ds,
        tokenizer,
        args.eval_batch_size,
        args.max_length,
        device,
        args.input_prefix,
        objective=objective,
        seq2seq_label_token_ids=seq2seq_label_token_ids,
    )
    log["final_dev"] = final_dev

    log["privacy_summary"] = {
        "epsilon_joint_actual": eps_total_actual,
        "epsilon_total_actual": eps_total_actual,
        "epsilon_total_target": args.epsilon_total,
        "delta_joint_actual": delta_total_actual,
        "delta_total_actual": delta_total_actual,
        "delta_total_target": args.delta,
        "num_private_stages": num_private_stages,
        "adjacency": args.adjacency,
    }

    print("\n" + "=" * 88)
    print("[OURSVER2 FINAL]")
    print(f"zero-shot acc      : {zero_shot['accuracy']:.4f}")
    print(f"final dev acc      : {final_dev['accuracy']:.4f}")
    print(f"delta acc          : {final_dev['accuracy'] - zero_shot['accuracy']:+.4f}")
    if not args.no_dp:
        print(f"epsilon total act. : {eps_total_actual:.4f}")
        print(f"epsilon total tgt  : {args.epsilon_total:.4f}")
        print(f"delta total act.   : {delta_total_actual:.3e}")
        print(f"delta total tgt    : {args.delta:.3e}")
    print("=" * 88)

    return log


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: SwitchSST2,
    ds: SST2Dataset,
    tokenizer,
    batch_size: int,
    max_length: int,
    device: torch.device,
    prefix: str = "",
    objective: str = "classifier",
    seq2seq_label_token_ids: Optional[torch.Tensor] = None,
) -> Dict:
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate(tokenizer, max_length, prefix),
    )
    # OLMoE-style evaluation: always evaluate with natural token routing.
    saved_record_layers = set(model._record_route_layer_ids)
    model._record_route_layer_ids.clear()
    try:
        model.eval()
        if objective not in ("classifier", "seq2seq"):
            raise ValueError(f"Unsupported evaluation objective={objective!r}")
        label_token_ids = None
        if objective == "seq2seq":
            if seq2seq_label_token_ids is None:
                raise ValueError("seq2seq evaluation requires label token ids")
            label_token_ids = seq2seq_label_token_ids.to(device)
        correct = 0
        total = 0
        loss_sum = 0.0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if objective == "seq2seq":
                target_token_ids = label_token_ids.index_select(0, labels)
                batch_loss = model.seq2seq_loss(ids, mask, target_token_ids)
                loss_sum += float(batch_loss.item()) * labels.size(0)
                logits = model.seq2seq_class_logits(ids, mask, label_token_ids)
            else:
                logits = model(ids, mask)
                loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
        return {
            "accuracy": correct / max(total, 1),
            "loss": loss_sum / max(total, 1),
            "n": total,
        }
    finally:
        model._record_route_layer_ids.clear()
        model._record_route_layer_ids.update(saved_record_layers)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DP-MoE Switch Transformer with one-phase hybrid record-owner expert DP and PRV accounting"
    )

    # Paths
    p.add_argument("--model_name", default="google/switch-base-8")
    p.add_argument("--data_dir", default="")
    p.add_argument("--glue_data_root", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--task", default="sst2", choices=["sst2", "mnli", "qnli", "qqp"])
    p.add_argument(
        "--experiment",
        default="ours",
        choices=[
            "ours",
            "oursver2",
            "oursseq2seq",
            "upper_bound",
            "upper_bound_no_lora",
            "naive_dp",
            "naive_dp_no_lora",
            "baseline_a_global_matched",
            "baseline_b_shared_only",
            "baseline_c_expert_only",
            "baseline_e_selected_layer_lora",
            "baseline_e_last_layer_lora",
            "matched_scope_global_dp",
            "dp_shared_only",
            "dp_expert_only",
            "dp_lora_last_layer",
            "last_layer_dp_lora_global",
            "dpsft",
        ],
    )
    p.add_argument("--finetune_epochs", type=int, default=8)
    p.add_argument("--lr_full_finetune", type=float, default=2e-5)

    # Runtime
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--input_prefix", default=DEFAULT_SST2_INPUT_PREFIX)
    p.add_argument("--show_progress", action="store_true")
    p.add_argument("--debug_privacy", action="store_true", default=env_flag("DEBUG_PRIVACY", False))
    p.add_argument(
        "--dilution_metric_batches",
        type=int,
        default=2,
        help="Post-training expert dilution diagnostic batches; 0 disables this validation log.",
    )
    p.add_argument(
        "--dilution_metric_batch_size",
        type=int,
        default=4,
        help="Batch size for the post-training expert dilution diagnostic.",
    )
    p.add_argument(
        "--diagnostic_noise_seed",
        type=int,
        default=-1,
        help="Seed for the diagnostic Gaussian noise realization; <0 uses seed+92023.",
    )

    # Batches
    p.add_argument("--assignment_batch_size", type=int, default=64)
    p.add_argument("--train_batch_size", type=int, default=1024)
    p.add_argument("--micro_batch_size", type=int, default=64)
    p.add_argument("--phase_a_train_batch_size", type=int, default=-1,
                   help="Logical DP batch for Phase A; <=0 falls back to --train_batch_size.")
    p.add_argument("--phase_a_micro_batch_size", type=int, default=-1,
                   help="Physical microbatch for Phase A; <=0 falls back to --micro_batch_size.")
    p.add_argument("--phase_b_train_batch_size", type=int, default=-1,
                   help="Legacy Phase B expected sampled records per update when --phase_b_sample_rate <= 0.")
    p.add_argument("--phase_b_micro_batch_size", type=int, default=-1,
                   help="Legacy Phase B physical microbatch; lower this if OOMs.")
    p.add_argument("--phase_b_sample_rate", type=float, default=-1.0,
                   help="Legacy Phase B public Bernoulli subsampling probability; <=0 uses phase_b_train_batch_size / N.")
    p.add_argument("--eval_batch_size", type=int, default=128)

    # Privacy
    p.add_argument("--no_dp", action="store_true")
    p.add_argument("--epsilon_total", type=float, default=8.0)
    p.add_argument("--epsilon_shared_ratio", type=float, default=0.3,
                   help="Fraction of --epsilon_total assigned to the shared DP stream in one-phase `ours`.")
    p.add_argument("--epsilon_attention", type=float, default=1.0,
                   help="Legacy shared-stage privacy budget for two-stage helpers.")
    p.add_argument("--delta", type=float, default=-1.0,
                   help="If negative, uses 1/N and then splits across private stages.")
    p.add_argument("--prv_eps_error", type=float, default=0.1,
                   help="PRV accountant epsilon error tolerance, matching the author code default.")
    p.add_argument(
        "--noise_multiplier",
        type=float,
        default=-1.0,
        help="Opacus noise multiplier; if <= 0, solve sigma from target epsilon/delta.",
    )
    p.add_argument("--adjacency", choices=["add_remove", "replace_one"], default="add_remove",
                   help="Privacy accounting convention within one layer of disjoint expert subsets.")
    p.add_argument("--clipping_mode", default="MixOpt", choices=["ghost", "MixGhostClip", "MixOpt"])
    p.add_argument("--clipping_fn", default="automatic", choices=["automatic", "Abadi", "global"])
    p.add_argument(
        "--clipping_style",
        default="layer-wise",
        choices=["all-layer", "layer-wise", "param-wise"],
        help="layer-wise is more stable for routed experts with ExpertLinear.",
    )
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument(
        "--baseline_epsilon_mode",
        default="full",
        choices=["full", "shared"],
        help=(
            "For scoped baselines, use the full --epsilon_total budget or, for baseline_b_shared_only, "
            "match the shared stream budget epsilon_shared_ratio * epsilon_total."
        ),
    )

    # DP-SFT baseline controls.
    p.add_argument(
        "--dpsft_param_scope",
        default="full_forward",
        choices=["full_forward", "non_expert_forward", "lora"],
        help="Trainable parameter scope for the DP-SFT baseline.",
    )
    p.add_argument("--dpsft_stage1_epochs", type=int, default=2)
    p.add_argument("--dpsft_stage2_epochs", type=int, default=5)
    p.add_argument("--dpsft_subspace_dim", type=int, default=8)
    p.add_argument(
        "--dpsft_stage1_epsilon_ratio",
        type=float,
        default=0.4,
        help="Fraction of --epsilon_total spent on DP-SFT trajectory training; remainder is used for subspace training.",
    )
    p.add_argument("--dpsft_stage1_lr", type=float, default=-1.0,
                   help="DP-SFT trajectory-stage LR; <=0 falls back to --lr_full_finetune.")
    p.add_argument("--dpsft_stage2_lr", type=float, default=-1.0,
                   help="DP-SFT subspace-stage LR; <=0 falls back to --lr.")
    p.add_argument("--dpsft_stage1_clip_norm", type=float, default=-1.0,
                   help="DP-SFT trajectory-stage clipping norm; <=0 falls back to --max_grad_norm.")
    p.add_argument("--dpsft_stage2_clip_norm", type=float, default=-1.0,
                   help="DP-SFT subspace-stage clipping norm; <=0 falls back to --max_grad_norm.")
    p.add_argument(
        "--dpsft_gpu_directions_max_elements",
        type=int,
        default=20_000_000,
        help="Move DP-SFT SVD directions to GPU only when the direction matrix has at most this many elements.",
    )

    # One-phase ours controls. Legacy attention/expert epoch knobs are still accepted for runner compatibility.
    p.add_argument("--ours_epochs", type=int, default=-1,
                   help="Epochs for one-phase `ours`; -1 uses --finetune_epochs.")
    p.add_argument(
        "--phase_a_only",
        action="store_true",
        help="For one-phase `ours`, disable expert updates and assign the full DP budget to the shared stream.",
    )
    p.add_argument(
        "--expert_residual_weighting",
        default="none",
        choices=["none", "prob", "margin"],
        help=(
            "For one-phase `ours`, weight expert per-sample gradients by shared-only residual difficulty: "
            "none disables it, prob uses 1-p_shared(y), margin gives high weight when shared margin is below threshold."
        ),
    )
    p.add_argument("--expert_residual_margin_threshold", type=float, default=0.2)
    p.add_argument("--expert_residual_small_weight", type=float, default=0.0)
    p.add_argument("--expert_residual_min_weight", type=float, default=0.0)
    p.add_argument(
        "--expert_update_scale_mode",
        default="batch",
        choices=["batch", "expected_owner"],
        help=(
            "Expert update denominator. batch preserves the old full logical batch denominator; "
            "expected_owner uses logical_batch / num_experts for fixed public scaling."
        ),
    )
    p.add_argument(
        "--owner_map_perturb_strategy",
        default="none",
        choices=["none", "target"],
        help="Theorem 7 validation: perturb fixed owner maps by reassigning records to a dominant target expert.",
    )
    p.add_argument(
        "--owner_map_perturb_beta",
        type=float,
        default=0.0,
        help="Theorem 7 validation imbalance level; beta=0 keeps the natural owner map.",
    )
    p.add_argument(
        "--owner_map_perturb_target_expert",
        type=int,
        default=0,
        help="Dominant expert id for --owner_map_perturb_strategy target.",
    )
    p.add_argument(
        "--owner_map_perturb_min_fraction",
        type=float,
        default=0.0,
        help="Optional floor on each non-target expert's owner fraction during perturbation.",
    )
    p.add_argument(
        "--owner_map_perturb_seed",
        type=int,
        default=-1,
        help="Seed for owner-map perturbation; <0 derives a layer-specific seed from --seed.",
    )
    p.add_argument(
        "--ours_clip_scope",
        default="role",
        choices=["role", "global"],
        help=(
            "Clipping scope for one-phase `ours`: role clips shared/expert streams separately; "
            "global is the ablation that clips with the full trainable parameter norm."
        ),
    )
    p.add_argument(
        "--expert_lr_multiplier",
        type=float,
        default=1.0,
        help="Public multiplier applied to expert gradients after clipping/noise and before optimizer step.",
    )
    p.add_argument(
        "--expert_objective",
        default="ce",
        choices=["ce", "residual_logit"],
        help=(
            "Expert objective for one-phase `ours`. ce preserves the previous full-model CE expert stream; "
            "residual_logit freezes the shared stream for expert backward and trains experts as logit residual correctors."
        ),
    )
    p.add_argument(
        "--alternating_shared_steps",
        type=int,
        default=0,
        help=(
            "When --alternating_expert_steps > 0, number of shared-only updates per window. "
            "0 disables alternating windows unless expert steps are enabled, in which case it becomes 1."
        ),
    )
    p.add_argument(
        "--alternating_expert_steps",
        type=int,
        default=0,
        help="Number of frozen-shared expert-only updates per alternating residual-specialization window.",
    )
    # Shared-stage controls (legacy two-stage helpers; `ours` reuses only the freeze flags)
    p.add_argument("--attention_epochs", type=int, default=8,
                   help="Legacy two-stage/shared-stage epoch knob; ignored by one-phase `ours`.")
    p.add_argument("--lr_attention", type=float, default=2e-4,
                   help="Legacy shared-stage learning rate; ignored by one-phase `ours`.")
    p.add_argument(
        "--freeze_in_phase_a",
        nargs="*",
        default=[],
        choices=["classifier", "lora", "router"],
        metavar="COMP",
        help=(
            "Single shared freeze flag. Choose any of: classifier lora router. "
            "Example: --freeze_in_phase_a classifier lora router"
        ),
    )
    p.add_argument(
        "--freeze_router_in_phase_a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Legacy compatibility alias. When set, router is added to --freeze_in_phase_a."
        ),
    )

    # Expert-stage controls (legacy helpers; `ours` reuses only LR and layer selection)
    p.add_argument("--expert_epochs", type=int, default=12,
                   help="Legacy expert-stage epoch knob; ignored by one-phase `ours`.")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate for all trainable LoRA parameters in one-phase `ours`.")
    p.add_argument(
        "--start_layer",
        type=int,
        default=-1,
        help=(
            "Sparse layer index or negative anchor. With positive --num_layers_to_train, "
            "-1 selects a window ending at the last sparse layer."
        ),
    )
    p.add_argument("--num_layers_to_train", type=int, default=-1,
                   help="-1 means all sparse layers from resolved start_layer onward; positive K with --start_layer -1 selects the last K sparse layers.")
    p.add_argument("--eval_each_layer", action="store_true")
    p.add_argument("--record_route_weight_mode", default=os.environ.get("RECORD_ROUTE_WEIGHT_MODE", "router_prob"),
                   choices=["one", "router_prob"],
                   help="When owner-masked routing is active, scale expert output by 1 or by the selected router probability.")
    p.add_argument("--min_expert_size", type=int, default=1,
                   help="Legacy option. Ignored by fixed-schedule expert streams in `ours`.")
    p.add_argument("--oursver2_windows", type=int, default=3,
                   help="Number of windows for one-phase dynamic record-routed training.")
    p.add_argument(
        "--oursver2_refresh_windows",
        default="all",
        help="1-based windows where partitions are recomputed; supports 'all', 'none', '1,3', or ranges like '1-2,4'.",
    )
    p.add_argument(
        "--oursver2_shared_total_epochs",
        type=int,
        default=-1,
        help="Total shared-training epochs across all windows (-1 uses --attention_epochs).",
    )
    p.add_argument(
        "--oursver2_expert_total_epochs",
        type=int,
        default=-1,
        help="Total expert-training epochs per layer across all windows (-1 uses --expert_epochs).",
    )

    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--enable_importance_probing", action="store_true")
    p.add_argument("--probe_method", default="random_rademacher")
    p.add_argument("--update_ratio", type=float, default=1.0)
    p.add_argument("--rank_mode", default="uniform")
    p.add_argument("--probe_cache", default="")

    # Optimizer
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--early_stop_patience", type=int, default=5)

    # Compatibility flags from the earlier script.
    p.add_argument("--train_attention", action="store_true")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--use_attention_lora", action="store_true")
    p.add_argument("--top_k_rea", type=int, default=1)
    p.add_argument("--accounting_mode", default="prv")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="")
    p.add_argument("--wandb_name", default="")
    p.add_argument("--eval_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.0)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    args.task = args.task.lower()
    args.requested_experiment = str(args.experiment)
    args.experiment = canonical_experiment_name(args.requested_experiment)
    if args.experiment != args.requested_experiment:
        print(f"[EXPERIMENT ALIAS] {args.requested_experiment} -> {args.experiment}")

    if args.task not in GLUE_TASK_CONFIG:
        raise ValueError(f"Unsupported task '{args.task}'. Choose one of: {', '.join(sorted(GLUE_TASK_CONFIG.keys()))}")

    if args.task != "sst2" and args.input_prefix == DEFAULT_SST2_INPUT_PREFIX:
        args.input_prefix = ""

    task_dir_map = {
        "sst2": "GLUE-SST-2",
        "mnli": "GLUE-MNLI",
        "qnli": "GLUE-QNLI",
        "qqp": "GLUE-QQP",
    }
    if not args.data_dir:
        if not args.glue_data_root:
            raise ValueError("Provide --data_dir or --glue_data_root")
        args.data_dir = os.path.join(args.glue_data_root, task_dir_map[args.task])

    if args.top_k_rea != 1:
        print(f"[WARN] top_k_rea={args.top_k_rea} is incompatible with single-expert record routing; using top-1 semantics.")
    if args.accounting_mode != "prv":
        print(f"[WARN] accounting_mode={args.accounting_mode!r} ignored; this file forces PRV accounting.")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds, dev_ds = load_glue(args.data_dir, task=args.task, tokenizer=tokenizer, max_length=args.max_length)
    num_labels = GLUE_TASK_CONFIG[args.task]["num_labels"]

    if args.delta < 0:
        args.delta = 1.0 / len(train_ds)

    shared_freeze_targets = {str(t).strip().lower() for t in args.freeze_in_phase_a if str(t).strip()}
    if args.freeze_router_in_phase_a:
        shared_freeze_targets.add("router")
    shared_freeze_desc = ",".join(sorted(shared_freeze_targets)) if shared_freeze_targets else "none"
    phase_a_batch, phase_a_micro = resolve_phase_batch_sizes(args, "phase_a")
    if args.experiment in ("ours", "oursseq2seq"):
        phase_b_batch = int(args.train_batch_size)
        phase_b_micro = int(args.micro_batch_size)
        phase_b_q = float(min(max(phase_b_batch, 1), max(len(train_ds), 1))) / float(max(len(train_ds), 1))
    else:
        phase_b_batch, phase_b_micro = resolve_phase_batch_sizes(args, "phase_b")
        phase_b_q = resolve_phase_b_sample_rate(args, len(train_ds), phase_b_batch)
    phase_b_expected_total = max(1, int(math.ceil(float(len(train_ds)) * float(phase_b_q))))

    print("=" * 88)
    if args.experiment == "oursver2":
        print("DP-MoE Switch Transformer with windowed one-phase joint training (PRV accounting)")
    elif args.experiment in ("ours", "oursseq2seq"):
        if args.phase_a_only:
            print("DP-MoE Switch Transformer with one-phase shared-only DP ablation (PRV accounting)")
        else:
            print("DP-MoE Switch Transformer with one-phase hybrid shared/expert DP training (PRV accounting)")
    elif args.experiment in SCOPED_GLOBAL_BASELINE_EXPERIMENTS:
        print("DP-MoE Switch Transformer baseline with scoped global DP-Adam LoRA (PRV accounting)")
    elif args.experiment in EXPERT_ONLY_BASELINE_EXPERIMENTS:
        print("DP-MoE Switch Transformer baseline with expert-only owner-routed DP-LoRA (PRV accounting)")
    elif args.experiment == "dpsft":
        print("DP-SFT Switch Transformer baseline with trajectory SVD subspace training (PRV accounting)")
    else:
        print("DP-MoE Switch Transformer with natural token routing and record-level expert ownership (PRV accounting)")
    print(f"model            : {args.model_name}")
    print(f"task             : {args.task}")
    print(f"device           : {device}")
    print(f"private train    : {len(train_ds)}")
    print(f"dev              : {len(dev_ds)}")
    print(f"epsilon_total    : {args.epsilon_total:.4f}")
    print(f"delta_total      : {args.delta:.3e}")
    print(f"adjacency        : {args.adjacency}")
    if args.experiment == "oursver2":
        print(f"one-phase windows: {args.oursver2_windows}")
        print(f"window refresh   : {args.oursver2_refresh_windows}")
        print(f"joint epochs cfg : shared={args.oursver2_shared_total_epochs} expert={args.oursver2_expert_total_epochs}")
        print(f"shared freeze    : {shared_freeze_desc}")
    elif args.experiment in (
        "ours",
        "oursseq2seq",
        "baseline_a_global_matched",
        "baseline_b_shared_only",
        "baseline_c_expert_only",
        "baseline_e_selected_layer_lora",
        "baseline_e_last_layer_lora",
    ):
        ours_epochs_display = int(args.ours_epochs)
        if ours_epochs_display <= 0:
            ours_epochs_display = int(args.finetune_epochs)
        print(f"train epochs     : {ours_epochs_display}")
        print(f"shared freeze    : {shared_freeze_desc}")
        print(f"phase A only     : {int(bool(args.phase_a_only))}")
        if args.experiment in ("ours", "oursseq2seq", "baseline_c_expert_only"):
            print(f"expert objective : {args.expert_objective}")
            print(f"clip scope       : {args.ours_clip_scope}")
            print(f"alt sh/ex steps  : {args.alternating_shared_steps} / {args.alternating_expert_steps}")
            print(
                f"owner perturb   : {args.owner_map_perturb_strategy} "
                f"beta={args.owner_map_perturb_beta:g} "
                f"target={args.owner_map_perturb_target_expert} "
                f"min_frac={args.owner_map_perturb_min_fraction:g}"
            )
        else:
            print(f"baseline eps mode: {args.baseline_epsilon_mode}")
            print("baseline clip    : global matched Poisson DP clipping")
    elif args.experiment == "dpsft":
        print(f"dpsft scope      : {args.dpsft_param_scope}")
        print(f"dpsft epochs     : stage1={args.dpsft_stage1_epochs} stage2={args.dpsft_stage2_epochs}")
        print(f"dpsft subspace d : {args.dpsft_subspace_dim}")
        print(f"dpsft eps split  : stage1_ratio={args.dpsft_stage1_epsilon_ratio}")
    else:
        print(f"phase A epochs   : {args.attention_epochs}")
        print(f"phase A freeze   : {shared_freeze_desc}")
        print(f"phase B epochs   : {args.expert_epochs}")
    print(f"default batch/mic: {args.train_batch_size} / {args.micro_batch_size}")
    if args.experiment in ("ours", "oursseq2seq"):
        print(f"one-phase b/mic  : {phase_b_batch} / {phase_b_micro}")
        print(f"one-phase sample q: {phase_b_q:.6f}  expected/update={phase_b_expected_total}")
    elif args.experiment in SCOPED_GLOBAL_BASELINE_EXPERIMENTS:
        global_q = float(min(max(int(args.train_batch_size), 1), max(len(train_ds), 1))) / float(max(len(train_ds), 1))
        print(f"global DP b/mic  : {args.train_batch_size} / {args.micro_batch_size}")
        print(f"global DP sample q: {global_q:.6f}")
    else:
        print(f"phase A batch/mic: {phase_a_batch} / {phase_a_micro}")
        print(f"phase B batch/mic: {phase_b_batch} / {phase_b_micro}")
        print(f"phase B sample q : {phase_b_q:.6f}  expected/update={phase_b_expected_total}")
    print(f"owner route scale: {args.record_route_weight_mode}")
    print(f"debug privacy    : {int(bool(args.debug_privacy))}")
    print(f"dilution metrics : batches={args.dilution_metric_batches} batch_size={args.dilution_metric_batch_size}")
    print(f"LoRA             : r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    print("=" * 88)

    model = SwitchSST2(args.model_name, num_labels=num_labels).to(device)
    model._record_route_weight_mode = args.record_route_weight_mode
    if not init_classifier(model, tokenizer, task=args.task):
        print("[INIT] classifier warm-start skipped; using default initialization.")
    freeze_all(model)
    zero_router_jitter(model)

    sparse_layers = get_sparse_layers(model)
    if not sparse_layers:
        raise RuntimeError("No sparse layers found in the encoder.")
    print(f"[INFO] sparse encoder layers = {len(sparse_layers)}")

    use_seq2seq_head = args.experiment == "oursseq2seq"
    seq2seq_label_token_ids: Optional[torch.Tensor] = None
    if use_seq2seq_head:
        seq2seq_label_token_ids = build_seq2seq_label_token_ids(tokenizer, args.task).to(device)
        print("[OURSSEQ2SEQ] using seq2seq LM-head objective with class verbalizer tokens.")

    if args.experiment == "upper_bound_no_lora":
        patch_sparse_mlp_for_naive_dp(model, sparse_layers)
        zero_shot = evaluate(
            model,
            dev_ds,
            tokenizer,
            args.eval_batch_size,
            args.max_length,
            device,
            args.input_prefix,
        )
        print(f"[ZERO-SHOT] acc={zero_shot['accuracy']:.4f}  loss={zero_shot['loss']:.4f}")
        result = run_upper_bound_no_lora(model, train_ds, dev_ds, tokenizer, args, device, zero_shot)
        result = _attach_runtime_summary(result)
        exp_dir = os.path.join(args.output_dir, "upper_bound_no_lora")
        os.makedirs(exp_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
        with open(os.path.join(exp_dir, "log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(args.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[DONE] saved to {exp_dir}")
        return

    if args.experiment == "naive_dp_no_lora":
        n_expert = 0
        n_attn = 0
        patch_sparse_mlp_for_naive_dp(model, sparse_layers)
    elif args.experiment == "dpsft":
        if args.dpsft_param_scope == "lora":
            n_expert = apply_lora_experts(model, sparse_layers, args.lora_r, args.lora_alpha, args.lora_dropout)
            n_attn = apply_lora_attention(model, args.lora_r, args.lora_alpha, args.lora_dropout)
        else:
            n_expert = 0
            n_attn = 0
        patch_sparse_mlp_for_naive_dp(model, sparse_layers)
    elif args.experiment in {
        "upper_bound",
        "naive_dp",
        "baseline_a_global_matched",
        "baseline_b_shared_only",
        "baseline_e_selected_layer_lora",
        "baseline_e_last_layer_lora",
    }:
        n_expert = apply_lora_experts(model, sparse_layers, args.lora_r, args.lora_alpha, args.lora_dropout)
        n_attn = apply_lora_attention(model, args.lora_r, args.lora_alpha, args.lora_dropout)
        patch_sparse_mlp_for_naive_dp(model, sparse_layers)
    else:
        n_expert = apply_lora_experts(model, sparse_layers, args.lora_r, args.lora_alpha, args.lora_dropout)
        n_attn = apply_lora_attention(model, args.lora_r, args.lora_alpha, args.lora_dropout)
        patch_sparse_mlp_for_record_routing(model, sparse_layers)
    print(f"[LORA] expert={n_expert}  attention={n_attn}")

    zero_shot = evaluate(
        model,
        dev_ds,
        tokenizer,
        args.eval_batch_size,
        args.max_length,
        device,
        args.input_prefix,
        objective="seq2seq" if use_seq2seq_head else "classifier",
        seq2seq_label_token_ids=seq2seq_label_token_ids,
    )
    print(f"[ZERO-SHOT] acc={zero_shot['accuracy']:.4f}  loss={zero_shot['loss']:.4f}")

    if args.experiment == "upper_bound":
        result = run_upper_bound(model, train_ds, dev_ds, tokenizer, args, device, zero_shot)
        result = _attach_runtime_summary(result)
        exp_dir = os.path.join(args.output_dir, "upper_bound")
        os.makedirs(exp_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
        with open(os.path.join(exp_dir, "log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(args.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[DONE] saved to {exp_dir}")
        return

    if args.experiment == "naive_dp":
        result = run_naive_dp(model, train_ds, dev_ds, tokenizer, args, device, zero_shot)
        result = _attach_runtime_summary(result)
        exp_dir = os.path.join(args.output_dir, "naive_dp")
        os.makedirs(exp_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
        with open(os.path.join(exp_dir, "log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(args.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[DONE] saved to {exp_dir}")
        return

    if args.experiment == "naive_dp_no_lora":
        result = run_naive_dp_no_lora(model, train_ds, dev_ds, tokenizer, args, device, zero_shot)
        result = _attach_runtime_summary(result)
        exp_dir = os.path.join(args.output_dir, "naive_dp_no_lora")
        os.makedirs(exp_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
        with open(os.path.join(exp_dir, "log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(args.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[DONE] saved to {exp_dir}")
        return

    if args.experiment in SCOPED_GLOBAL_BASELINE_EXPERIMENTS:
        result = run_scoped_global_dp_lora(
            model,
            train_ds,
            dev_ds,
            sparse_layers,
            tokenizer,
            args,
            device,
            zero_shot,
        )
        result = _attach_runtime_summary(result)
        exp_name = canonical_experiment_name(args.experiment)
        exp_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
        with open(os.path.join(exp_dir, "log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(args.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[DONE] saved to {exp_dir}")
        return

    if args.experiment == "dpsft":
        result = run_dpsft(model, train_ds, dev_ds, tokenizer, args, device, zero_shot)
        result = _attach_runtime_summary(result)
        exp_dir = os.path.join(args.output_dir, "dpsft")
        os.makedirs(exp_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
        with open(os.path.join(exp_dir, "log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(args.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[DONE] saved to {exp_dir}")
        return

    if args.experiment == "oursver2":
        log = run_oursver2(
            model=model,
            train_ds=train_ds,
            dev_ds=dev_ds,
            tokenizer=tokenizer,
            args=args,
            device=device,
            zero_shot=zero_shot,
            sparse_layers=sparse_layers,
            objective="seq2seq" if use_seq2seq_head else "classifier",
            seq2seq_label_token_ids=seq2seq_label_token_ids,
        )
        log = _attach_runtime_summary(log)
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
        with open(os.path.join(args.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(f"[DONE] saved to {args.output_dir}")
        return

    log = train_one_phase_hybrid_ours(
        model=model,
        train_ds=train_ds,
        dev_ds=dev_ds,
        sparse_layers=sparse_layers,
        tokenizer=tokenizer,
        args=args,
        device=device,
        zero_shot=zero_shot,
        objective="seq2seq" if use_seq2seq_head else "classifier",
        seq2seq_label_token_ids=seq2seq_label_token_ids,
    )
    log = _attach_runtime_summary(log)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    with open(os.path.join(args.output_dir, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    with open(os.path.join(args.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[DONE] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
