#!/usr/bin/env python3
"""
DeepSeek-VL2 backend with multi-method finetuning (OLMoE-style experiments).

Modes:
1) infer: single image+prompt generation
2) eval:  evaluate predictions from a JSONL dataset
3) finetune: run one or multiple methods:
   - upper_bound          (no-DP, LoRA)
   - upper_bound_no_lora  (no-DP, no-LoRA)
   - naive_dp             (DP, LoRA)
   - naive_dp_no_lora     (DP, no-LoRA)
   - ours                 (record-routing DP-MoE: phase A + per-layer/per-expert phase B)
   - all                  (run all methods sequentially)

Dataset JSONL fields (flexible aliases):
- image path: image | image_path | img | image_file
- question:   question | prompt | query | instruction
- answer:     answer | label | gt | target
- options:    optional list/string for MCQ prompts
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import time
import types
import weakref
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForCausalLM, AutoProcessor

try:
    from transformers import AutoModelForImageTextToText  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForImageTextToText = None

from fastDP import PrivacyEngine

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


ACCOUNTING_MODE = "rdp"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_python(v):
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v.detach().cpu().tolist()
    return v


def extract_epsilon_rdp(spent: Dict) -> float:
    for key in ("eps_rdp", "epsilon_rdp", "epsilon"):
        if key in spent and spent[key] is not None:
            return float(spent[key])
    raise ValueError(f"Cannot extract RDP epsilon from: {spent}")


def sanitize_parameters(params: Sequence[nn.Parameter], clamp: float = 100.0) -> None:
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=clamp, neginf=-clamp)
            p.data.clamp_(-clamp, clamp)


@dataclass
class VQASample:
    image_path: str
    question: str
    answer: Optional[str] = None
    options: Optional[str] = None
    meta: Optional[Dict] = None


class VQATrainDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[VQASample],
        processor,
        max_length: int,
    ):
        self.samples = list(samples)
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")

        prompt_text = build_prompt_for_generation(self.processor, s.question, s.options)
        full_text = build_prompt_for_training(self.processor, s.question, s.options, s.answer or "")

        prompt_inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        full_inputs = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        out: Dict[str, torch.Tensor] = {}
        for k, v in full_inputs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.squeeze(0)

        if "input_ids" not in out:
            raise RuntimeError("Processor output missing input_ids for training")

        labels = out["input_ids"].clone()
        prompt_len = int(prompt_inputs["input_ids"].shape[1]) if "input_ids" in prompt_inputs else 0
        prompt_len = max(0, min(prompt_len, labels.shape[0]))
        labels[:prompt_len] = -100
        out["labels"] = labels
        return out


class SubsetDataset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]


@dataclass
class SparseLayerRef:
    layer_id: int
    module_name: str
    sparse_mlp: nn.Module


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


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, use_expert_linear: bool = False):
        super().__init__()
        self.linear = base
        for p in self.linear.parameters():
            p.requires_grad_(False)

        A = nn.Linear(base.in_features, r, bias=False)
        B = nn.Linear(r, base.out_features, bias=False)
        if use_expert_linear:
            self.lora_A: nn.Module = ExpertLinear.from_linear(A)
            self.lora_B: nn.Module = ExpertLinear.from_linear(B)
        else:
            self.lora_A = A
            self.lora_B = B
        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / float(r)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
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


def _torch_dtype(dtype_arg: str):
    if dtype_arg == "fp32":
        return torch.float32
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def load_vlm(
    model_name: str,
    device: torch.device,
    dtype_arg: str,
    trust_remote_code: bool,
):
    dtype = _torch_dtype(dtype_arg)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    model = None
    if AutoModelForImageTextToText is not None:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            model = None
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    model.to(device)
    model.eval()
    return processor, model


def _first_non_empty(d: Dict, keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _format_options(options_obj) -> Optional[str]:
    if options_obj is None:
        return None
    if isinstance(options_obj, list):
        rows = [f"{chr(ord('A') + i)}. {x}" for i, x in enumerate(options_obj)]
        return "\n".join(rows)
    if isinstance(options_obj, str) and options_obj.strip():
        return options_obj.strip()
    return None


def load_jsonl_dataset(path: str, image_root: str = "", require_answer: bool = False) -> List[VQASample]:
    samples: List[VQASample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img = _first_non_empty(obj, ["image", "image_path", "img", "image_file"])
            q = _first_non_empty(obj, ["question", "prompt", "query", "instruction"])
            if not img or not q:
                continue
            if image_root and not os.path.isabs(img):
                img = os.path.join(image_root, img)
            ans = _first_non_empty(obj, ["answer", "label", "gt", "target", "output"])
            if require_answer and not ans:
                continue
            opts = _format_options(obj.get("options"))
            samples.append(VQASample(image_path=img, question=q, answer=ans, options=opts, meta=obj))
    return samples


def build_user_prompt(question: str, options: Optional[str]) -> str:
    if options:
        return f"{question}\n\nOptions:\n{options}\n\nGive the best answer."
    return question


def build_prompt_for_generation(processor, question: str, options: Optional[str]) -> str:
    user_text = build_user_prompt(question, options)
    if hasattr(processor, "format_generation_prompt"):
        return processor.format_generation_prompt(user_text)
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<image>\n{user_text}\nAssistant:"


def build_prompt_for_training(processor, question: str, options: Optional[str], answer: str) -> str:
    user_text = build_user_prompt(question, options)
    if hasattr(processor, "format_training_prompt"):
        return processor.format_training_prompt(user_text, answer)
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"<image>\n{user_text}\nAssistant: {answer}"


def _decode(processor, token_ids: torch.Tensor) -> str:
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        return tok.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
    return ""


@torch.no_grad()
def generate_answer(
    *,
    model,
    processor,
    image: Image.Image,
    question: str,
    options: Optional[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = build_prompt_for_generation(processor, question, options)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    do_sample = temperature > 0
    prev_mask = getattr(model, "_active_attention_mask", None)
    if "attention_mask" in inputs:
        model._active_attention_mask = inputs["attention_mask"]
    try:
        import logging
        pad_id = getattr(processor, "tokenizer", processor).pad_token_id
        _trf_logger = logging.getLogger("transformers.generation.utils")
        _prev_level = _trf_logger.level
        _trf_logger.setLevel(logging.ERROR)
        try:
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-6) if do_sample else 1.0,
                top_p=top_p,
                pad_token_id=pad_id,
            )
        finally:
            _trf_logger.setLevel(_prev_level)
    finally:
        model._active_attention_mask = prev_mask
    if "input_ids" in inputs:
        prompt_len = int(inputs["input_ids"].shape[1])
        gen_tail = gen_ids[:, prompt_len:]
    else:
        gen_tail = gen_ids
    return _decode(processor, gen_tail)


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_text(pred) == normalize_text(ref))


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def _split_parent_child(module_path: str) -> Tuple[str, str]:
    if "." not in module_path:
        return "", module_path
    parent, child = module_path.rsplit(".", 1)
    return parent, child


def _get_submodule(root: nn.Module, name: str) -> nn.Module:
    if not name:
        return root
    cur = root
    for part in name.split("."):
        cur = getattr(cur, part)
    return cur


def _is_attention_linear_name(name: str) -> bool:
    return any(x in name for x in (".self_attn.", ".attn.", ".attention.")) and isinstance(name, str)


def _is_expert_linear_name(name: str) -> bool:
    if ".experts." not in name:
        return False
    return any(
        x in name
        for x in (
            "gate_proj",
            "up_proj",
            "down_proj",
            "w1",
            "w2",
            "w3",
            ".wi.",
            ".wo.",
            ".ffn.",
            "fc1",
            "fc2",
        )
    )


def _is_router_name(name: str) -> bool:
    return ".gate." in name and ".experts." not in name


def _is_head_name(name: str) -> bool:
    return ("lm_head" in name) or name.startswith("score.") or name.startswith("classifier.")


def _is_vision_encoder_param_name(name: str) -> bool:
    return any(
        key in name
        for key in (
            "vision_tower",
            "image_tower",
            "vision_encoder",
            "vision_model",
            "visual_encoder",
        )
    )


def _is_multimodal_projector_param_name(name: str) -> bool:
    return any(
        key in name
        for key in (
            "mm_projector",
            "multi_modal_projector",
            "multimodal_projector",
            "vision_proj",
            "image_projection",
        )
    )


def _is_expert_param_name(name: str) -> bool:
    return ".experts." in name


def _is_sparse_moe_module(mod: nn.Module) -> bool:
    deepspeed_moe = getattr(mod, "deepspeed_moe", None)
    if deepspeed_moe is not None and hasattr(deepspeed_moe, "experts") and hasattr(deepspeed_moe, "gate"):
        return True
    if not hasattr(mod, "experts"):
        return False
    if hasattr(mod, "gate") and isinstance(getattr(mod, "gate"), nn.Linear):
        return True
    router = getattr(mod, "router", None)
    if router is not None and hasattr(router, "classifier"):
        return True
    return False


def get_sparse_layers(model: nn.Module) -> List[SparseLayerRef]:
    refs: List[SparseLayerRef] = []
    lid = 0
    for name, mod in model.named_modules():
        if any(name.startswith(f"{ref.module_name}.") for ref in refs):
            continue
        if _is_sparse_moe_module(mod):
            refs.append(SparseLayerRef(layer_id=lid, module_name=name, sparse_mlp=mod))
            lid += 1
    return refs


def _unwrap_sparse_moe(sparse_mlp: nn.Module) -> nn.Module:
    return getattr(sparse_mlp, "deepspeed_moe", sparse_mlp)


def _experts_container(sparse_mlp: nn.Module) -> nn.Module:
    experts = getattr(_unwrap_sparse_moe(sparse_mlp), "experts")
    return getattr(experts, "deepspeed_experts", experts)


def _num_experts(sparse_mlp: nn.Module) -> int:
    core = _unwrap_sparse_moe(sparse_mlp)
    if hasattr(core, "num_experts"):
        return int(getattr(core, "num_experts"))
    experts = _experts_container(sparse_mlp)
    if isinstance(experts, (nn.ModuleList, list, tuple)):
        return len(experts)
    if isinstance(experts, nn.ModuleDict):
        return len(experts)
    raise TypeError(f"Unsupported experts container: {type(experts)}")


def _get_expert_module(experts: nn.Module, expert_id: int) -> nn.Module:
    if hasattr(experts, "deepspeed_experts"):
        experts = getattr(experts, "deepspeed_experts")
    if isinstance(experts, nn.ModuleList):
        return experts[int(expert_id)]
    if isinstance(experts, nn.ModuleDict):
        key = f"expert_{expert_id}"
        if key in experts:
            return experts[key]
        return experts[str(expert_id)]
    if isinstance(experts, list):
        return experts[int(expert_id)]
    key = f"expert_{expert_id}"
    if hasattr(experts, key):
        return getattr(experts, key)
    if hasattr(experts, str(expert_id)):
        return getattr(experts, str(expert_id))
    raise TypeError(f"Unsupported experts container: {type(experts)}")


def _router_logits(sparse_mlp: nn.Module, flat_hidden: torch.Tensor) -> torch.Tensor:
    gate = getattr(_unwrap_sparse_moe(sparse_mlp), "gate", None)
    if isinstance(gate, nn.Linear):
        return gate(flat_hidden)
    wg = getattr(gate, "wg", None)
    if isinstance(wg, nn.Linear):
        x = flat_hidden.to(wg.weight.dtype) if flat_hidden.dtype != wg.weight.dtype else flat_hidden
        return wg(x)
    router = getattr(sparse_mlp, "router", None)
    if router is not None:
        if hasattr(router, "classifier"):
            classifier = getattr(router, "classifier")
            return classifier(flat_hidden)
        if isinstance(router, nn.Linear):
            return router(flat_hidden)
    raise RuntimeError(f"Cannot find router logits path for sparse module type={type(sparse_mlp)}")


def _forward_expert_with_lora_owner_mask(
    expert_mod: nn.Module,
    expert_input: torch.Tensor,
    owner_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if owner_mask is None:
        return expert_mod(expert_input)

    touched: List[Tuple[LoRALinear, bool, Optional[torch.Tensor]]] = []
    for mod in expert_mod.modules():
        if not isinstance(mod, LoRALinear):
            continue
        had_mask = hasattr(mod, "_record_owner_mask")
        previous = getattr(mod, "_record_owner_mask", None)
        mod._record_owner_mask = owner_mask
        touched.append((mod, had_mask, previous))
    try:
        return expert_mod(expert_input)
    finally:
        for mod, had_mask, previous in touched:
            if had_mask:
                mod._record_owner_mask = previous
            elif hasattr(mod, "_record_owner_mask"):
                delattr(mod, "_record_owner_mask")


def expert_forward_routing_table(
    expert_mod: nn.Module,
    flat_tokens: torch.Tensor,
    token_pos: torch.Tensor,
    seq_len: int,
    batch_size: int,
    owner_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    t_j = token_pos.numel()
    if t_j == 0:
        return torch.zeros(0, flat_tokens.shape[-1], device=flat_tokens.device, dtype=flat_tokens.dtype)

    token_to_sample = torch.div(token_pos, seq_len, rounding_mode="floor")
    for subm in expert_mod.modules():
        if isinstance(subm, ExpertLinear):
            subm._token_to_sample = token_to_sample.detach()
            subm._num_samples = batch_size

    return _forward_expert_with_lora_owner_mask(expert_mod, flat_tokens[token_pos], owner_mask)


def _native_topk_routing(
    sparse_mlp: nn.Module,
    flat_hidden: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = _router_logits(sparse_mlp, flat_hidden)
    probs = torch.softmax(logits.float(), dim=-1)
    core = _unwrap_sparse_moe(sparse_mlp)
    gate = getattr(core, "gate", None)
    top_k = int(getattr(gate, "k", getattr(sparse_mlp, "k", 1)))
    weights, selected = torch.topk(probs, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return weights.to(flat_hidden.dtype), selected, logits


def _select_primary_record_owners(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    num_experts: int,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    selected = selected_experts.view(batch_size, seq_len, -1)
    weights = routing_weights.view(batch_size, seq_len, -1)
    scores = torch.zeros(batch_size, num_experts, device=weights.device, dtype=weights.dtype)
    masked_weights = weights * valid_mask.unsqueeze(-1).to(weights.dtype)
    scores.scatter_add_(1, selected.reshape(batch_size, -1), masked_weights.reshape(batch_size, -1))
    return scores.argmax(dim=-1)


def _valid_token_mask(owner: nn.Module, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    mask = getattr(owner, "_active_attention_mask", None)
    if mask is None or tuple(mask.shape) != (batch_size, seq_len):
        return torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
    return mask.to(device).bool()


def _deepspeed_moe_forward(
    sparse_mlp: nn.Module,
    hidden_states: torch.Tensor,
    *,
    owner: Optional[nn.Module],
    layer_id: int,
    owner_masked: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, hidden_dim = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_dim)
    routing_weights, selected_experts, router_logits = _native_topk_routing(sparse_mlp, flat)
    num_experts = _num_experts(sparse_mlp)
    valid_mask = _valid_token_mask(owner, batch_size, seq_len, hidden_states.device) if owner is not None else (
        torch.ones(batch_size, seq_len, device=hidden_states.device, dtype=torch.bool)
    )
    forced_map = getattr(owner, "_forced_record_expert_by_layer", {}) if owner is not None else {}
    forced_expert = forced_map.get(layer_id)
    if forced_expert is None:
        record_experts = _select_primary_record_owners(
            selected_experts,
            routing_weights,
            batch_size=batch_size,
            seq_len=seq_len,
            num_experts=num_experts,
            valid_mask=valid_mask,
        )
    else:
        record_experts = torch.full(
            (batch_size,),
            int(forced_expert),
            device=hidden_states.device,
            dtype=torch.long,
        )
    sparse_mlp._last_raw_expert_index = selected_experts[:, 0].view(batch_size, seq_len).detach()
    sparse_mlp._last_raw_topk_expert_indices = selected_experts.view(batch_size, seq_len, -1).detach()
    sparse_mlp._last_raw_topk_routing_weights = routing_weights.view(batch_size, seq_len, -1).detach()
    sparse_mlp._last_record_expert_index = record_experts.detach()

    out_flat = torch.zeros_like(flat)
    experts = _experts_container(sparse_mlp)
    if owner is not None and getattr(owner, "_record_route_weight_mode", "router_prob") == "one":
        routing_weights = torch.ones_like(routing_weights)
    flat_valid = valid_mask.reshape(-1)
    token_to_sample_all = torch.arange(batch_size, device=flat.device).repeat_interleave(seq_len)
    for expert_id in range(num_experts):
        slot_mask = selected_experts == int(expert_id)
        token_mask = slot_mask.any(dim=-1) & flat_valid
        token_pos = torch.nonzero(token_mask, as_tuple=True)[0]
        if token_pos.numel() == 0:
            continue
        expert_mod = _get_expert_module(experts, expert_id)
        token_to_sample = token_to_sample_all.index_select(0, token_pos)
        expert_owner_mask = None
        if owner_masked:
            expert_owner_mask = record_experts.index_select(0, token_to_sample) == int(expert_id)
        expert_out = expert_forward_routing_table(
            expert_mod,
            flat,
            token_pos,
            seq_len,
            batch_size,
            owner_mask=expert_owner_mask,
        )
        token_weights = (routing_weights * slot_mask.to(routing_weights.dtype)).sum(dim=-1)
        expert_out = expert_out * token_weights.index_select(0, token_pos).unsqueeze(-1).to(expert_out.dtype)
        out_flat.index_add_(0, token_pos, expert_out.to(out_flat.dtype))

    counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts).to(flat.dtype)
    l_aux = router_logits.sum() * 0.0
    return out_flat.view(batch_size, seq_len, hidden_dim), l_aux, counts


def patch_sparse_mlp_for_record_routing(model: nn.Module, sparse_layers: Sequence[SparseLayerRef]) -> None:
    for sl in sparse_layers:
        mlp = sl.sparse_mlp
        if getattr(mlp, "_record_routing_patched", False):
            continue
        owner_ref = weakref.ref(model)
        layer_id = int(sl.layer_id)
        orig_fwd = mlp.forward

        def _patched_forward(self, hidden_states: torch.Tensor, _owner_ref=owner_ref, _layer_id=layer_id, _orig_fwd=orig_fwd):
            owner = _owner_ref()
            if owner is None:
                raise RuntimeError("Owner model reference was lost in patched sparse MLP")
            if hasattr(self, "deepspeed_moe"):
                if _layer_id not in owner._record_route_layer_ids:
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    flat = hidden_states.reshape(-1, hidden_dim)
                    weights, selected, _logits = _native_topk_routing(self, flat)
                    self._last_raw_expert_index = selected[:, 0].view(batch_size, seq_len).detach()
                    self._last_raw_topk_expert_indices = selected.view(batch_size, seq_len, -1).detach()
                    self._last_raw_topk_routing_weights = weights.view(batch_size, seq_len, -1).detach()
                    result = _orig_fwd(hidden_states)
                    self._record_route_returns_tuple = isinstance(result, tuple)
                    return result
                return _deepspeed_moe_forward(
                    self,
                    hidden_states,
                    owner=owner,
                    layer_id=_layer_id,
                    owner_masked=True,
                )

            batch_size, seq_len, hidden_dim = hidden_states.shape
            flat = hidden_states.reshape(-1, hidden_dim)
            router_logits = _router_logits(self, flat)
            raw_probs = torch.softmax(router_logits.float(), dim=-1).to(flat.dtype)
            raw_top1 = raw_probs.argmax(dim=-1).view(batch_size, seq_len)
            self._last_raw_expert_index = raw_top1.detach()

            if _layer_id not in owner._record_route_layer_ids:
                result = _orig_fwd(hidden_states)
                self._record_route_returns_tuple = isinstance(result, tuple)
                return result

            attn_mask = getattr(owner, "_active_attention_mask", None)
            if attn_mask is None:
                valid_mask = torch.ones(batch_size, seq_len, device=flat.device, dtype=torch.bool)
            else:
                valid_mask = attn_mask.to(flat.device).bool()

            forced_map = getattr(owner, "_forced_record_expert_by_layer", {})
            forced_expert = forced_map.get(_layer_id)
            n_experts = _num_experts(self)
            if forced_expert is not None:
                record_experts = torch.full((batch_size,), int(forced_expert), device=flat.device, dtype=torch.long)
            else:
                chosen: List[int] = []
                for i in range(batch_size):
                    active = raw_top1[i][valid_mask[i]]
                    if active.numel() == 0:
                        chosen.append(0)
                    else:
                        counts = torch.bincount(active.long(), minlength=n_experts)
                        chosen.append(int(counts.argmax().item()))
                record_experts = torch.tensor(chosen, device=flat.device, dtype=torch.long)
            self._last_record_expert_index = record_experts.detach()

            selected_ids = record_experts.unsqueeze(1).expand(batch_size, seq_len).reshape(-1)
            flat_valid = valid_mask.reshape(-1)
            weight_mode = getattr(owner, "_record_route_weight_mode", "one")
            if weight_mode == "router_prob":
                weights = raw_probs.gather(-1, selected_ids.view(-1, 1)).squeeze(-1)
            else:
                weights = torch.ones(batch_size * seq_len, device=flat.device, dtype=flat.dtype)
            weights = weights.to(flat.dtype)
            weights[~flat_valid] = 0.0

            out_flat = torch.zeros_like(flat)
            active_experts = torch.unique(selected_ids[flat_valid]).tolist()
            for expert_id in active_experts:
                token_pos = torch.nonzero((selected_ids == expert_id) & flat_valid, as_tuple=True)[0]
                if token_pos.numel() == 0:
                    continue
                expert_mod = _get_expert_module(_experts_container(self), int(expert_id))
                expert_out = expert_forward_routing_table(expert_mod, flat, token_pos, seq_len, batch_size)
                expert_out = expert_out * weights[token_pos].to(expert_out.dtype).view(-1, 1)
                out_flat.index_add_(0, token_pos, expert_out.to(out_flat.dtype))

            routed = out_flat.view(batch_size, seq_len, hidden_dim)
            if getattr(self, "_record_route_returns_tuple", False):
                return routed, router_logits
            return routed

        mlp.forward = types.MethodType(_patched_forward, mlp)
        mlp._record_routing_patched = True


def patch_sparse_mlp_for_naive_dp(model: nn.Module, sparse_layers: Sequence[SparseLayerRef]) -> None:
    for sl in sparse_layers:
        mlp = sl.sparse_mlp
        if getattr(mlp, "_naive_dp_patched", False):
            continue
        orig_fwd = mlp.forward

        returns_tuple = bool(getattr(mlp, "_record_route_returns_tuple", False))

        def _naive_forward(self, hidden_states: torch.Tensor, _orig=orig_fwd, _returns_tuple=returns_tuple):
            if hasattr(self, "deepspeed_moe"):
                return _deepspeed_moe_forward(
                    self,
                    hidden_states,
                    owner=None,
                    layer_id=-1,
                    owner_masked=False,
                )
            batch_size, seq_len, hidden_dim = hidden_states.shape
            flat = hidden_states.reshape(-1, hidden_dim)

            router_logits = _router_logits(self, flat)
            probs = torch.softmax(router_logits.float(), dim=-1).to(flat.dtype)
            selected = probs.argmax(dim=-1)
            out_flat = torch.zeros_like(flat)
            n_experts = _num_experts(self)

            for expert_id in range(n_experts):
                token_pos = torch.nonzero(selected == expert_id, as_tuple=True)[0]
                if token_pos.numel() == 0:
                    continue
                expert_mod = _get_expert_module(self.experts, int(expert_id))
                token_to_sample = torch.div(token_pos, seq_len, rounding_mode="floor")
                for subm in expert_mod.modules():
                    if isinstance(subm, ExpertLinear):
                        subm._token_to_sample = token_to_sample.detach()
                        subm._num_samples = batch_size
                expert_out = expert_mod(flat[token_pos])
                expert_out = expert_out * probs[token_pos, expert_id].to(expert_out.dtype).view(-1, 1)
                out_flat.index_add_(0, token_pos, expert_out.to(out_flat.dtype))

            routed = out_flat.view(batch_size, seq_len, hidden_dim)
            if _returns_tuple:
                return routed, router_logits
            return routed

        mlp.forward = types.MethodType(_naive_forward, mlp)
        mlp._naive_dp_patched = True


def freeze_routers(model: nn.Module) -> None:
    for name, mod in model.named_modules():
        if ".experts." in name:
            continue
        if name.endswith(".gate") and isinstance(mod, nn.Linear):
            for p in mod.parameters():
                p.requires_grad_(False)
        if name.endswith(".router") and hasattr(mod, "classifier"):
            for p in mod.parameters():
                p.requires_grad_(False)


def apply_expert_linear_no_lora(sparse_layers: Sequence[SparseLayerRef]) -> int:
    """
    Replace expert-internal nn.Linear with ExpertLinear for no-LoRA DP paths.
    """
    replaced = 0
    for sl in sparse_layers:
        experts = getattr(sl.sparse_mlp, "experts")
        n_experts = _num_experts(sl.sparse_mlp)
        for eid in range(n_experts):
            expert = _get_expert_module(experts, eid)
            to_swap: List[Tuple[str, nn.Linear]] = []
            for name, mod in expert.named_modules():
                if isinstance(mod, nn.Linear) and not isinstance(mod, ExpertLinear):
                    to_swap.append((name, mod))
            for name, old in to_swap:
                if not name:
                    continue
                parent_name, child_name = _split_parent_child(name)
                parent = _get_submodule(expert, parent_name)
                setattr(parent, child_name, ExpertLinear.from_linear(old))
                replaced += 1
    return replaced


def register_expert_linear_fastdp() -> None:
    from fastDP import supported_layers_grad_samplers as sgs

    if ExpertLinear in sgs._supported_layers_norm_sample_AND_clipping:
        return

    def _sampler(layer: ExpertLinear, A: torch.Tensor, B: torch.Tensor, _mode: str) -> None:
        if A is None or B is None or layer._token_to_sample is None or layer._num_samples is None:
            return

        token_to_sample = layer._token_to_sample.to(A.device)
        num_samples = int(layer._num_samples)
        grad_w = torch.zeros(num_samples, B.size(1), A.size(1), device=A.device, dtype=A.dtype)
        grad_w.index_add_(0, token_to_sample, torch.einsum("no,ni->noi", B, A))
        layer.weight.grad_sample = grad_w.detach()
        layer.weight.norm_sample = grad_w.flatten(1).norm(dim=1).detach()

        if layer.bias is not None:
            grad_b = torch.zeros(num_samples, B.size(1), device=A.device, dtype=A.dtype)
            grad_b.index_add_(0, token_to_sample, B)
            layer.bias.grad_sample = grad_b.detach()
            layer.bias.norm_sample = grad_b.norm(dim=1).detach()

        layer._token_to_sample = None
        layer._num_samples = None

    sgs._supported_layers_norm_sample_AND_clipping[ExpertLinear] = (_sampler, sgs._clip_linear_grad)


def _patch_fastdp_use_full_backward_hook() -> None:
    try:
        import fastDP.autograd_grad_sample as _ags
        from fastDP.supported_layers_grad_samplers import _supported_layers_norm_sample_AND_clipping as _supported
    except Exception:
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


def _patch_fastdp_for_expert_linear() -> None:
    try:
        import fastDP.autograd_grad_sample as _ags
        from fastDP.supported_layers_grad_samplers import _create_or_extend_private_grad as _create_private_grad
    except Exception:
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
            raise ValueError(f"Unknown clipping function {clipping_fn}.")

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


def apply_lora_with_filter(
    model: nn.Module,
    *,
    filter_fn,
    r: int,
    alpha: float,
    dropout: float,
    use_expert_linear: bool = False,
) -> int:
    to_replace: List[str] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not isinstance(mod, LoRALinear) and filter_fn(name):
            to_replace.append(name)

    for module_name in to_replace:
        parent_name, child_name = _split_parent_child(module_name)
        parent = _get_submodule(model, parent_name)
        old = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(old, r, alpha, dropout, use_expert_linear=use_expert_linear))
    return len(to_replace)


def convert_lora_expert_bases_to_expertlinear(model: nn.Module) -> int:
    """
    For LoRA-based expert training, convert LoRALinear.base to ExpertLinear so
    token->sample routing-table aggregation remains valid under DP.
    """
    converted = 0
    for name, mod in model.named_modules():
        if ".experts." not in name:
            continue
        if not isinstance(mod, LoRALinear):
            continue
        base = mod.linear
        if isinstance(base, ExpertLinear):
            continue
        if isinstance(base, nn.Linear):
            mod.linear = ExpertLinear.from_linear(base)
            converted += 1
    return converted


def select_named_params(
    model: nn.Module,
    predicate,
) -> List[Tuple[str, nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if predicate(name, p)]


def select_all_lora_plus_head(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    def _pred(name: str, _p: nn.Parameter) -> bool:
        return ("lora_A" in name or "lora_B" in name or _is_head_name(name))

    return select_named_params(model, _pred)


def select_phase_a_lora(model: nn.Module, train_router: bool, freeze_vision: bool = False) -> List[Tuple[str, nn.Parameter]]:
    def _pred(name: str, _p: nn.Parameter) -> bool:
        if freeze_vision and _is_vision_encoder_param_name(name):
            return False
        if _is_expert_param_name(name):
            return False
        if _is_router_name(name):
            return bool(train_router)
        return True

    return select_named_params(model, _pred)


def select_phase_b_lora(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    def _pred(name: str, _p: nn.Parameter) -> bool:
        return ("lora_A" in name or "lora_B" in name) and ".experts." in name

    return select_named_params(model, _pred)


def select_expert_lora_params(model: nn.Module, layer: SparseLayerRef, expert_id: int) -> List[Tuple[str, nn.Parameter]]:
    pattern_a = f"{layer.module_name}.experts.{expert_id}."
    pattern_b = f"{layer.module_name}.experts.expert_{expert_id}."
    pattern_c = f"{layer.module_name}.deepspeed_moe.experts.deepspeed_experts.{expert_id}."
    out: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if pattern_a in name or pattern_b in name or pattern_c in name:
            out.append((name, p))
    return out


def select_full_params(model: nn.Module, freeze_vision: bool) -> List[Tuple[str, nn.Parameter]]:
    def _pred(name: str, _p: nn.Parameter) -> bool:
        if freeze_vision and _is_vision_encoder_param_name(name):
            return False
        return True

    return select_named_params(model, _pred)


def enable_only(named_params: Sequence[Tuple[str, nn.Parameter]]) -> None:
    for _name, p in named_params:
        p.requires_grad_(True)


def make_train_collate(tokenizer):
    seq_keys = {"input_ids", "attention_mask", "labels", "token_type_ids", "position_ids"}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        keys = set()
        for item in batch:
            keys.update(item.keys())

        for k in keys:
            vals = [item[k] for item in batch if k in item]
            if not vals:
                continue
            if not isinstance(vals[0], torch.Tensor):
                continue

            if k in seq_keys and vals[0].dim() == 1:
                if k == "input_ids":
                    pv = pad_id
                elif k == "labels":
                    pv = -100
                else:
                    pv = 0
                out[k] = pad_sequence(vals, batch_first=True, padding_value=pv)
            else:
                out[k] = torch.stack(vals, dim=0)
        return out

    return _collate


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
    return out


def _fallback_causal_loss(outputs, labels: torch.Tensor) -> torch.Tensor:
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def make_privacy_engine(
    *,
    model: nn.Module,
    named_params: List[Tuple[str, nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    effective_batch_size: int,
    sample_size: int,
    epochs: int,
    target_epsilon: float,
    target_delta: float,
    args: argparse.Namespace,
) -> PrivacyEngine:
    pe = PrivacyEngine(
        model,
        batch_size=effective_batch_size,
        sample_size=sample_size,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=args.max_grad_norm,
        clipping_mode=args.clipping_mode,
        clipping_fn=args.clipping_fn,
        clipping_style=args.clipping_style,
        accounting_mode=ACCOUNTING_MODE,
        named_params=named_params,
    )
    pe.attach(optimizer)
    return pe


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
) -> Dict:
    if not named_params:
        raise RuntimeError(f"[{stage_name}] no trainable params selected")
    if len(train_ds) == 0:
        raise RuntimeError(f"[{stage_name}] empty training dataset")

    for _name, p in named_params:
        p.requires_grad_(True)
    params = [p for _name, p in named_params]
    n_trainable = sum(p.numel() for p in params)

    effective = min(args.train_batch_size, len(train_ds))
    micro = min(args.micro_batch_size, effective, len(train_ds))
    if dp:
        effective = max(micro, (effective // micro) * micro)
        accum = max(1, effective // micro)
    else:
        accum = max(1, math.ceil(effective / micro))

    optimizer = AdamW(params, lr=lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    loader = DataLoader(
        train_ds,
        batch_size=micro,
        shuffle=True,
        drop_last=dp,
        num_workers=0,
        collate_fn=make_train_collate(tokenizer),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(math.ceil(len(loader) / max(accum, 1)) * max(epochs, 1), 1),
        eta_min=1e-6,
    )

    pe = None
    if dp:
        pe = make_privacy_engine(
            model=model,
            named_params=named_params,
            optimizer=optimizer,
            effective_batch_size=effective,
            sample_size=len(train_ds),
            epochs=epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            args=args,
        )
        print(
            f"[{stage_name}] DP target eps={target_epsilon:.4f} delta={target_delta:.3e} "
            f"batch={effective} micro={micro}"
        )
    else:
        print(f"[{stage_name}] no-DP mode batch={effective} micro={micro}")

    best_loss = float("inf")
    best_state: Dict[str, torch.Tensor] = {}
    patience = 0
    t0 = time.time()
    print(f"[{stage_name}] trainable={n_trainable:,} lr={lr} epochs={epochs}")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_n = 0
        step_counter = 0

        iterator: Iterable = loader
        if tqdm and args.show_progress:
            iterator = tqdm(loader, desc=f"{stage_name} ep{epoch+1}/{epochs}", leave=False)

        for batch in iterator:
            batch = move_batch_to_device(batch, device)
            labels = batch.get("labels")
            prev_mask = getattr(model, "_active_attention_mask", None)
            if "attention_mask" in batch:
                model._active_attention_mask = batch["attention_mask"]
            try:
                outputs = model(**batch, return_dict=True)
            finally:
                model._active_attention_mask = prev_mask
            loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else _fallback_causal_loss(outputs, labels)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            step_counter += 1
            bs = int(batch["input_ids"].shape[0]) if "input_ids" in batch else micro
            epoch_loss_sum += float(loss.detach().item()) * bs
            epoch_n += bs

            if step_counter % accum == 0:
                if not dp:
                    nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                sanitize_parameters(params)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if step_counter % accum != 0 and step_counter > 0 and not dp:
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            sanitize_parameters(params)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss = epoch_loss_sum / max(epoch_n, 1)
        print(f"[{stage_name}] epoch={epoch+1}/{epochs} loss={epoch_loss:.4f} n={epoch_n}")

        if not dp:
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                    if any(k == n for n, _ in named_params)
                }
                patience = 0
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    print(f"[{stage_name}] early stop at epoch {epoch+1}")
                    break

    if not dp and best_state:
        model.load_state_dict(best_state, strict=False)

    epsilon = 0.0
    sigma = 0.0
    spent_python: Dict = {}
    if pe is not None:
        spent = pe.get_privacy_spent()
        spent_python = {k: tensor_to_python(v) for k, v in spent.items()}
        epsilon = extract_epsilon_rdp(spent_python)
        sigma = float(pe.noise_multiplier)
        pe.detach()
        print(f"[{stage_name}] actual eps={epsilon:.4f} sigma={sigma:.4f}")

    for _name, p in named_params:
        p.requires_grad_(False)

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
        "seconds": round(time.time() - t0, 1),
    }


@torch.no_grad()
def evaluate_generation(
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
) -> Dict:
    saved_record_layers = set(getattr(model, "_record_route_layer_ids", set()))
    forced_map = getattr(model, "_forced_record_expert_by_layer", None)
    forced_saved = dict(forced_map) if isinstance(forced_map, dict) else {}
    model._record_route_layer_ids = set()
    if isinstance(forced_map, dict):
        forced_map.clear()

    n = 0
    n_with_answer = 0
    em_sum = 0.0
    fout = open(pred_path, "w", encoding="utf-8") if pred_path else None
    try:
        for s in samples:
            if limit > 0 and n >= limit:
                break
            if not os.path.exists(s.image_path):
                continue
            image = Image.open(s.image_path).convert("RGB")
            pred = generate_answer(
                model=model,
                processor=processor,
                image=image,
                question=s.question,
                options=s.options,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            rec = {
                "image_path": s.image_path,
                "question": s.question,
                "prediction": pred,
                "answer": s.answer,
            }
            if s.answer is not None:
                rec["exact_match"] = exact_match(pred, s.answer)
                em_sum += float(rec["exact_match"])
                n_with_answer += 1
            if fout is not None:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    finally:
        model._record_route_layer_ids = saved_record_layers
        if isinstance(forced_map, dict):
            forced_map.clear()
            forced_map.update(forced_saved)
        if fout is not None:
            fout.close()
    out = {"n_samples": n}
    if n_with_answer > 0:
        out["exact_match"] = em_sum / n_with_answer
    return out


@torch.no_grad()
def construct_record_assignments(
    model: nn.Module,
    train_ds: Dataset,
    tokenizer,
    target_layer: SparseLayerRef,
    batch_size: int,
    device: torch.device,
) -> List[int]:
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_train_collate(tokenizer),
    )

    assignments = [-1 for _ in range(len(train_ds))]
    model.eval()
    idx_ptr = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        prev_mask = getattr(model, "_active_attention_mask", None)
        if "attention_mask" in batch:
            model._active_attention_mask = batch["attention_mask"]
        try:
            _ = model(**batch)
        finally:
            model._active_attention_mask = prev_mask

        raw_top1 = getattr(target_layer.sparse_mlp, "_last_raw_expert_index", None)
        if raw_top1 is None:
            raise RuntimeError(
                f"Sparse layer {target_layer.layer_id} did not expose _last_raw_expert_index. "
                f"Record-routing patch may be missing."
            )
        raw_top1 = raw_top1.detach().cpu()
        if "attention_mask" in batch and tuple(batch["attention_mask"].shape) == tuple(raw_top1.shape):
            valid = batch["attention_mask"].detach().cpu().bool()
        else:
            valid = torch.ones_like(raw_top1, dtype=torch.bool)
        n_experts = _num_experts(target_layer.sparse_mlp)

        raw_topk = getattr(target_layer.sparse_mlp, "_last_raw_topk_expert_indices", None)
        raw_weights = getattr(target_layer.sparse_mlp, "_last_raw_topk_routing_weights", None)
        if raw_topk is not None and raw_weights is not None:
            owners = _select_primary_record_owners(
                raw_topk.detach().cpu().reshape(raw_top1.size(0) * raw_top1.size(1), -1),
                raw_weights.detach().cpu().reshape(raw_top1.size(0) * raw_top1.size(1), -1),
                batch_size=raw_top1.size(0),
                seq_len=raw_top1.size(1),
                num_experts=n_experts,
                valid_mask=valid,
            )
            for i, chosen in enumerate(owners.tolist()):
                assignments[idx_ptr + i] = int(chosen)
        else:
            for i in range(raw_top1.size(0)):
                active = raw_top1[i][valid[i]]
                if active.numel() == 0:
                    chosen = 0
                else:
                    counts = torch.bincount(active.long(), minlength=n_experts)
                    chosen = int(counts.argmax().item())
                assignments[idx_ptr + i] = chosen
        idx_ptr += raw_top1.size(0)

    if any(x < 0 for x in assignments):
        raise RuntimeError(f"Incomplete record assignments for layer={target_layer.layer_id}")
    return assignments


def group_indices_by_expert(assignments: Sequence[int], num_experts: int) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {i: [] for i in range(num_experts)}
    for idx, expert_id in enumerate(assignments):
        out[int(expert_id)].append(idx)
    return out


def run_single_infer(args: argparse.Namespace, processor, model, device: torch.device) -> None:
    if not args.image_path or not args.prompt:
        raise ValueError("--mode infer requires both --image_path and --prompt")
    image = Image.open(args.image_path).convert("RGB")
    pred = generate_answer(
        model=model,
        processor=processor,
        image=image,
        question=args.prompt,
        options=None,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(pred)


def run_eval_mode(args: argparse.Namespace, processor, model, device: torch.device) -> None:
    if not args.dataset_jsonl:
        raise ValueError("--mode eval requires --dataset_jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    samples = load_jsonl_dataset(args.dataset_jsonl, image_root=args.image_root, require_answer=False)
    if not samples:
        raise RuntimeError(f"No valid samples loaded from {args.dataset_jsonl}")
    pred_path = os.path.join(args.output_dir, "predictions.jsonl") if args.save_predictions else ""
    metrics = evaluate_generation(
        model=model,
        processor=processor,
        samples=samples,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        limit=args.eval_max_samples,
        pred_path=pred_path,
    )
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def run_finetune_experiment(
    *,
    exp: str,
    model,
    processor,
    train_samples: Sequence[VQASample],
    dev_samples: Sequence[VQASample],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict:
    train_ds = VQATrainDataset(train_samples, processor, max_length=args.max_length)
    tokenizer = getattr(processor, "tokenizer", processor)
    sparse_layers = get_sparse_layers(model)

    model._record_route_layer_ids = set()
    model._forced_record_expert_by_layer = {}
    model._record_route_weight_mode = args.record_route_weight_mode
    model._active_attention_mask = None

    if sparse_layers:
        patch_sparse_mlp_for_record_routing(model, sparse_layers)

    zero = evaluate_generation(
        model=model,
        processor=processor,
        samples=dev_samples,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        limit=args.eval_max_samples,
        pred_path="",
    )
    print(f"[{exp}] zero-shot EM={zero.get('exact_match', 0.0):.4f} n={zero.get('n_samples', 0)}")

    result: Dict = {"experiment": exp, "zero_shot": zero}
    if exp in ("upper_bound", "naive_dp", "ours"):
        n_attn = apply_lora_with_filter(
            model,
            filter_fn=_is_attention_linear_name,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            use_expert_linear=False,
        )
        n_expert = apply_lora_with_filter(
            model,
            filter_fn=_is_expert_linear_name,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            use_expert_linear=(exp in ("naive_dp", "ours")),
        )
        print(f"[{exp}] LoRA injected: attention={n_attn} expert={n_expert}")
        if exp in ("naive_dp", "ours"):
            n_conv = convert_lora_expert_bases_to_expertlinear(model)
            print(f"[{exp}] converted expert LoRA base projections to ExpertLinear: {n_conv}")

    if exp == "upper_bound":
        freeze_all(model)
        named = select_all_lora_plus_head(model)
        enable_only(named)
        stats = run_train_loop(
            stage_name="UPPER_BOUND",
            model=model,
            train_ds=train_ds,
            tokenizer=tokenizer,
            device=device,
            args=args,
            named_params=named,
            lr=args.lr,
            epochs=args.finetune_epochs,
            dp=False,
            target_epsilon=0.0,
            target_delta=0.0,
        )
        dev = evaluate_generation(
            model=model,
            processor=processor,
            samples=dev_samples,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            limit=args.eval_max_samples,
            pred_path="",
        )
        stats.update({"dev_exact_match": dev.get("exact_match", 0.0), "dev_n": dev.get("n_samples", 0)})
        result["result"] = stats
        return result

    if exp == "naive_dp":
        if sparse_layers:
            patch_sparse_mlp_for_naive_dp(model, sparse_layers)
        freeze_all(model)
        named = select_all_lora_plus_head(model)
        enable_only(named)
        stats = run_train_loop(
            stage_name="NAIVE_DP",
            model=model,
            train_ds=train_ds,
            tokenizer=tokenizer,
            device=device,
            args=args,
            named_params=named,
            lr=args.lr,
            epochs=args.finetune_epochs,
            dp=not args.no_dp,
            target_epsilon=args.epsilon_total,
            target_delta=args.delta,
        )
        dev = evaluate_generation(
            model=model,
            processor=processor,
            samples=dev_samples,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            limit=args.eval_max_samples,
            pred_path="",
        )
        stats.update({"dev_exact_match": dev.get("exact_match", 0.0), "dev_n": dev.get("n_samples", 0)})
        result["result"] = stats
        return result

    if exp == "upper_bound_no_lora":
        freeze_all(model)
        named = select_full_params(model, freeze_vision=args.freeze_vision_in_full_finetune)
        enable_only(named)
        stats = run_train_loop(
            stage_name="UPPER_BOUND_NO_LORA",
            model=model,
            train_ds=train_ds,
            tokenizer=tokenizer,
            device=device,
            args=args,
            named_params=named,
            lr=args.lr_full_finetune,
            epochs=args.finetune_epochs,
            dp=False,
            target_epsilon=0.0,
            target_delta=0.0,
        )
        dev = evaluate_generation(
            model=model,
            processor=processor,
            samples=dev_samples,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            limit=args.eval_max_samples,
            pred_path="",
        )
        stats.update({"dev_exact_match": dev.get("exact_match", 0.0), "dev_n": dev.get("n_samples", 0)})
        result["result"] = stats
        return result

    if exp == "naive_dp_no_lora":
        if sparse_layers:
            n_replaced = apply_expert_linear_no_lora(sparse_layers)
            print(f"[naive_dp_no_lora] ExpertLinear converted layers: {n_replaced}")
            patch_sparse_mlp_for_naive_dp(model, sparse_layers)
        freeze_all(model)
        named = select_full_params(model, freeze_vision=args.freeze_vision_in_full_finetune)
        enable_only(named)
        stats = run_train_loop(
            stage_name="NAIVE_DP_NO_LORA",
            model=model,
            train_ds=train_ds,
            tokenizer=tokenizer,
            device=device,
            args=args,
            named_params=named,
            lr=args.lr_full_finetune,
            epochs=args.finetune_epochs,
            dp=not args.no_dp,
            target_epsilon=args.epsilon_total,
            target_delta=args.delta,
        )
        dev = evaluate_generation(
            model=model,
            processor=processor,
            samples=dev_samples,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            limit=args.eval_max_samples,
            pred_path="",
        )
        stats.update({"dev_exact_match": dev.get("exact_match", 0.0), "dev_n": dev.get("n_samples", 0)})
        result["result"] = stats
        return result

    if exp == "ours":
        if not sparse_layers:
            raise RuntimeError(
                "No sparse MoE layers were detected in this model. "
                "Record-routing per-expert DP strategy requires sparse experts."
            )

        phase_a_active = args.phase_a_epochs > 0
        if args.num_layers_to_train < 0:
            layers_b = sparse_layers[args.start_layer:]
        else:
            end = min(args.start_layer + args.num_layers_to_train, len(sparse_layers))
            layers_b = sparse_layers[args.start_layer:end]
        if not phase_a_active and not layers_b:
            raise RuntimeError("No trainable stages selected: phase A disabled and no phase B layers selected.")

        if args.no_dp:
            eps_a_target = 0.0
            eps_layer_target = 0.0
            delta_stage = 0.0
        else:
            if not (0.0 <= args.epsilon_attention < args.epsilon_total):
                raise ValueError("epsilon_attention must satisfy 0 <= epsilon_attention < epsilon_total")
            eps_a_target = args.epsilon_attention if phase_a_active else 0.0
            eps_b_total = args.epsilon_total - eps_a_target
            eps_layer_target = eps_b_total / len(layers_b) if layers_b else 0.0
            n_stages = (1 if phase_a_active else 0) + len(layers_b)
            delta_stage = args.delta / max(n_stages, 1)

        log: Dict = {"phase_a": None, "phase_b_layers": []}
        eps_b_actual = 0.0

        if phase_a_active:
            freeze_all(model)
            train_router = not args.freeze_router_in_phase_a
            print(f"[ours] Phase A train_router={train_router}  (experts frozen)")
            named_a = select_phase_a_lora(
                model,
                train_router=train_router,
                freeze_vision=args.freeze_vision_in_full_finetune,
            )
            enable_only(named_a)
            stats_a = run_train_loop(
                stage_name="OURS_PHASE_A",
                model=model,
                train_ds=train_ds,
                tokenizer=tokenizer,
                device=device,
                args=args,
                named_params=named_a,
                lr=args.lr_attention,
                epochs=args.phase_a_epochs,
                dp=not args.no_dp,
                target_epsilon=eps_a_target,
                target_delta=delta_stage,
            )
            dev_a = evaluate_generation(
                model=model,
                processor=processor,
                samples=dev_samples,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                limit=args.eval_max_samples,
                pred_path="",
            )
            stats_a.update({"dev_exact_match": dev_a.get("exact_match", 0.0), "dev_n": dev_a.get("n_samples", 0)})
            log["phase_a"] = stats_a

        for layer in layers_b:
            assignments = construct_record_assignments(
                model=model,
                train_ds=train_ds,
                tokenizer=tokenizer,
                target_layer=layer,
                batch_size=args.assignment_batch_size,
                device=device,
            )
            num_experts = _num_experts(layer.sparse_mlp)
            per_expert_indices = group_indices_by_expert(assignments, num_experts)

            if args.no_dp:
                expert_eps_target = 0.0
                expert_delta_target = 0.0
            else:
                if args.adjacency == "add_remove":
                    expert_eps_target = eps_layer_target
                    expert_delta_target = delta_stage
                else:
                    expert_eps_target = eps_layer_target / 2.0
                    expert_delta_target = delta_stage / 2.0

            expert_logs: List[Dict] = []
            expert_epsilons: List[float] = []
            model._record_route_layer_ids.add(layer.layer_id)
            try:
                for expert_id in range(num_experts):
                    indices = per_expert_indices[expert_id]
                    subset_size = len(indices)
                    if subset_size < args.min_expert_size:
                        expert_logs.append(
                            {
                                "layer_id": layer.layer_id,
                                "expert_id": expert_id,
                                "subset_size": subset_size,
                                "skipped": True,
                                "epsilon": 0.0,
                                "target_epsilon": expert_eps_target,
                                "target_delta": expert_delta_target,
                            }
                        )
                        continue

                    freeze_all(model)
                    freeze_routers(model)
                    named_b = select_expert_lora_params(model, layer, expert_id)
                    if not named_b:
                        expert_logs.append(
                            {
                                "layer_id": layer.layer_id,
                                "expert_id": expert_id,
                                "subset_size": subset_size,
                                "skipped": True,
                                "reason": "no_lora_params_for_expert",
                                "epsilon": 0.0,
                            }
                        )
                        continue
                    enable_only(named_b)
                    model._forced_record_expert_by_layer[layer.layer_id] = expert_id
                    try:
                        stats_e = run_train_loop(
                            stage_name=f"OURS_L{layer.layer_id}_E{expert_id}",
                            model=model,
                            train_ds=SubsetDataset(train_ds, indices),
                            tokenizer=tokenizer,
                            device=device,
                            args=args,
                            named_params=named_b,
                            lr=args.lr,
                            epochs=args.phase_b_epochs,
                            dp=not args.no_dp,
                            target_epsilon=expert_eps_target,
                            target_delta=expert_delta_target,
                        )
                    finally:
                        model._forced_record_expert_by_layer.pop(layer.layer_id, None)
                    stats_e.update(
                        {"layer_id": layer.layer_id, "expert_id": expert_id, "subset_size": subset_size}
                    )
                    expert_logs.append(stats_e)
                    expert_epsilons.append(float(stats_e.get("epsilon", 0.0)))
            finally:
                model._record_route_layer_ids.discard(layer.layer_id)

            max_expert_eps = max(expert_epsilons) if expert_epsilons else 0.0
            max_expert_delta = expert_delta_target if expert_logs else 0.0
            if args.adjacency == "add_remove":
                layer_eps_actual = max_expert_eps
                layer_delta_actual = max_expert_delta
            else:
                layer_eps_actual = 2.0 * max_expert_eps
                layer_delta_actual = 2.0 * max_expert_delta

            layer_log = {
                "layer_id": layer.layer_id,
                "module_name": layer.module_name,
                "target_layer_epsilon": eps_layer_target,
                "target_layer_delta": delta_stage,
                "expert_target_epsilon": expert_eps_target,
                "expert_target_delta": expert_delta_target,
                "max_expert_epsilon_actual": max_expert_eps,
                "layer_epsilon_actual": layer_eps_actual,
                "layer_delta_actual": layer_delta_actual,
                "experts": expert_logs,
            }
            if args.eval_each_layer:
                dev_l = evaluate_generation(
                    model=model,
                    processor=processor,
                    samples=dev_samples,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    limit=args.eval_max_samples,
                    pred_path="",
                )
                layer_log["dev_exact_match"] = dev_l.get("exact_match", 0.0)
                layer_log["dev_n"] = dev_l.get("n_samples", 0)

            log["phase_b_layers"].append(layer_log)
            eps_b_actual += layer_eps_actual

        dev_b = evaluate_generation(
            model=model,
            processor=processor,
            samples=dev_samples,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            limit=args.eval_max_samples,
            pred_path="",
        )
        eps_a_actual = float((log["phase_a"] or {}).get("epsilon", 0.0))
        result["result"] = {
            "phase_a": log["phase_a"],
            "phase_b_layers": log["phase_b_layers"],
            "privacy_summary": {
                "eps_A": eps_a_actual,
                "eps_B": eps_b_actual,
                "eps_total": eps_a_actual + eps_b_actual,
                "eps_target": args.epsilon_total,
                "adjacency": args.adjacency,
            },
            "dev_exact_match": dev_b.get("exact_match", 0.0),
            "dev_n": dev_b.get("n_samples", 0),
            "epsilon_total": eps_a_actual + eps_b_actual,
        }
        return result

    raise ValueError(f"Unknown experiment: {exp}")


def parse_args() -> argparse.Namespace:
    description = globals().get("CLI_DESCRIPTION", "DeepSeek-VL2 backend with OLMoE-style finetuning methods")
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model_name", default="deepseek-ai/deepseek-vl2-small")
    p.add_argument("--mode", choices=["infer", "eval", "finetune"], default="infer")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--image_path", default="")
    p.add_argument("--prompt", default="")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)

    p.add_argument("--dataset_jsonl", default="")
    p.add_argument("--train_jsonl", default="")
    p.add_argument("--dev_jsonl", default="")
    p.add_argument("--train_max_samples", type=int, default=-1)
    p.add_argument("--image_root", default="")
    p.add_argument("--output_dir", default="./deepseek_vl2_output")
    p.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_max_samples", type=int, default=-1)
    p.add_argument("--show_progress", action="store_true")

    p.add_argument(
        "--experiment",
        default="ours",
        choices=["upper_bound", "upper_bound_no_lora", "naive_dp", "naive_dp_no_lora", "ours", "all"],
    )
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--finetune_epochs", type=int, default=2)
    p.add_argument("--phase_a_epochs", type=int, default=1)
    p.add_argument("--phase_b_epochs", type=int, default=1)
    p.add_argument("--assignment_batch_size", type=int, default=8)
    p.add_argument("--start_layer", type=int, default=0)
    p.add_argument("--num_layers_to_train", type=int, default=-1)
    p.add_argument("--min_expert_size", type=int, default=1)
    p.add_argument("--eval_each_layer", action="store_true")
    p.add_argument("--record_route_weight_mode", default="one", choices=["one", "router_prob"])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_attention", type=float, default=2e-4)
    p.add_argument("--lr_full_finetune", type=float, default=2e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--early_stop_patience", type=int, default=3)

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--freeze_vision_in_full_finetune", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze_router_in_phase_a", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--no_dp", action="store_true")
    p.add_argument("--epsilon_total", type=float, default=8.0)
    p.add_argument("--epsilon_attention", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--adjacency", choices=["add_remove", "replace_one"], default="add_remove")
    p.add_argument("--clipping_mode", default="MixOpt", choices=["ghost", "MixGhostClip", "MixOpt"])
    p.add_argument("--clipping_fn", default="automatic", choices=["automatic", "Abadi", "global"])
    p.add_argument("--clipping_style", default="layer-wise", choices=["all-layer", "layer-wise", "param-wise"])
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "infer":
        processor, model = load_vlm(
            model_name=args.model_name,
            device=device,
            dtype_arg=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        run_single_infer(args, processor, model, device)
        return

    if args.mode == "eval":
        processor, model = load_vlm(
            model_name=args.model_name,
            device=device,
            dtype_arg=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        run_eval_mode(args, processor, model, device)
        return

    if not args.train_jsonl or not args.dev_jsonl:
        raise ValueError("--mode finetune requires --train_jsonl and --dev_jsonl")

    train_samples = load_jsonl_dataset(args.train_jsonl, image_root=args.image_root, require_answer=True)
    dev_samples = load_jsonl_dataset(args.dev_jsonl, image_root=args.image_root, require_answer=True)
    if args.train_max_samples > 0:
        train_samples = train_samples[: args.train_max_samples]
    if not train_samples:
        raise RuntimeError("No valid training samples found")
    if not dev_samples:
        raise RuntimeError("No valid dev samples found")

    exps = (
        ["upper_bound", "upper_bound_no_lora", "naive_dp", "naive_dp_no_lora", "ours"]
        if args.experiment == "all"
        else [args.experiment]
    )
    if args.delta < 0:
        args.delta = 1.0 / max(len(train_samples), 1)

    _patch_fastdp_use_full_backward_hook()
    register_expert_linear_fastdp()
    _patch_fastdp_for_expert_linear()
    all_results: Dict[str, Dict] = {}

    for exp in exps:
        print("\n" + "=" * 88)
        print(f"[EXPERIMENT] {exp}")
        print("=" * 88)

        processor_exp, model_exp = load_vlm(
            model_name=args.model_name,
            device=device,
            dtype_arg=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        result = run_finetune_experiment(
            exp=exp,
            model=model_exp,
            processor=processor_exp,
            train_samples=train_samples,
            dev_samples=dev_samples,
            args=args,
            device=device,
        )
        all_results[exp] = result

        exp_dir = os.path.join(args.output_dir, exp)
        os.makedirs(exp_dir, exist_ok=True)
        torch.save(model_exp.state_dict(), os.path.join(exp_dir, "model.pt"))
        with open(os.path.join(exp_dir, "log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        del model_exp
        del processor_exp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[DONE] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
