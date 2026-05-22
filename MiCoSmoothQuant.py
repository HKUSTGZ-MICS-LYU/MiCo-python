import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

'''
SmoothQuant Port for MiCo Attentions
https://github.com/mit-han-lab/smoothquant/
'''

@dataclass
class SmoothQuantMapping:
    norm_name: str
    linear_names: Tuple[str, ...]
    scale_name: str
    reason: str


def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(_move_to_device(value, device) for value in obj)
    return obj


def default_forward_batch(model: nn.Module, batch, device):
    batch = _move_to_device(batch, device)
    if torch.is_tensor(batch):
        return model(batch)
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError("Empty calibration batch.")
        return model(batch[0])
    raise TypeError(f"Unsupported calibration batch type: {type(batch)!r}")


def collect_smoothquant_act_scales(
    model: nn.Module,
    calib_loader: Iterable,
    num_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
    forward_batch: Optional[Callable[[nn.Module, object, torch.device], object]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Collect per-input-channel activation max values for every nn.Linear module.

    The returned dict maps module names to tensors of shape [in_features]. This
    mirrors the core calibration used by SmoothQuant but is self-contained and
    works with this repository's local models.
    """
    was_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    if forward_batch is None:
        forward_batch = default_forward_batch

    act_scales: Dict[str, torch.Tensor] = {}
    hooks = []

    def stat_input(name, module, inputs):
        if not inputs:
            return
        x = inputs[0]
        if not torch.is_tensor(x) or x.numel() == 0:
            return
        if x.shape[-1] != module.in_features:
            return
        x_absmax = x.detach().reshape(-1, x.shape[-1]).abs().amax(dim=0).float().cpu()
        if name in act_scales:
            act_scales[name] = torch.maximum(act_scales[name], x_absmax)
        else:
            act_scales[name] = x_absmax

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_pre_hook(
                lambda m, inputs, name=name: stat_input(name, m, inputs)
            ))

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(calib_loader):
                if num_batches is not None and batch_idx >= num_batches:
                    break
                forward_batch(model, batch, device)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    return act_scales


@torch.no_grad()
def smooth_norm_linears(
    norm: nn.Module,
    linears: Sequence[nn.Linear],
    act_scales: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Fold SmoothQuant scales into a normalization module and one or more
    following linear layers.

    For x_norm = norm(x), Linear(x_norm) is preserved by applying:
      norm.weight /= scale, norm.bias /= scale, linear.weight *= scale
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not linears:
        raise ValueError("linears must contain at least one module")
    if not hasattr(norm, "weight") or norm.weight is None:
        raise ValueError(f"{norm.__class__.__name__} has no affine weight to absorb SmoothQuant scales")

    in_features = linears[0].in_features
    if norm.weight.numel() != in_features:
        raise ValueError(
            f"Norm/Linear shape mismatch: norm={norm.weight.numel()} linear.in_features={in_features}"
        )
    for linear in linears:
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(linear)!r}")
        if linear.in_features != in_features:
            raise ValueError("All smoothed linears must share the same input feature size")

    device = linears[0].weight.device
    dtype = linears[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype).clamp(min=eps)
    if act_scales.numel() != in_features:
        raise ValueError(
            f"act_scales size mismatch: got {act_scales.numel()}, expected {in_features}"
        )

    weight_scales = torch.stack(
        [linear.weight.detach().abs().amax(dim=0).to(device=device, dtype=dtype) for linear in linears],
        dim=0,
    ).amax(dim=0).clamp(min=eps)
    scales = (act_scales.pow(alpha) / weight_scales.pow(1.0 - alpha)).clamp(min=eps)

    norm.weight.div_(scales.to(device=norm.weight.device, dtype=norm.weight.dtype))
    if getattr(norm, "bias", None) is not None:
        norm.bias.div_(scales.to(device=norm.bias.device, dtype=norm.bias.dtype))
    for linear in linears:
        linear.weight.mul_(scales.to(device=linear.weight.device, dtype=linear.weight.dtype).view(1, -1))

    return scales.detach().cpu()


def _join_name(prefix: str, child: str) -> str:
    return child if prefix == "" else f"{prefix}.{child}"


def _has_named_modules(modules: Dict[str, nn.Module], names: Sequence[str]) -> bool:
    return all(name in modules for name in names)


def find_smoothquant_mappings(model: nn.Module) -> List[SmoothQuantMapping]:
    """
    Find safe norm -> linear groups for local pre-norm Transformer-style models.

    Supported automatic patterns:
      - models.LLaMa TransformerBlock
      - models.CCT TransformerEncoderLayer attention path
      - models.ViT TransformerEncoder
    """
    modules = dict(model.named_modules())
    mappings: List[SmoothQuantMapping] = []

    for prefix, module in model.named_modules():
        # LLaMa.TransformerBlock: attention_norm -> wq/wk/wv,
        # ffn_norm -> w1/w3. w2 is not directly fed by the norm output.
        llama_attn = [
            _join_name(prefix, "attention_norm"),
            _join_name(prefix, "attention.wq"),
            _join_name(prefix, "attention.wk"),
            _join_name(prefix, "attention.wv"),
        ]
        llama_ffn = [
            _join_name(prefix, "ffn_norm"),
            _join_name(prefix, "feed_forward.w1"),
            _join_name(prefix, "feed_forward.w3"),
        ]
        if _has_named_modules(modules, llama_attn):
            mappings.append(SmoothQuantMapping(
                norm_name=llama_attn[0],
                linear_names=tuple(llama_attn[1:]),
                scale_name=llama_attn[1],
                reason="llama_attention_qkv",
            ))
        if _has_named_modules(modules, llama_ffn):
            mappings.append(SmoothQuantMapping(
                norm_name=llama_ffn[0],
                linear_names=tuple(llama_ffn[1:]),
                scale_name=llama_ffn[1],
                reason="llama_ffn_gate_up",
            ))

        # CCT.TransformerEncoderLayer: pre_norm -> q/k/v. Do not smooth
        # norm1 -> linear1 because norm1 is post-norm and its output is also
        # the residual base in the FFN block.
        cct_attn = [
            _join_name(prefix, "pre_norm"),
            _join_name(prefix, "self_attn.q"),
            _join_name(prefix, "self_attn.k"),
            _join_name(prefix, "self_attn.v"),
        ]
        if _has_named_modules(modules, cct_attn):
            mappings.append(SmoothQuantMapping(
                norm_name=cct_attn[0],
                linear_names=tuple(cct_attn[1:]),
                scale_name=cct_attn[1],
                reason="cct_attention_qkv",
            ))

        # ViT.TransformerEncoder: la1 -> q/k/v, la2 -> first MLP linear.
        vit_attn = [
            _join_name(prefix, "la1"),
            _join_name(prefix, "msa.q"),
            _join_name(prefix, "msa.k"),
            _join_name(prefix, "msa.v"),
        ]
        vit_ffn = [
            _join_name(prefix, "la2"),
            _join_name(prefix, "mlp.0"),
        ]
        if _has_named_modules(modules, vit_attn):
            mappings.append(SmoothQuantMapping(
                norm_name=vit_attn[0],
                linear_names=tuple(vit_attn[1:]),
                scale_name=vit_attn[1],
                reason="vit_attention_qkv",
            ))
        if _has_named_modules(modules, vit_ffn):
            mappings.append(SmoothQuantMapping(
                norm_name=vit_ffn[0],
                linear_names=(vit_ffn[1],),
                scale_name=vit_ffn[1],
                reason="vit_ffn_linear1",
            ))

    deduped = []
    seen = set()
    for mapping in mappings:
        key = (mapping.norm_name, mapping.linear_names, mapping.scale_name)
        if key not in seen:
            deduped.append(mapping)
            seen.add(key)
    return deduped


@torch.no_grad()
def apply_smoothquant(
    model: nn.Module,
    act_scales: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    mappings: Optional[Sequence[SmoothQuantMapping]] = None,
    strict: bool = False,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Apply SmoothQuant smoothing to supported norm -> linear groups.

    This should run before MiCo set_qscheme()/PTQ layer replacement.
    Returns the per-group smoothing scales keyed by scale_name.
    """
    modules = dict(model.named_modules())
    if mappings is None:
        mappings = find_smoothquant_mappings(model)

    applied: Dict[str, torch.Tensor] = {}
    for mapping in mappings:
        missing = [
            name for name in (mapping.norm_name, *mapping.linear_names, mapping.scale_name)
            if name not in modules and name not in act_scales
        ]
        if missing:
            if strict:
                raise KeyError(f"Missing SmoothQuant modules/scales for {mapping}: {missing}")
            continue
        if mapping.norm_name not in modules or any(name not in modules for name in mapping.linear_names):
            if strict:
                raise KeyError(f"Missing module for SmoothQuant mapping: {mapping}")
            continue
        if mapping.scale_name not in act_scales:
            if strict:
                raise KeyError(f"Missing activation scale for {mapping.scale_name}")
            continue

        norm = modules[mapping.norm_name]
        linears = [modules[name] for name in mapping.linear_names]
        try:
            applied[mapping.scale_name] = smooth_norm_linears(
                norm=norm,
                linears=linears,
                act_scales=act_scales[mapping.scale_name],
                alpha=alpha,
            )
        except (TypeError, ValueError) as exc:
            if strict:
                raise
            if verbose:
                print(f"[SmoothQuant] skipped {mapping.scale_name}: {exc}")
            continue
        if verbose:
            print(
                f"[SmoothQuant] {mapping.reason}: {mapping.norm_name} -> "
                f"{', '.join(mapping.linear_names)}"
            )

    return applied


def save_act_scales(act_scales: Dict[str, torch.Tensor], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({key: value.cpu() for key, value in act_scales.items()}, path)


def load_act_scales(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")
