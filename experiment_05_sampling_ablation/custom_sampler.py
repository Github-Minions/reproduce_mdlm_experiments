"""
自定义 Token Selection 和 Unmask 策略
======================================

用法：在官方 MDLM 代码的采样循环中替换对应的策略函数。

示例（在 diffusion.py 的 ddpm_cache 采样中）：
    from custom_sampler import select_token, unmask_positions
    
    # Token selection
    new_tokens = select_token(logits, strategy="topk", k=5, temperature=1.0)
    
    # Unmask
    updated = unmask_positions(new_tokens, old_tokens, mask_id, 
                               strategy="confidence_first", logits=logits)
"""

import torch
import torch.nn.functional as F


def select_token_greedy(logits):
    """Greedy: argmax"""
    return logits.argmax(dim=-1)


def select_token_categorical(logits, temperature=1.0):
    """Categorical sampling with temperature."""
    probs = F.softmax(logits / temperature, dim=-1)
    B, L, V = probs.shape
    sampled = torch.multinomial(probs.view(-1, V), 1).view(B, L)
    return sampled


def select_token_topk(logits, k=5, temperature=1.0):
    """Top-K sampling."""
    topk_logits, topk_idx = torch.topk(logits, k, dim=-1)
    topk_probs = F.softmax(topk_logits / temperature, dim=-1)
    B, L, _ = topk_probs.shape
    sampled = torch.multinomial(topk_probs.view(-1, k), 1).view(B, L)
    return torch.gather(topk_idx, -1, sampled.unsqueeze(-1)).squeeze(-1)


def select_token(logits, strategy="greedy", **kwargs):
    """统一的 token selection 接口。"""
    if strategy == "greedy":
        return select_token_greedy(logits)
    elif strategy.startswith("categorical"):
        temp = kwargs.get("temperature", 1.0)
        return select_token_categorical(logits, temp)
    elif strategy.startswith("topk"):
        k = kwargs.get("k", 5)
        temp = kwargs.get("temperature", 1.0)
        return select_token_topk(logits, k, temp)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def unmask_native(new_tokens, old_tokens, mask_id, **kwargs):
    """Native: unmask all newly predicted tokens."""
    mask = (old_tokens == mask_id)
    result = old_tokens.clone()
    result[mask] = new_tokens[mask]
    return result


def unmask_random(new_tokens, old_tokens, mask_id, ratio=0.3, **kwargs):
    """Random: only unmask a random subset."""
    mask = (old_tokens == mask_id)
    rand_mask = torch.rand_like(new_tokens.float()) < ratio
    result = old_tokens.clone()
    update = mask & rand_mask
    result[update] = new_tokens[update]
    return result


def unmask_confidence_first(new_tokens, old_tokens, mask_id, logits=None, ratio=0.3, **kwargs):
    """Confidence-first: unmask most confident positions first."""
    if logits is None:
        return unmask_native(new_tokens, old_tokens, mask_id)
    probs = F.softmax(logits, dim=-1)
    conf = probs.max(dim=-1).values
    mask = (old_tokens == mask_id)
    masked_conf = conf * mask.float()
    n_mask = mask.sum().item()
    n_unmask = max(1, int(n_mask * ratio))
    _, top_idx = masked_conf.view(-1).topk(n_unmask)
    result = old_tokens.clone()
    result.view(-1)[top_idx] = new_tokens.view(-1)[top_idx]
    return result


def unmask_left_to_right(new_tokens, old_tokens, mask_id, **kwargs):
    """Left-to-right: unmask leftmost position only."""
    mask = (old_tokens == mask_id)
    result = old_tokens.clone()
    for b in range(old_tokens.size(0)):
        pos = mask[b].nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            result[b, pos[0]] = new_tokens[b, pos[0]]
    return result


def unmask_block(new_tokens, old_tokens, mask_id, **kwargs):
    """Block: unmask first contiguous block."""
    mask = (old_tokens == mask_id)
    result = old_tokens.clone()
    for b in range(old_tokens.size(0)):
        pos = mask[b].nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            end = 1
            while end < len(pos) and pos[end] == pos[end-1] + 1:
                end += 1
            for p in pos[:end]:
                result[b, p] = new_tokens[b, p]
    return result


def unmask_positions(new_tokens, old_tokens, mask_id, strategy="native", **kwargs):
    """统一的 unmask 接口。"""
    strategies = {
        "native": unmask_native,
        "random": unmask_random,
        "confidence_first": unmask_confidence_first,
        "left_to_right": unmask_left_to_right,
        "block": unmask_block,
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown unmask strategy: {strategy}")
    return strategies[strategy](new_tokens, old_tokens, mask_id, **kwargs)
