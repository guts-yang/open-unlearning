"""Forward-only key collection. Pin Batorskq/GROM@6dd9591."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


@torch.no_grad()
def forward_collect(
    model,
    tok,
    ids_list: List[List[int]],
    lp_list: List[int],
    device: str,
    module=None,
    bs: int = 16,
    max_cols: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    cap = {}
    handle = None
    if module is not None:

        def hook(_m, inp):
            cap["x"] = inp[0]

        handle = module.register_forward_pre_hook(hook)

    feats = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(ids_list), bs):
        chunk = ids_list[i : i + bs]
        lps = lp_list[i : i + bs]
        width = max(len(x) for x in chunk)
        inp = torch.full((len(chunk), width), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, ids in enumerate(chunk):
            inp[j, : len(ids)] = torch.tensor(ids)
            att[j, : len(ids)] = 1
        out = model.model(input_ids=inp.to(device), attention_mask=att.to(device))
        src = cap["x"] if module is not None else out.last_hidden_state
        for j, ids in enumerate(chunk):
            pos = list(range(lps[j] - 1, len(ids) - 1)) or [len(ids) - 1]
            feats.append(src[j, pos, :].float().cpu())

    if handle is not None:
        handle.remove()

    K = torch.cat(feats, 0).T.contiguous()
    if max_cols and K.shape[1] > max_cols:
        idx = torch.randperm(K.shape[1])[:max_cols]
        return K[:, idx], idx
    return K, None
