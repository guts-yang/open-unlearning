"""Temporary debug script — DELETE AFTER USE."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(7, 2)
        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc2 = nn.Linear(3, 7, bias=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask, labels
        hidden = F.relu(self.fc1(self.embed(input_ids)))
        return SimpleNamespace(logits=self.fc2(hidden))


torch.manual_seed(0)
m = Toy()
input_ids = torch.randint(0, 7, (1, 4), generator=torch.Generator().manual_seed(0))
labels = input_ids.clone()
labels[:, 0] = -100
batch = {
    "input_ids": input_ids,
    "attention_mask": torch.ones_like(input_ids),
    "labels": labels,
}
b = {k: v.to(next(m.parameters()).device) for k, v in batch.items()}
out = m(**b)
logits = out.logits[..., :-1, :].contiguous()
lab = b["labels"][..., 1:].contiguous()
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    lab.reshape(-1),
    ignore_index=-100,
    reduction="sum",
)
names = [n for n, p in m.named_parameters() if p.requires_grad]
params = [p for n, p in m.named_parameters() if p.requires_grad]
print("loss:", loss, "ndim:", loss.ndim)
grads = torch.autograd.grad(loss, params, allow_unused=True)
for n, g in zip(names, grads):
    print(n, "None" if g is None else tuple(g.shape))
flat = torch.cat(
    [(g.reshape(-1) if g is not None else torch.zeros_like(p)).to(torch.float32) for g, p in zip(grads, params)]
)
print("flat shape:", tuple(flat.shape))
