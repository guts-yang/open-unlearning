import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs


def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_specdiff_statistics(
    draft_logits,
    target_logits,
    labels,
    chunk_size=8192,
    compute_kl=True,
):
    """Return per-sample overlap and KL(q || draft) on answer prediction tokens.

    Softmax normalization is over the full vocabulary. Vocabulary chunks only
    bound the temporary probability tensors; all arithmetic used by the
    overlap integral is fp32. Gradients flow through ``target_logits`` only.
    """
    if draft_logits.shape != target_logits.shape:
        raise ValueError(
            "Draft/target logits differ in shape: "
            f"{tuple(draft_logits.shape)} vs {tuple(target_logits.shape)}"
        )
    if draft_logits.ndim != 3:
        raise ValueError("SpecDiff logits must have shape [batch, sequence, vocab]")
    if labels.shape != draft_logits.shape[:2]:
        raise ValueError(
            f"Labels shape {tuple(labels.shape)} does not match logits "
            f"{tuple(draft_logits.shape[:2])}"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    draft = draft_logits[..., :-1, :].detach()
    target = target_logits[..., :-1, :]
    mask = labels[..., 1:].ne(-100)
    token_counts = mask.sum(dim=-1)
    if bool((token_counts == 0).any()):
        raise ValueError("Every SpecDiff sample must contain an answer token")

    draft_lse = torch.logsumexp(draft.float(), dim=-1, keepdim=True)
    target_lse = torch.logsumexp(target.float(), dim=-1, keepdim=True)
    token_overlap = torch.zeros_like(target_lse[..., 0])
    token_kl = torch.zeros_like(token_overlap) if compute_kl else None

    vocab_size = target.shape[-1]
    for start in range(0, vocab_size, chunk_size):
        stop = min(start + chunk_size, vocab_size)
        draft_log_prob = draft[..., start:stop].float() - draft_lse
        target_log_prob = target[..., start:stop].float() - target_lse
        draft_prob = draft_log_prob.exp()
        target_prob = target_log_prob.exp()
        token_overlap = token_overlap + torch.minimum(
            draft_prob, target_prob
        ).sum(dim=-1)
        if compute_kl:
            token_kl = token_kl + (
                target_prob * (target_log_prob - draft_log_prob)
            ).sum(dim=-1)

    mask_float = mask.to(dtype=token_overlap.dtype)
    sample_overlap = (token_overlap * mask_float).sum(dim=-1) / token_counts
    sample_overlap = sample_overlap.clamp(0.0, 1.0)
    if not compute_kl:
        return sample_overlap, None

    sample_kl = (token_kl * mask_float).sum(dim=-1) / token_counts
    return sample_overlap, sample_kl


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_undial_loss(model, ref_model, inputs, beta):
    # Forward pass on the student (trainable) model
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Build the mask that identifies the tokens need to be unlearned
    mask = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
    mask[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0

    # Adjust teacher logits: subtract di_strength on the correct token
    pre_softmax = shift_teacher_logits - mask * beta
    soft_label = F.softmax(pre_softmax, dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )
    return loss.mean(), outputs


def compute_wga_loss(model, inputs, beta):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_ce = ((-lm_loss).exp().detach()) ** beta
    forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100].mean()
    return forget_loss, outputs


def compute_satimp_loss(model, inputs, beta1, beta2):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_sat = ((-lm_loss).exp().detach()) ** beta1
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    forget_loss = -((weight_sat * weight_imp) * lm_loss)[
        shift_labels.view(-1) != -100
    ].mean()
    return forget_loss, outputs
