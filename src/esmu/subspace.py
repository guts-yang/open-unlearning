"""可计算充分子空间（Computable Sufficient Subspace, S）的线性代数接口。

对应方案（主方案_v3_ES-MU_2026-08-28）§7「核心命题」：S = span{∇_θ s_k}，
其中 s_k 为遗忘指标探针的标量输出；方法（§8）的扰动
Δ = ∇_θ ⟨G^{-1/2} c, s(θ)⟩ 被约束在 S 内，G 为探针梯度的 Gram 矩阵。

本模块仅依赖 torch，不 import evals / trainer，可独立单测。

🔴 硬约束（09 实现提示词【C】GOTCHA / 【E】A800 显存账）：
不同时物化 K 个 d 维梯度向量（1B fp32 单个 4.96 GB）。accumulate_gram
在 GPU 上逐条物化当前梯度（峰值 O(d)）+ K×K Gram（O(K²)）；精确
K×K Gram 的两两内积需要历史梯度，历史梯度转存 CPU（O(K·d)），
该 CPU 驻留与「仅单梯度」的显存账之间的张力见 accumulate_gram 的
TODO(DECISION-NEEDED #9)。
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("esmu.subspace")

__all__ = [
    "accumulate_gram",
    "effective_rank",
    "sample_direction",
    "whitening_sqrt_inv",
]

_IGNORE_INDEX = -100


def accumulate_gram(
    model,
    probe_batches,
    *,
    scalar_fn=None,
    param_filter=None,
    dtype=torch.float32,
    device=None,
) -> torch.Tensor:
    """逐探针前向+反传累计 Gram 矩阵 G_ij = ⟨∇s_i, ∇s_j⟩（方案 §7，K×K）。

    对每条探针独立执行前向求标量 s_k、反传取 ∇_θ s_k 并立即拼成单条
    d 维梯度，随后 detach().cpu() 转存；全部探针处理完后在 CPU 上计算
    G = J Jᵀ（J 为 K×d 梯度行矩阵）。返回前打印进程级峰值显存（CUDA 可用时）。

    🔴 内存语义（09 文档【E】显存账）：
    - GPU 峰值 O(d) + O(K²)：每条梯度在 GPU 上只驻留一次即转存，禁止
      同时物化 K 个 d 维 GPU 向量（1B fp32 单个 4.96 GB）。
    - CPU 侧 O(K·d)：G_ij = ⟨g_i, g_j⟩ 的每个非对角元素需要两条梯度向量
      同时在场，因此精确 K×K Gram 的流式累计在数学上必须保留历史梯度
      （K=96, 1B fp32 ≈ 476 GB CPU 内存）——该张力与「仅单梯度」的显存账
      冲突，实现方式待定，见 TODO(DECISION-NEEDED #9)。

    Args:
        model: 待探针的模型（nn.Module）。调用方负责置于目标设备，并
            按探针语义设置 ``model.train()`` / ``model.eval()``（见
            TODO(DECISION-NEEDED #8)）。
        probe_batches: 探针批次序列（先物化为 list，K = len(...)，即
            子空间维数）。每条为 dict，默认契约含 ``input_ids`` /
            ``attention_mask`` / ``labels``，与仓库
            ``DataCollatorForSupervisedDataset`` 产出一致。
        scalar_fn: 可调用 ``(model, batch) -> 0 维张量 s_k``。为 None 时
            使用默认逐样本 NLL 之和（仅占位，见 TODO(DECISION-NEEDED #1)）。
        param_filter: 可调用 ``(name: str, param: nn.Parameter) -> bool``，
            用于限定参与梯度的参数子集；None 表示全部 ``requires_grad``
            参数（见 TODO(DECISION-NEEDED #2)）。
        dtype: 梯度拼接与 Gram 的数值类型（默认 float32）。
        device: Gram 所在设备（默认取第一个被选参数的设备）。

    Returns:
        K×K 对称 Gram 矩阵（dtype × device），元素为内积
        G_ij = ⟨g_i, g_j⟩, g_k = vec(∇_θ s_k)。
    """
    probe_batches = list(probe_batches)
    K = len(probe_batches)
    if K == 0:
        raise ValueError("probe_batches is empty: need at least one probe")

    if scalar_fn is None:
        # TODO(DECISION-NEEDED #1): s_k 的正式定义未定（候选：单样本 NLL /
        # 逐样本 NLL 和 / TR 分量（perturbed+paraphrased 配对）/ 组合 reward），
        # 直接决定 K=96 还是 160。以下默认为逐样本 NLL 之和，仅为让接口可
        # 运行与单测，不构成方法设计决策。
        scalar_fn = _default_scalar_nll_sum

    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param_filter is not None and not param_filter(name, param):
            continue
        params.append(param)
    if not params:
        raise ValueError("No model parameters selected for gradient computation")

    device = device if device is not None else params[0].device

    # TODO(DECISION-NEEDED #9): 精确 K×K Gram 需要历史梯度（CPU O(K·d)）。
    # 候选：A) 两遍磁盘法（GPU 峰值 O(d)+O(K²)，pass1 存盘 / pass2 内积）；
    # B) 分块驻留（块内 GPU，块间结果累计）；C) bf16 压缩 J（K=96 ≈ 238 GB）；
    # D) 近似（Nyström / 随机投影，非精确）。当前实现为 A 的内存版（历史
    # 梯度驻留 CPU），仅保证 GPU 峰值 O(d)+O(K²) 与数学精确，CPU 内存不
    # 满足 1B/K=96 场景，需裁定后替换。
    jacobian_rows = []
    for k, batch in enumerate(probe_batches):
        s_k = scalar_fn(model, batch)
        if not isinstance(s_k, torch.Tensor) or s_k.ndim != 0:
            raise ValueError(
                "scalar_fn must return a 0-dim tensor, "
                f"got {type(s_k).__name__} with ndim={getattr(s_k, 'ndim', None)}"
            )
        # 单条梯度：GPU 上仅物化当前一条（硬约束，09 文档【E】），随即转存 CPU
        grads = torch.autograd.grad(s_k, params, allow_unused=True)
        g_flat = torch.cat(
            [
                (
                    g.reshape(-1)
                    if g is not None
                    else torch.zeros(p.numel(), dtype=dtype, device=p.device)
                )
                for g, p in zip(grads, params)
            ]
        )
        jacobian_rows.append(g_flat.detach().to(device="cpu", dtype=dtype))
        del grads, g_flat, s_k
        if k % 32 == 0:
            logger.debug("accumulate_gram: probe %d/%d accumulated", k + 1, K)

    # 精确 K×K Gram：G = J Jᵀ（CPU 计算，避免 GPU 上 O(K²·d) 算力与驻留）
    J = torch.stack(jacobian_rows)
    del jacobian_rows
    gram = J @ J.T
    del J
    gram = gram.to(device=device, dtype=dtype)

    _log_peak_memory()
    return gram


def effective_rank(G: torch.Tensor, eps: float) -> int:
    """有效秩 K_eff：特征值 λ_i > eps·λ_max 的个数（方案 §7 / §10 G1′ 判据）。

    Args:
        G: K×K 对称 Gram 矩阵。
        eps: 相对阈值系数（0 < eps），阈值本身为 ``eps·λ_max``，与
            ``whitening_sqrt_inv`` 的 eps 语义一致。

    Returns:
        int，满足 0 ≤ K_eff ≤ K；λ_max ≤ 0（如全零矩阵）时返回 0。
    """
    _assert_square(G)
    if not (eps > 0):
        raise ValueError(f"eps must be > 0, got {eps}")
    evals = torch.linalg.eigvalsh(G)  # 升序
    lambda_max = evals[-1].item()
    if lambda_max <= 0:
        return 0
    return int((evals > eps * lambda_max).sum().item())


def whitening_sqrt_inv(G: torch.Tensor, eps: float) -> torch.Tensor:
    """返回 G^{-1/2}（方案 §8 白化步骤）。

    近奇异处理：对 G' = G + eps·I 做对称特征分解 G' = U diag(λ) Uᵀ，
    特征值下限 clamp 到 eps 防除零，返回 U diag(λ^{-1/2}) Uᵀ。

    Args:
        G: K×K 对称半正定 Gram 矩阵。
        eps: 正则强度与特征值下限（> 0）。

    Returns:
        K×K 对称矩阵，满足 G^{-1/2} G G^{-1/2} ≈ I（eps 较小时）。
    """
    _assert_square(G)
    if not (eps > 0):
        raise ValueError(f"eps must be > 0, got {eps}")
    device, dtype = G.device, G.dtype
    reg = G + eps * torch.eye(G.shape[0], dtype=dtype, device=device)
    evals, evecs = torch.linalg.eigh(reg)
    inv_sqrt_evals = torch.clamp(evals, min=eps).pow(-0.5)
    # U @ diag(λ^{-1/2}) @ Uᵀ：列缩放后右乘 Uᵀ
    return (evecs * inv_sqrt_evals.unsqueeze(0)) @ evecs.T


def sample_direction(G_inv_sqrt: torch.Tensor, rng=None) -> torch.Tensor:
    """子空间内随机方向系数 c（方案 §8：Δ = ∇_θ⟨G^{-1/2} c, s(θ)⟩ ∈ S）。

    返回 c = G^{-1/2} z, z ~ N(0, I_K)。调用方须现场用 Jacobian 反传
    Δ = Σ_k c_k·∇_θ s_k 得到实际扰动向量——严禁缓存 K 个基向量 ∇s_k
    （与 accumulate_gram 的显存硬约束一致）。

    Args:
        G_inv_sqrt: ``whitening_sqrt_inv`` 的输出（K×K）。
        rng: 可选 ``torch.Generator``；None 时使用无种子随机源（不可复现）。

    Returns:
        K 维系数向量 c（与 G_inv_sqrt 同 dtype / device）。
    """
    _assert_square(G_inv_sqrt)
    K = G_inv_sqrt.shape[0]
    z = torch.randn(K, dtype=G_inv_sqrt.dtype, device=G_inv_sqrt.device, generator=rng)
    return G_inv_sqrt @ z


def _default_scalar_nll_sum(model, batch) -> torch.Tensor:
    """默认探针标量：batch 内逐样本 NLL 之和（0 维，占位实现）。

    语义对应 ``src/trainer/utils.py::compute_batch_nll``（逐样本和后再
    reduce=sum），仅依赖 torch 以便本模块独立单测。

    TODO(DECISION-NEEDED #1): 正式 s_k 定义待定，此处仅为可运行占位。
    """
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    logits = outputs.logits[..., :-1, :].contiguous()
    labels = batch["labels"][..., 1:].contiguous()
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=_IGNORE_INDEX,
        reduction="sum",
    )


def _assert_square(G: torch.Tensor) -> None:
    """校验 G 为方阵，否则抛 ValueError。"""
    if not isinstance(G, torch.Tensor) or G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(
            f"G must be a square 2D tensor, got shape {tuple(getattr(G, 'shape', ())) if hasattr(G, 'shape') else G}"
        )


def _log_peak_memory() -> None:
    """打印进程级峰值显存（CUDA 可用时），供 A3 显存目标核查。

    目标：1B 模型 A800 80GB 峰值 < 20 GB（09 文档【E】理论账 ≈ 7.5 GB）。
    注：``torch.cuda.max_memory_allocated`` 为进程级峰值而非本次调用增量，
    数值仅作参考，正式验收在远程 2×A800 上记录。
    """
    if torch.cuda.is_available():
        logger.info(
            "esmu.subspace peak GPU memory: allocated=%.2f GB, reserved=%.2f GB"
            " (process-level, target < 20 GB on 1B/A800)",
            torch.cuda.max_memory_allocated() / 1e9,
            torch.cuda.max_memory_reserved() / 1e9,
        )
    else:
        logger.info("esmu.subspace: CUDA not available, peak-memory print skipped")
