"""ES-MU 可计算充分子空间模块（方案 §7 / §8）。

当前（Stage 3）仅提供子空间 S = span{∇_θ s_k} 的线性代数接口：

- ``accumulate_gram``：逐探针前向+反传累计 K×K Gram 矩阵（峰值显存 O(d)+O(K²)）
- ``effective_rank``：按特征值谱取有效秩 K_eff（G1′ 判据之一）
- ``whitening_sqrt_inv``：返回 G^{-1/2}（白化算子）
- ``sample_direction``：返回子空间内随机方向系数 c

搜索（Stage 5）与 G1′ 闸门的探针保真度扫描不在本模块范围。
"""

from esmu.subspace import (
    accumulate_gram,
    effective_rank,
    sample_direction,
    whitening_sqrt_inv,
)

__all__ = [
    "accumulate_gram",
    "effective_rank",
    "sample_direction",
    "whitening_sqrt_inv",
]
