# SimNPO: A Simple and Effective Non-Preference Optimization for Machine Unlearning
- Paper: https://arxiv.org/html/2410.07163v4
- Code: https://github.com/OPTML-Group/Unlearn-Simple

SimNPO reformulates unlearning as a non-preference optimization: it maximizes the
log-sigmoid of the (mean-subtracted, margin-shifted) negative NLL on the forget set,
scaled by `2/beta`, and adds a retain-side loss to preserve utility:

```
forget_loss = -logsigmoid(beta * (nll/len - delta)).mean() * 2 / beta
loss = gamma * forget_loss + alpha * retain_loss
```

The default `gamma=0.125` (npo_coeff) heavily suppresses the forget signal, which on
TOFU `forget10` with `Llama-3.2-1B-Instruct` causes *under-forgetting* (FQ ≈ 2.78e-16,
forget Prob 0.5855 ≫ Retain 0.1161). Careful tuning of `beta`/`gamma` and the stopping
epoch is required to lift FQ.

## Setup

- [x] **Hyperparameters & Search Space:** grid over
  `beta ∈ {0.5, 1.0, 2.5, 4.5}`, `gamma ∈ {0.125, 0.5, 1.0, 2.0}`,
  `learning_rate ∈ {1e-5, 2e-5, 5e-5}`, `alpha ∈ {1.0, 2.0, 5.0}`,
  `delta ∈ {0.0, 0.05}`, `retain_loss_type ∈ {NLL, KL}`, plus the **stopping epoch**
  (1–10, read off the in-training FQ curve). Roughly 28 trials in 4 stages:
  trajectory smoke → coarse strength (beta×gamma) → lr sweep on top-3 →
  stop-point (single-GPU trajectory) → utility refinement (alpha/delta/KL).
- [x] **Computational Setup:** 2× GPU (DeepSpeed ZeRO-3 offload via
  `configs/accelerate/default_config.yaml`), effective batch size 32
  (2-GPU: `per_device=4 × accum=4 × 2`; 1-GPU trajectory: `per_device=8 × accum=4`).
  Full search ≈ 6h; endpoint-only trials ≈ 12min, single-GPU trajectory ≈ 41min.
- [x] **DeepSpeed Configuration:** unchanged from `configs/accelerate/default_config.yaml`
  (ZeRO-3 offload). No method-specific DeepSpeed changes.
- [x] **Other Details:** FQ curve is obtained at **zero weight cost** by running
  `python src/train.py` on a single GPU (`num_processes==1`) with
  `trainer.args.eval_strategy=epoch`; the custom evaluator writes per-epoch
  `checkpoint-*/evals/TOFU_SUMMARY.json`. `retain_logs_path` must be set or FQ is None.
  Each trial's weights are deleted right after eval; only `evals/` and configs remain.

## Results

Final tuned configuration (TOFU · `Llama-3.2-1B-Instruct` · `forget10`) and its 8 metrics
are recorded in `results/tofu_Llama-3.2-1B-Instruct.md` under the `SimNPO-tuned` row,
side by side with the untuned `SimNPO`. Target: FQ ≥ 0.05 (statistically forgotten at
α=0.05) while keeping `model_utility ≥ 0.55`.

| 配置 | FQ↑ | MU↑ | TR | forget Prob | forget ROUGE | privleak |
|------|-----|-----|----|-------------|--------------|----------|
| SimNPO (未调参) | 2.78e-16 | — | 0.5177 | 0.5855 | 0.4789 | -97.79 |
| SimNPO-tuned (目标) | ≥0.05 | ≥0.55 | ≈0.63 | ≈0.116 | ≈0.379 | ≈0 |

## Citation

```bibtex
@misc{zhang2024simnpo,
  title={SimNPO: A Simple and Effective Non-Preference Optimization for Machine Unlearning},
  author={Zhang, Zhexin and Liu, Junxiao and Lin, Bill Yuchen and Lu, Shuai and Wang, Bowen and others},
  year={2024},
  howpublished={\url{https://arxiv.org/html/2410.07163v4}}
}

@misc{openunlearning2025,
  title={OpenUnlearning: A Unified Framework for LLM Unlearning Benchmarks},
  author={Dorna, Vineeth and Mekala, Anmol and Zhao, Wenlong and McCallum, Andrew and Kolter, J Zico and Maini, Pratyush},
  year={2025},
  howpublished={\url{https://github.com/locuslab/open-unlearning}},
  note={Accessed: 2025}
}
```
