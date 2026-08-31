# MOGP-U external seed provenance

External repositories are evidence sources only. MOGP-U does not import, vendor, or
copy their training code; each seed is independently reduced to the closed E/R/S DSL.
The executable catalog is [`configs/mogpu/seed_catalog.yaml`](../configs/mogpu/seed_catalog.yaml).
Search-discovered top-3 specs (open 8×8×4, F2=50, forget10; not FQ-feasible) live in [`configs/mogpu/discovered/`](../configs/mogpu/discovered/).

| Method | Status | Source revision | License | M-DSL evidence |
| --- | --- | --- | --- | --- |
| NPO/NPO+RT | enabled | `1207bbe9a1abf502bc2d7b43a0a467d9524b4036` | unlicensed | reference-calibrated E + R |
| SimNPO | enabled | `3083017cb317753725f35cc404c2d5aede1242ef` | MIT | smooth E + R/S |
| GA/LLMU | enabled | `647f309519f91c29d87e62cf63d9a43759810040` | MIT | direct E paired with R |
| WGA/TNPO/WTNPO | enabled | `ef368eea3b2c6dba1e090b9ebb021ac9f047e0ae` | unlicensed | frozen-manifest weighted E + R |
| SatImp | enabled | `51314c05abe393801fb8134b78ef402b0c95e4e2` | unlicensed | frozen-manifest weighted E + R |
| TPO | pending_proof | `54d2216f3fbb8e7c2ea9a60d0f21c1e0d2a0668b` | unlicensed | target mask and direction proof missing |
| LoKU | pending_proof | `272484aabc7ba970ed357dc9212fde4f4d62bb5a` | MIT | fixed mask and E-contract proof missing |
| ULD | pending_proof | `858608c960819a8002cc60cc3788bbbd57f4a913` | MIT | auxiliary-model composition is outside v2 |

The catalog contains the official repository URLs, exact source symbols, and paper
formula locations. `pending_proof`, `baseline_only`, and `excluded` records are never
passed to the GP initial population. Representation interventions, alternate-answer
objectives, dynamic optimizers, distillation, model editing, inference-time methods,
and synthetic-relearning data paradigms remain baselines rather than M-DSL atoms.

This provenance index is not a claim that any external method was fully reproduced.
