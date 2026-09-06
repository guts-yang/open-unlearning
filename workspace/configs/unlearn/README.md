Hydra 实验配置在 `workspace/configs/experiment/unlearn/`：

- `blade_tofu_{1b,3b}_{01,05,10}.yaml`
- `blade_muse_{news,books}.yaml`
- `blade_tofu01_smoke.yaml`
- `grom_tofu05.yaml` / `grom_tofu10.yaml` / `grom_muse_{news,books}.yaml` / `grom_wmdp_bio.yaml`
- `baldro_npo_{dv,g}_tofu01.yaml`
- `alter_wmdp_cyber.yaml`

Trainer overlay：`workspace/configs/trainer/`。
通过 `python workspace/scripts/train.py experiment=unlearn/<name>` 加载。
