# 贡献指南

对应原文：[`docs/contributing.md`](../contributing.md)

欢迎各种贡献：修代码、答疑、改文档、传播项目、引用、点星。本指南大量参考 HuggingFace Transformers 的贡献说明。

## 可以做什么

- 修复现有代码问题
- 提交 bug 或功能需求 Issue
- 支持新组件（模型、数据集、collator 等）
- 实现新的遗忘方法
- 实现新的评测
- 改进文档

功能合入后，可把相关论文链到 [`docs/links.md`](../links.md)（中文目录：[文献与链接](./links.md)）。

## 修 bug

确认 Issue 区没有重复报告，且问题出在本库而非调用方代码。Issue 请尽量包含：

- 可复现的短代码
- 完整 traceback
- 硬件（GPU 数量与型号等）
- 对应的 Hydra 配置（太长可用链接或 Markdown 折叠）
- 其他截图等

## 新功能需求

请说明动机、尽可能细的描述、用法代码片段；若来自论文请附链接。

## 新组件

需要：定义类 → 注册 → 写配置。细节见 [组件扩展](./components.md)。包括 Trainer、Dataset、Metric、Benchmark、Model、Collator、Experiment。

尤其欢迎贡献**你自己提出的方法与基准**（你最清楚怎么用）。实现卡住可联系维护者加入 Discord 细聊。

## 贡献新的遗忘方法

1. 实现 Unlearning Trainer（自定义 loss 等），对接方式见组件文档中的 Trainer。
2. 若方法需要多步命令，写清楚 `run.sh`。
3. 在相关基准上跑通并调参；在 `community/methods/<YOUR_METHOD>/` 放 README（方法、超参、如何选 checkpoint）和可复现的 `run.sh`。
4. 把结果写入 [leaderboard](../../community/leaderboard.md)，把遗忘后模型传到 HuggingFace；论文链接写入 `docs/links.md`。

上传示例（用户名与仓库名自行替换）：

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create {benchmark}-{model}-{datasplit}-{method}
cd <CHECKPOINT_DIR>
git init
git remote add origin https://huggingface.co/<username>/{benchmark}-{model}-{datasplit}-{method}
git add .
git commit -m "Initial commit"
git push origin main
```

## 贡献新的遗忘评测 / 基准

评测仍是开放问题。新增 metric 见 [评测](./evaluation.md)；新数据集/模型见 [组件扩展](./components.md)。

新增基准的大致步骤：准备数据与微调/retain 模型 → 如需则实现新 Benchmark → 在基准上跑通并调基线方法 → 在 `community/benchmarks/<YOUR_BENCHMARK>/` 写清复现步骤，并更新 `docs/links.md`。

## 文档

欢迎指出错别字、缺失、含糊或过时内容。

## 提交 Pull Request

先搜现有 PR/Issue，避免重复劳动。不确定时可先开 Issue。

1. Fork 本仓库（原文误写成 transformers 的 Fork 链接；请 Fork **open-unlearning**）。
2. 克隆 fork，并添加上游：

```bash
git clone git@github.com:<your Github handle>/open-unlearning.git
cd open-unlearning
git remote add upstream https://github.com/locuslab/open-unlearning.git
```

3. 新建分支：`git checkout -b a-descriptive-name-for-my-changes`
4. 按快速开始装环境后：`pip install ".[dev]"`（ruff、pre-commit 等）。
5. 开发时保持格式：`make quality` 检查，`make style` 自动改。写清楚的 commit message。开 PR 前可 `git fetch upstream && git rebase upstream/main`，再 `git push -u origin <branch>`。
6. 在 GitHub 上开 PR，勾选下方清单。后续修改推送到同一分支即可出现在 PR 里。

### PR 清单

- 标题能概括改动
- 若对应 Issue，在描述里写清编号以便自动关联
- 未完成请用 `[WIP]` 前缀，避免重复劳动
- 现有测试与检查应通过
- 方法需有有用的 docstring
