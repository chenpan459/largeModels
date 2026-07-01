# 附录 A：PyTorch 入门

### 主章节代码

- [code-part1.ipynb](code-part1.ipynb) / [code-part1_ch.ipynb](code-part1_ch.ipynb) 包含 A.1 至 A.8 节在书中出现的全部代码
- [code-part2.ipynb](code-part2.ipynb) / [code-part2_ch.ipynb](code-part2_ch.ipynb) 包含 A.9 节 GPU 相关代码（与书中一致）
- [DDP-script.py](DDP-script.py) 演示多 GPU 用法的脚本（Jupyter Notebook 仅支持单 GPU，因此以脚本而非 notebook 形式提供）。 运行方式：`python DDP-script.py`。若机器 GPU 超过 2 块，可使用 `CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py`。
- [exercise-solutions.ipynb](exercise-solutions.ipynb) / [exercise-solutions_ch.ipynb](exercise-solutions_ch.ipynb) 包含本章练习解答

### 可选代码

- [DDP-script-torchrun.py](DDP-script-torchrun.py) 是 `DDP-script.py` 的可选版本，通过 PyTorch 的 `torchrun` 命令运行，而非使用 `multiprocessing.spawn` 自行创建并管理多进程。`torchrun` 会自动处理分布式初始化（含多节点协调），配置更简单。 用法：`torchrun --nproc_per_node=2 DDP-script-torchrun.py`

## 中文文档

| 原文 | 中文版 |
|------|--------|
| [README.md](README.md) | [README_ch.md](README_ch.md) |
| 各子目录 `*.ipynb` | 对应 `*_ch.ipynb` |
