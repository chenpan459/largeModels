# 附录 A：PyTorch 入门

### 主章节代码

- [code-part1.ipynb](code-part1.ipynb) / [code-part1_ch.ipynb](code-part1_ch.ipynb) 包含章节 A.1 至 A.8 的全部代码
- [code-part2.ipynb](code-part2.ipynb) / [code-part2_ch.ipynb](code-part2_ch.ipynb) 包含章节 A.9 GPU 相关代码
- [DDP-script.py](DDP-script.py) 演示多 GPU 用法的脚本（Jupyter Notebook 仅支持单 GPU，因此为脚本而非 notebook）。可运行 `python DDP-script.py`；若机器有多于 2 块 GPU，可运行 `CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py`。
- [exercise-solutions.ipynb](exercise-solutions.ipynb) / [exercise-solutions_ch.ipynb](exercise-solutions_ch.ipynb) 包含本章练习解答

### 可选代码

- [DDP-script-torchrun.py](DDP-script-torchrun.py) 是 `DDP-script.py` 的可选版本，通过 PyTorch 的 `torchrun` 命令运行，而非用 `multiprocessing.spawn` 自行管理多进程。`torchrun` 可自动处理分布式初始化（含多节点协调），略微简化配置。用法：`torchrun --nproc_per_node=2 DDP-script-torchrun.py`
