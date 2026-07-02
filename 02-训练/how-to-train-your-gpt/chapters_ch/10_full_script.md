# 第 10 章 — 完整脚本 (main.py)

完整可运行的训练脚本位于仓库根目录的 `main.py`。

其中内联包含全部组件——分词器、模型、训练循环与推理——克隆仓库后一条命令即可运行：

```bash
python main.py
```

默认使用小模型（d_model=256，4 层，4 头），在 5,000 篇 Wikipedia 文章上训练 500 步。CPU 约 2–5 分钟，GPU 只需几秒。脚本中还包含注释掉的 GPT-2 Small 配置（768 维，12 层，12 头），有 GPU 时可启用。

训练结束后会保存 checkpoint 到 `checkpoints/model.pt`，绘制损失曲线，并对若干 prompt 生成示例文本。

---

**上一章：** [第 9 章 — 推理](09_inference.md)
**下一章：** [第 11 章 — 术语表](11_glossary.md)
