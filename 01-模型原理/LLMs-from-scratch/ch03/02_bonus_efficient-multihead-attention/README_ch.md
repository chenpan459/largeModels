# 更高效的多头注意力实现

- [mha-implementations.ipynb](mha-implementations.ipynb) / [mha-implementations_ch.ipynb](mha-implementations_ch.ipynb) 包含并比较多头注意力的不同实现



### 摘要

下图汇总性能基准测试结果（越低越好）。


&nbsp;
#### 仅前向传播

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/1_forward-only.webp?1" width="500px"></a>

&nbsp;
#### 前向与反向传播

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/2_forward-and-backward.webp?1" width="500px"></a>

&nbsp;
#### 编译后的前向与反向传播

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/3_forward-and-backward-compiled.webp?1" width="500px"></a>
