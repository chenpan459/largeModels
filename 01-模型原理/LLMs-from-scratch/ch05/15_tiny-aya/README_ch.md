# 从零实现 Tiny Aya 3.35B

Tiny Aya 是 Cohere 推出的一款新型「小型」LLM，据称是 3B 参数量级上「能力最强的多语言开源权重模型」。（根据[发布公告](https://cohere.com/blog/cohere-labs-tiny-aya)，Tiny Aya 在多项指标上优于 Qwen3-4B、Gemma 3 4B 和 Ministral 3 3B。）

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/tiny-aya/01.webp">



这是一款非常适合在本地运行和实验的模型。唯一需要注意的是：虽然它是开源权重模型，但许可条款相对严格，仅允许非商业用途。

除此之外，Arya 是一款 3.35B 参数模型，提供多种变体，适用于个人及（非商业）研究用途：

  - [tiny-aya-base](https://huggingface.co/CohereLabs/tiny-aya-base)（基座模型）
  - [tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global)（各语言与地区最佳平衡；notebook 默认）
  - [tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire)（针对南亚语言优化）
  - [tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water)（针对欧洲与亚太语言优化）
  - [tiny-aya-earth](https://huggingface.co/CohereLabs/tiny-aya-earth)（针对西亚与非洲语言优化）



更具体地说，以下是各模型针对的语言列表：

| 地区           | 语言                                                         | 优化模型        |
| -------------- | ------------------------------------------------------------ | --------------- |
| **亚太**       | 繁体中文、粤语、越南语、他加禄语、爪哇语、高棉语、泰语、缅甸语、马来语、韩语、老挝语、印尼语、简体中文、日语 | tiny-aya-water  |
| **非洲**       | 祖鲁语、阿姆哈拉语、豪萨语、伊博语、斯瓦希里语、科萨语、沃洛夫语、绍纳语、约鲁巴语、尼日利亚皮钦语、马达加斯加语 | tiny-aya-earth  |
| **南亚**       | 泰卢固语、马拉地语、孟加拉语、泰米尔语、印地语、旁遮普语、古吉拉特语、乌尔都语、尼泊尔语 | tiny-aya-fire   |
| **欧洲**       | 加泰罗尼亚语、加利西亚语、荷兰语、丹麦语、芬兰语、捷克语、葡萄牙语、法语、立陶宛语、斯洛伐克语、巴斯克语、英语、瑞典语、波兰语、西班牙语、斯洛文尼亚语、乌克兰语、希腊语、书面挪威语、罗马尼亚语、塞尔维亚语、德语、意大利语、俄语、爱尔兰语、匈牙利语、保加利亚语、克罗地亚语、爱沙尼亚语、拉脱维亚语、威尔士语 | tiny-aya-water  |
| **西亚**       | 阿拉伯语、马耳他语、土耳其语、希伯来语、波斯语               | tiny-aya-earth  |


从架构上看，Tiny Aya 是经典的 decoder-only Transformer，并有几处值得注意的改动（除 SwiGLU、分组查询注意力等常见设计外）：

1. **并行 Transformer 块。** 并行 Transformer 块从同一归一化输入同时计算注意力与 MLP，再一步将两者加到残差上。我推测这是为了减少层内串行依赖，以提升计算吞吐。

2. **滑动窗口注意力。** 具体采用与 Arcee Trinity、Olmo 3 类似的 3:1 局部:全局比例，窗口大小为 4096。与 Arcee 类似，滑动窗口层使用 RoPE，全注意力层使用 NoPE。

3. **LayerNorm。** 多数架构已转向 RMSNorm，因其计算略省且表现良好。Tiny Aya 仍采用更经典的 LayerNorm 变体（实现上类似标准 LayerNorm，但无 shift/bias 参数）。



&nbsp;
## 文件

[standalone-tiny-aya.ipynb](standalone-tiny-aya.ipynb) / [standalone-tiny-aya_ch.ipynb](standalone-tiny-aya_ch.ipynb) 是实现 Tiny Aya 架构并加载预训练权重的独立 Jupyter notebook。


另一个 [standalone-tiny-aya-plus-kv-cache.ipynb](standalone-tiny-aya-plus-kv-cache.ipynb) / [standalone-tiny-aya-plus-kv-cache_ch.ipynb](standalone-tiny-aya-plus-kv-cache_ch.ipynb) notebook 增加了 KV cache 以提升运行时性能（但代码复杂度更高）。如需了解 KV cache，请参阅我的文章 [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。


<br>

如需了解架构差异并阅读与其他架构的对比，请参阅我的文章 [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)。



