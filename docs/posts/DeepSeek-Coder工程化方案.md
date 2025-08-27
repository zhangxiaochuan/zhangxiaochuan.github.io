---
title: DeepSeek-Coder工程化方案
tags:
  - LLM
  - DeepSeek
---

{{ render_tags() }}


## 引言

这是一篇把概念、方法与实验放在同一条叙事线上、面向工程实践的梳理文。主轴是：**为什么做仓库级数据工程 → 怎么做 → 训练目标如何选型与落地**。


## 一、数据工程：让模型“理解项目”，而不只是“见过文件”

### 语料构成与入口筛选：先把大噪声挡在门外

* 训练集由 87% 源代码、10% 英文代码相关文本、3% 与代码无关的中文文本组成其中中文部分是“高质量文章，目的是提升模型的中文理解能力”；英文主要来自 GitHub Markdown 与 StackExchange，以补足库用法、调试语境。
* 抓取范围限定为 2023 年 2 月之前的公开仓库，并**先做规则过滤**，直接把原始体量压缩到 32.8%。这些规则复用了 StarCoder 的做法：限制平均行长与最大行长、字母占比低于 25% 的文件剔除；前 100 字符含 `<?xml version=` 视作 XML 并除 XSLT 外过滤；HTML 要求可见文本比例至少 20% 且不短于 100 字符；JSON 与 YAML 仅保留 50 到 5000 字符的文件等。

**为什么这样做**：StarCoder 的这套启发式过滤能以极低代价砍掉日志、数据文件与模板类噪声，为后续的依赖解析与去重节省大量算力和时间。

### 依赖解析与仓库级排序：把“因果结构”编码进序列

多数开源代码模型按“文件级”拼接训练样本，忽略了项目内部的跨文件依赖。DeepSeek-Coder 显式解析 import using include 等调用关系，**对仓库内文件做拓扑式排序**，保证被依赖文件先于引用者出现，并把文件路径以注释形式保留，从而把“项目的结构语境”喂给模型。

以下代码片段展示了这种依赖解析后的代码片段
```python
# path: src/core/engine.py
def run(x):
    print("result:", x)

# path: src/utils/math.py
import core.engine
def add(a, b):
    return a + b

# path: src/main.py
import utils.math
from core.engine import run
def main():
    x = utils.math.add(2, 3)
    run(x)
```

### 仓库级 near-dedup：去重的“最小原子”应该是仓库

近重复去重不是按“文件”而是按“仓库级串联样本”做，避免误删关键文件导致项目结构断裂。这是跨文件补全场景里非常关键的一步。

### 质量筛选与评测去污染：自证清白

除规则外，辅以编译器与质量模型加启发式规则，剔除语法错误与可读性差代码。为防评测泄漏，对 HumanEval、MBPP、GSM8K、MATH 等做 n-gram 过滤：出现与测试集完全相同的 10-gram 直接剔除，长度 3 到 9 的用精确匹配。

> 小结：这条**仓库级数据流水线**，把真实工程的结构信息显式注入预训练语料，使得模型的“知识载体”不再是散碎文件，而是“有依赖、有顺序的项目”。这为后文的跨文件实验提升提供了可解释的因果链条。



## 二、训练方法：FIM 如何与常规自回归目标共存

### 训练目标与采样策略

#### 目标组合：CausalLM + FIM

* **主目标**：常规自回归下一 token 预测（CausalLM）。
* **辅目标**：FIM（Fill-in-the-Middle），令模型在已知前缀与后缀时，预测中间片段 $m$，即最大化 $P(m\mid p,s)$。DeepSeek-Coder 在**预训练阶段**就混入 FIM，以对齐真实的“插入式补全”使用场景。

#### FIM 模式选择与比例

* 论文在 1.3B 模型上做了 0%、50%、100% 的 FIM 率和 50% MSP 的对比：
  **100% FIM** 虽然单项 FIM 指标最好，但**常规补全**能力最弱；**50% PSM** 在两者间更平衡，**优于 50% MSP**。最终将 **PSM 50%** 作为默认训练策略。
* PSM 与 SPM 是 FIM 的两种数据重排方式（Prefix-Suffix-Middle / Suffix-Prefix-Middle）。OpenAI 的系统研究也证明了通过数据重排即可在自回归框架内稳定学到 infill 能力。


### FIM 的工程实现细节

#### 哨兵 token 与样本构造

* DeepSeek-Coder为 FIM **引入三枚专用哨兵 token**，并在**文档级**完成 FIM 改写后再进行 pack：
  模板（PSM）：
  `<｜fim_start｜> prefix <｜fim_hole｜> suffix <｜fim_end｜> middle <|eos_token|>`
  这一实现直接来自论文正文示例。

#### 文档级 vs 上下文级

* 论文采用**文档级 FIM**（先对单文档切分，再参与 pack），遵循 Bavarian 等的做法；相比之下，OpenAI 也给出了**上下文级 FIM**（在长上下文切片后再对部分片段做 FIM）的实践建议。两者本质相同，侧重点不同：文档级实现简单，上下文级在极长序列下更鲁棒。


### 分词与特殊 token

* 分词：HuggingFace Tokenizers 训练的 **BPE**，**词表 32,000**。
* 特殊 token：在常规特符之外，加入 **<｜fim_start｜> / <｜fim_hole｜> / <｜fim_end｜>** 来标注三段位置，配合 $P(m\mid p,s)$ 的目标计算。


### 网络架构与效率优化

* **骨架**：Decoder-Only Transformer + **RoPE**。33B 采用 **GQA**（组大小 8），全系集成 **FlashAttention v2**。激活函数为 **SwiGLU**。这些选择在论文表 2 与方法节均有明确记载。
* **GQA 的动机与收益**：将 $h$ 个查询头分成 $G$ 组，共享更少的 $k,v$ 头（$G<h$），从而**显著降低 KV 缓存开销**并接近 MQA 的吞吐，同时保持接近 MHA 的质量。记 $h_k$ 为 $k,v$ 头数，则 KV 内存、带宽与读写成本相对 MHA 约按 $h/h_k$ 比例下降。
* **FlashAttention-2 的价值**：通过更优的并行与工作划分，在 A100 上训练端到端可达 **225 TFLOPs/s（约 72% 模型 FLOPs 利用）**，显著改善长序列场景的注意力算子效率。对于 16K 上下文，这是性价比较高的“刚性加速”。


### 优化器与学习率日程

* **优化器**：AdamW，$\beta_1=0.9,\ \beta_2=0.95$。
* **三阶段 LR 日程**：含 **2000** 步 warmup；每一阶段 LR 按 $\sqrt{1/10}$ 递减；**最终 LR 为初始值的 10%**。这套“强收敛”策略有助于在长序列与混合目标下保持稳定。



### 并行策略与训练环境

* **并行组合**：Tensor Parallel + **ZeRO** 数据并行 + **PipeDream** 流水并行。
* **硬件**：A100 与 H800 集群，单机 8 卡，节点内 **NVLink/NVSwitch**，跨节点 **InfiniBand**。

> 这类组合是当下大多数 Decoder-Only 预训练的“安全默认”：显存友好、扩展平滑、生态完善。


### 长上下文适配与 RoPE 缩放

将 RoPE 缩放因子从 1 提至 4，基频从 10000 调至 100000，并以 **batch size 512、序列 16K、额外 1000 步**做长序列稳定训练；理论上可达 **64K**，但**最可靠的实用区间是 16K**。



## 附录

### FIM 是什么

把一段序列切成三段：前缀$p$、中间$m$、后缀$s$。把原文重排为一种“可自回归学习”的顺序，让模型在标准左到右损失下，等价于最大化$P(m\mid p,s)$。常见两种编排：

* **PSM**：先放$p$和$s$，最后放$m$
* **SPM**：先放$s$再放$p$，最后放$m$
  OpenAI 的研究系统比较了 PSM 与 SPM，并给出文档级与上下文级两种构造方式；要点是仅通过**数据重排**就能学会 infill，无需改网络结构。

DeepSeek-Coder 在预训练时混入 FIM，并做消融后选定“**PSM，采样率约 50%**”，在代码补全与 FIM 能力之间取得平衡。

### 训练时如何“喂数据”

以代码模型常用的哨兵 token 为例（StarCoder 模型卡中展示了实际写法）：
`<fim_prefix>` 表示前缀位置，`<fim_suffix>` 表示后缀位置，`<fim_middle>` 表示“从这里开始预测中间段”。训练时把文本重排并直接用**普通自回归损失**。([Hugging Face][3])

损失形式可写成
$L = -\sum_{t\in m}\log P(x_t \mid p,s,x_{<t})$，在实现上等价为对“重排后的整段序列”做标准 NLL。


### 示例 1：Python 代码，一行被挖空

原文片段

```
def area(r):
    pi = 3.14159
    return pi * r * r
```

一次随机切分

* p 前缀

```
def area(r):
    pi = 3.14159
```

* m 中间

```
    return pi * r * r
```

* s 后缀

```
```

（此例 s 为空行，完全没问题）

#### PSM 训练样本

把 p、s 放前面，m 放后面，并加哨兵：

```
<fim_prefix>def area(r):
    pi = 3.14159
<fim_suffix>

<fim_middle>    return pi * r * r
```

此时模型在看到 `<fim_middle>` 后，按普通左到右目标去预测“return pi \* r \* r”的每个 token，等价于学习 $P(m\mid p,s)$。

#### SPM 训练样本

把 s、p 放前面，m 放后面：

```
<fim_prefix>

<fim_suffix>def area(r):
    pi = 3.14159
<fim_middle>    return pi * r * r
```

SPM 的好处之一是推理时更利于缓存复用，但训练法仍是相同的“重排加自回归”。


### 示例 2：多行插入的典型编辑场景

原文片段

```
def load_config(path):
    data = json.load(open(path))
    return data
```

切分得到

* p

```
def load_config(path):
```

* m

```
    if not os.path.exists(path):
        raise FileNotFoundError(path)
```

* s

```
    data = json.load(open(path))
    return data
```

#### PSM 训练样本

```
<fim_prefix>def load_config(path):
<fim_suffix>    data = json.load(open(path))
    return data
<fim_middle>    if not os.path.exists(path):
        raise FileNotFoundError(path)
```

训练时，损失主要落在 `<fim_middle>` 之后的 m 上；但实现层面通常对整段序列做标准 LM 损失，这正是 OpenAI FIM 的“仅靠数据重排即可学习”的关键。


#### SPM 训练样本


```
<fim_prefix>    data = json.load(open(path))
    return data
<fim_suffix>def load_config(path):
<fim_middle>    if not os.path.exists(path):
        raise FileNotFoundError(path)

```


### 推理时怎么用（与训练目标对齐）

推理时把已知的前缀与后缀放在 `<fim_prefix>...<fim_suffix>...<fim_middle>` 模板里，模型续写“中间”。例如 StarCoder 的模型卡给出的最小调用示例：
`<fim_prefix>def print_hello_world():\n<fim_suffix>\n    print('Hello world!')<fim_middle>`，模型将生成缺失的中间部分。
