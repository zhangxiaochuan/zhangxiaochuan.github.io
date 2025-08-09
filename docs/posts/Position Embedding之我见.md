---
title: Position Embedding之我见
tags:
  - 位置编码
  - LLM
---
{{ render_tags() }}


## 引言
Transformer 架构抛弃了 RNN 和 CNN 中天然带有顺序感的结构，因此为了让模型理解 token 之间的**位置信息**，必须显式地注入**位置编码（Positional Encoding）**。

最常见的编码方式，是《Attention is All You Need》中提出的 **正余弦位置编码（Sinusoidal Positional Encoding）**，其核心设计是：

* 不同**维度**使用不同**频率**；
* **偶数维**使用 $\sin$，**奇数维**使用 $\cos$。

这引发两个问题：

> 1. 为什么要区分奇偶维度（sin / cos）？
> 2. 为什么要为不同维度分配不同的频率？

我将在本文中深入推导和剖析，并谈下我的看法。

---

## 正余弦位置编码的形式

设模型维度为 $d_{model}$，位置为 $pos$，第 $i$ 个维度的编码为：

* 偶数维度（$2i$）：

  $$
  PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
  $$

* 奇数维度（$2i+1$）：

  $$
  PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
  $$

其中 $\lambda_i = 10000^{2i/d_{model}}$ 控制频率。

---

## 为什么要使用不同的频率？

###  **构建多尺度的位置信息表达**

每个维度的频率不同，相当于在频域上观察序列：

* 低频（大波长）维度感知**长距离依赖**；
* 高频（小波长）维度感知**局部变化**。

这种方式可以像小波或傅里叶变换一样，从多尺度上对序列位置进行建模。

###  **避免维度间冗余**

如果所有维度使用相同频率，那不同维度之间会高度冗余，无法提供新的信息。

频率的指数级递增（$10000^{2i/d}$）确保不同维度携带的位置信息在数学上线性无关。

---

## 为什么要区分sin 与 cos？

即：为什么不能所有维度都用 $\sin$，或者都用 $\cos$？

✅ 原因是：

**使用正弦和余弦作为一组傅里叶基底，可以让两个位置之间的点积直接反映它们的相对距离。**

我们可以推导如下：

对于两个位置 $pos_1$ 和 $pos_2$，它们的编码为：

* 第 $i$ 对维度（$2i, 2i+1$）为：

  $$
  \left[ \sin\left(\frac{pos}{\lambda_i}\right), \cos\left(\frac{pos}{\lambda_i}\right) \right]
  $$

它们的点积为：

$$
PE(pos_1) \cdot PE(pos_2)
= \sum_i \left( \sin\left(\frac{pos_1}{\lambda_i}\right)\sin\left(\frac{pos_2}{\lambda_i}\right) + \cos\left(\frac{pos_1}{\lambda_i}\right)\cos\left(\frac{pos_2}{\lambda_i}\right) \right)
= \sum_i \cos\left(\frac{pos_1 - pos_2}{\lambda_i}\right)
$$

这个结果说明：

> **两个位置编码的点积表示的是“相对位置差”而不是“绝对位置”，从而允许 Transformer 间接感知位置信息。**

这正是注意力机制缺失位置信息的一个优雅补救。

若只用 $\sin$，点积就变成：

$$
PE(pos_1) \cdot PE(pos_2) = \sum_i \sin\left(\frac{pos_1}{\lambda_i}\right)\sin\left(\frac{pos_2}{\lambda_i}\right)
$$

这无法简化为 $\cos(pos_1 - pos_2)$ 的形式，模型失去了相对位置信息的内积解码能力。

---

## 为什么需要奇偶交错


> 如果不暗招奇偶交替的形式排布，而用其他排布方式，比如把前一半维度都用 sin，后一半都用 cos，还能保持这种性质吗？

我们设：

* 前 $d/2$ 维使用 $\sin\left(\frac{pos}{\lambda_i}\right)$；
* 后 $d/2$ 维使用 $\cos\left(\frac{pos}{\lambda_i}\right)$；

那么点积仍然为：

$$
PE(pos_1) \cdot PE(pos_2) = \sum_i \sin\left(\frac{pos_1}{\lambda_i}\right)\sin\left(\frac{pos_2}{\lambda_i}\right) + \cos\left(\frac{pos_1}{\lambda_i}\right)\cos\left(\frac{pos_2}{\lambda_i}\right) = \sum_i \cos\left(\frac{pos_1 - pos_2}{\lambda_i}\right)
$$

这表明：

> 只要 sin 和 cos 是**配对使用**并且频率一一对应，无论是“交错排布”还是其他排布方式，都能保留原有性质。

**⚠️ 但注意：**

如果前半 sin 使用的是一组频率，后半 cos 使用的是**另一组频率**（比如重复、翻转或错位），那么点积结果将不再是纯粹的 $\cos(pos_1 - pos_2)$ 叠加，也就失去了相对位置可解释性。


