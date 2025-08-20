---
title: GRPO原理及相关对比
tags:
  - 强化学习
  - LLM
---
{{ render_tags() }}

## 1. 背景与动机

在大语言模型（LLM）的后训练阶段，常用方法是 **RLHF (Reinforcement Learning with Human Feedback)**。其中 **PPO (Proximal Policy Optimization)** 被广泛使用（如 InstructGPT、ChatGPT）。
然而，PPO 存在以下问题：

* **训练不稳定**：奖励模型噪声较大，容易导致 collapse 或奖励 hacking。
* **样本效率低**：LLM 每次采样成本高，而 PPO 在收集 trajectories 时利用不足。
* **对比性弱**：奖励模型提供的信号常常绝对化，无法刻画相对偏好。

为解决这些问题，提出了 **GRPO (Group Relative Policy Optimization)**。
它的核心思想是：**利用相对比较（preference）而非绝对奖励值，构造组内归一化的优势函数，提升训练稳定性和样本效率。**



## 2. 技术原理与数理表达

### 2.1 算法思想

* 将候选生成结果按组（group）进行比较。
* 通过组内排序或相对奖励差异来构造 **优势函数 A**。
* 策略更新时不依赖单一奖励模型输出，而是依赖 **相对偏好分布**。

### 2.2 数学定义

设策略 $\pi_\theta(y|x)$ 生成文本 $y$，输入为 $x$。
奖励模型给出得分 $r(y|x)$。

在 GRPO 中，对于同一提示 $x$，生成一组候选 ${y_1, y_2, \dots, y_K}$，形成 group $G$。
定义归一化相对优势：

$$
A(y_i) = \frac{r(y_i) - \mu_G}{\sigma_G + \epsilon}
$$

其中 $\mu_G, \sigma_G$ 分别是组内均值和标准差。

策略优化目标为：

$$
L^{GRPO}(\theta) = \mathbb{E}_{x,G}\left[ \sum_{y_i \in G} \min \left( 
\rho_\theta(y_i) A(y_i), \; \text{clip}(\rho_\theta(y_i), 1-\epsilon, 1+\epsilon) A(y_i)
\right) \right]
$$

其中

* $\rho_\theta(y_i) = \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}$ 是重要性采样比。
* 类似 PPO，引入 clip 防止更新过大。
* 与 PPO 不同的是，优势函数来源于 **组内相对奖励**。

### 2.3 直观理解

* PPO：依赖单个样本的绝对奖励，噪声大。
* GRPO：依赖一组样本的相对奖励，减少了奖励模型误差影响。



## 3. 与 PPO / TRPO 的对比

| 特性     | TRPO         | PPO             | GRPO                   |
| ------ | ------------ | --------------- | ---------------------- |
| 策略约束   | KL约束，保证单步收敛性 | Clipping 近似约束   | Clipping + Group 相对奖励  |
| 奖励信号   | 绝对奖励         | 绝对奖励            | **组内相对奖励**             |
| 样本效率   | 中等           | 较高              | **更高**                 |
| 收敛稳定性  | 高，但计算量大      | 较好，仍可能 collapse | **更稳定**                |
| LLM 应用 | 不常用（计算过大）    | 主流（ChatGPT）     | 新兴（减少奖励 hacking，偏好更一致） |


## 4. 训练过程

1. **数据来源**

   * Prompt 来自人工设计或合成任务。
   * 策略模型生成多样化候选答案。

2. **奖励模型**

   * 基于人类偏好数据训练。
   * 输出相对分数，而不是绝对 reward。

3. **分组采样**

   * 对每个 prompt，采样 $K$ 个候选结果形成 group。

4. **相对优势计算**

   * 组内归一化：$A(y_i) = \frac{r(y_i)-\mu_G}{\sigma_G}$.

5. **策略更新**

   * 使用 GRPO 损失函数进行反向传播。

6. **迭代训练**

   * 更新策略模型，重复采样-训练流程。


## 5. 伪代码

```python
# GRPO Training Pseudocode

for iteration in range(num_iters):
    batch_prompts = sample_prompts()
    groups = []
    
    for x in batch_prompts:
        candidates = [policy.generate(x) for _ in range(K)]
        rewards = [reward_model(x, y) for y in candidates]
        
        mu = mean(rewards)
        sigma = std(rewards) + eps
        advantages = [(r - mu) / sigma for r in rewards]
        
        groups.append((x, candidates, advantages))
    
    loss = 0
    for (x, candidates, advantages) in groups:
        for y, A in zip(candidates, advantages):
            rho = policy_prob(y|x) / old_policy_prob(y|x)
            clipped = clip(rho, 1 - eps, 1 + eps)
            loss += -min(rho * A, clipped * A)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```


## 6. 示例

**GRPO 核心机制：组内相对奖励优化**

```
Prompt: "Explain quantum computing"
   ├── Candidate 1 → r=0.8
   ├── Candidate 2 → r=0.6
   ├── Candidate 3 → r=0.2
   └── Candidate 4 → r=0.9

组均值 μ=0.625, σ=0.25
   A1=(0.8-0.625)/0.25=+0.7
   A2=-0.1
   A3=-1.7
   A4=+1.1
```

这样避免了绝对值差异过大或奖励噪声导致的不稳定更新。

# 总结

* **GRPO (Group Relative Policy Optimization)** 是对 PPO 的改进，将奖励从绝对值转为组内相对优势，提升稳定性和样本效率。
* 在 LLM RLHF 中尤其有效，减少奖励 hacking 和模式坍缩。
* 与 PPO 相比：更稳定、更节省样本，但计算成本稍高（需要组采样）。
* 已逐渐成为 LLM 后训练的重要优化方法。

