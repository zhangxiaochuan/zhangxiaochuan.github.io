---
title: 在 Kubernetes 容器中，如何正确理解和获取 CPU 核心数？
tags:
  - Kubernetes
  - 容器
  - 工程实践
  - 性能优化
---

{{ render_tags() }}

在 **PyTorch DataLoader、DALI、并行解码、CPU 密集型预处理** 等场景里，`os.cpu_count()` 经常是吞吐与稳定性问题的起点。容器里看到的“CPU 核心数”并不是你实际能用的并行度，正确理解它，是避免并发失控的第一步。

---

## 1. `os.cpu_count()` 的语义：它看到的是宿主机

`os.cpu_count()` 返回的是**操作系统视角的在线逻辑 CPU 数**，在容器里通常等于宿主机的核数，而不是 cgroup 配额允许的并行度。

容器并不会“隐藏 CPU”，它只是通过 **cgroup** 限制**单位时间内可使用的 CPU 时间**。

---

## 2. CPU 限制的本质是时间配额（CFS quota）

在 cgroup v1 中，CPU 配额由两项决定：

* `cpu.cfs_period_us`：调度周期（默认 100ms）
* `cpu.cfs_quota_us`：一个周期内允许使用的 CPU 时间

可用并行度近似为：

```
可用 CPU 并行度 ≈ cpu.cfs_quota_us / cpu.cfs_period_us
```

当配额用完时，cgroup 会被 **throttle**，所有 runnable 进程暂停，等下一周期重启——这就是常见的 **CPU 使用率锯齿、延迟抖动** 的根源。

---

## 3. 没有设置 limit 的容器：quota 可能是无限

在 K8s 里，很多 Pod 只有 `requests`，没有 `limits`。这时：

* cgroup v1：`cpu.cfs_quota_us = -1`
* cgroup v2：`cpu.max = max <period>`

**语义是 unlimited**，此时不能再用 quota 推导核数，应该 fallback 到宿主机视角或 cpuset。

---

## 4. cgroup v1/v2 差异必须同时兼容

cgroup 版本由 **节点 OS + systemd** 决定，与 K8s 版本无关。常见路径差异：

| 功能 | v1 | v2 |
| --- | --- | --- |
| CPU quota | `cpu.cfs_quota_us` + `cpu.cfs_period_us` | `cpu.max` |
| cpuset | `cpuset.cpus` | `cpuset.cpus.effective` |

**只支持一种版本的实现，在多集群环境中必然失效。**

---

## 5. 推荐的获取逻辑（Python）

优先读 cgroup，显式处理 unlimited，再 fallback：

```python
import os
import math
from pathlib import Path

def get_container_cpu_count() -> int:
    # cgroup v2
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.exists():
        quota, period = cpu_max.read_text().strip().split()
        if quota != "max":
            return max(1, math.floor(int(quota) / int(period)))

    # cgroup v1
    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.exists() and period_path.exists():
        quota = int(quota_path.read_text())
        period = int(period_path.read_text())
        if quota > 0 and period > 0:
            return max(1, math.floor(quota / period))

    # fallback
    return os.cpu_count() or 1
```

---

## 6. 工程实践建议

* **并发度以 cgroup 配额为准**，不要默认 `os.cpu_count()`。
* 对于 unlimited 的 Pod，建议设置业务侧上限（如 `max_workers_cap`）。
* 在 CPU + IO 混合任务中，过度并发更容易触发 throttle 与 cache 退化。

---

## 核心结论

1. 容器里没有“真实 CPU 核数”，只有 **可用 CPU 时间**。
2. `os.cpu_count()` 看到的是宿主机，不是容器配额。
3. cgroup v1 / v2 差异必须兼容，并显式处理 unlimited。
4. 并行策略应以 cgroup 为事实来源，而不是宿主机视角。

**一句话总结：在 Kubernetes 容器中，正确的 CPU 并发决策必须基于 cgroup 配额，并处理未设置 limit 的情况。**
