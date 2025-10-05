---
title: TFDS vs WebDataset
tags:
  - 数据工程
---

{{ render_tags() }}

## 1. 两种格式的典型应用场景

* **TFDS (TensorFlow Datasets)**：提供**标准化的数据集打包格式（TFRecord + 元信息）**与“像官方数据集一样”分发/复现的能力。非常适合**团队内外分发、版本管理、元数据校验**，让数据成为“可复现实验资产”。

    > 在 PyTorch 场景中，可将 TFDS 作为**中立的发布格式**：上游用 TFDS 构建与验收，下游训练（尤其大规模分布式）再转换为更高吞吐的格式。

* **WebDataset (wds)**：通过**`.tar` 分片 + 键约定**组织样本，天然适配 **PyTorch IterableDataset**，支持 **本地/HTTP/S3 流式读取**、**分布式切分**、**高吞吐顺序 IO**。非常适合**海量多模态数据**与**跨机分布式训练**。


## 2. 数据存储结构

### 2.1 TFDS

* **目录化结构**（示意）：

  ```
  my_dataset/1.0.0/
    my_dataset-train.tfrecord-00000-of-00050
    my_dataset-train.tfrecord-00001-of-00050
    ...
    my_dataset-val.tfrecord-00000-of-00005
    dataset_info.json
    checksums.tsv
  ```
  
* **样本单位**：TF Example（Protocol Buffers），字段由 `features`（schema）约束。
* **元信息**：`dataset_info.json`（版本、样本数、features、splits）、`checksums.tsv`（完整性校验）。
* **切分**：天然支持 `train/val/test`。
* {dataset_name}-{split}.tfrecord-{shard_index:05d}-of-{num_shards:05d}是TFDS对一个 split 的 TFRecord 分片命名规范，如my_dataset-train.tfrecord-00000-of-00050
    * `my_dataset`：数据集名（dataset name）
    * `train`：切分（split），常见还有 validation/test 等
    * `.tfrecord`：存储格式，文件里是序列化的 tf.train.Example
    * `-00000-`：这是分片序号（shard_index），从 0 开始；这里是第 0 片
    * `of-00050`：这个 split 一共有 50 个分片（num_shards=50）

### 2.2 WebDataset

* **分片文件**：`shard-000000.tar`, `shard-000001.tar`, …（推荐 **100–500MB/片** 或 **1–5 万样本/片**，依数据体量与介质调优）。
* **样本组织**：同一**前缀**代表一个样本，后缀扩展名表示通道/模态：

  ```
  000123.jpg        # 图像
  000123.json       # 元数据/标注
  000123.txt        # 文本
  000124.jpg
  000124.json
  ...
  ```
  
* **常见键**：`"jpg"|"png"|"mp4"|"wav"|"json"|"txt"|"cls"|"__key__"`。
* **位置**：本地/HTTP/S3 均可（可直接**流式**读取，避免预下载）。

## 3. TFDS：从原始文件到可复现的数据集

目标：把**本地已有的原始文件**（图片/标注等）规范化为一个**带版本与切分**、可复现的 TFDS 数据集，能一键转换为 WebDataset 供 PyTorch 分布式训练。

### 3.1 原始数据准备

假设原始数据已整理成如下结构（示例）：

```
/data/my_dataset_raw/
  train/
    000001.jpg
    000001.json
    000002.jpg
    000002.json
    ...
  val/
    100001.jpg
    100001.json
    ...
```

其中 `*.json` 至少包含：

```json
{"id": "000001", "image_path": "/data/my_dataset_raw/train/000001.jpg", "label_name": "cat"}
```

> 提示：也可以用 CSV/JSONL 等任意格式，关键是能在 `_generate_examples` 里读出来并映射到特征字段。

### 3.2 定义最小 Builder（仅本地自用所需字段）

```python
# pip install tensorflow tensorflow-datasets
import tensorflow_datasets as tfds

class MyDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(None, None, 3)),
                "label": tfds.features.ClassLabel(names=["cat", "dog"]),
                "id": tf.string,
            }),
            supervised_keys=("image", "label"),  # 便于 as_supervised=True 返回 (x, y)
        )

    def _split_generators(self, dl_manager):
        root = "/data/my_dataset_raw"  # 改成原始数据根目录
        return {
            "train": self._generate_examples(f"{root}/train"),
            "val":   self._generate_examples(f"{root}/val"),
        }

    def _generate_examples(self, path):
        # 读取 path 下的 *.json，产出 (key, example_dict)
        import os, json, glob
        for meta_file in glob.glob(os.path.join(path, "*.json")):
            meta = json.load(open(meta_file, "r"))
            key  = meta["id"]                       # 任何可唯一标识样本的字符串
            yield key, {
                "image": meta["image_path"],        # 路径/bytes/ndarray 均可
                "label": meta["label_name"],        # 必须在 names=["cat","dog"] 内
                "id":    str(meta["id"]),
            }
```

> 说明：`example_dict` 的键 **必须与** `_info()` 中 `features` 的键一致；`meta["..."]` 是原始标注里的字段名，不要求同名。

### 3.3 构建（写入本地 TFDS 分片）

```python
import tensorflow_datasets as tfds

# data_dir 是 TFDS 输出目录（写出的 .tfrecord + dataset_info.json + checksums）
DATA_DIR = "/data/tfds_store"

builder = tfds.builder("my_dataset")   # 类名 MyDataset -> 名字 "my_dataset"
builder.download_and_prepare(data_dir=DATA_DIR)
```

构建完成后，目录类似：

```
/data/tfds_store/my_dataset/1.0.0/
  my_dataset-train.tfrecord-00000-of-00050
  my_dataset-train.tfrecord-00001-of-00050
  ...
  my_dataset-val.tfrecord-00000-of-00005
  dataset_info.json
  checksums.tsv
```

> **分片命名含义**：`{name}-{split}.tfrecord-{idx:05d}-of-{num:05d}`，即第 `idx` 个分片、共 `num` 片。多分片便于并行读与失败重试。

### 3.4 数据查验（快速抽样与元信息）

```python
import tensorflow_datasets as tfds
import numpy as np

ds_train = tfds.load(
    "my_dataset", split="train", data_dir="/data/tfds_store",
    as_supervised=True, shuffle_files=False
)

for i, (x, y) in enumerate(tfds.as_numpy(ds_train.take(3))):
    print(i, x.shape, x.dtype, int(y))
```

查看元信息：

```python
b = tfds.builder("my_dataset", data_dir="/data/tfds_store")
info = b.info
print(info.version, info.splits, info.features)
```


## 4. WebDataset：从 TFDS 到 PyTorch 分布式训练

### 4.1 将 TFDS 转为 WebDataset

本地构建完成后，可以把 TFDS 样本流转成 `.tar` 分片，便于 PyTorch 在多机多卡下高吞吐训练。核心转换示意（可嵌入其他数据工程脚本）：

```python
import tensorflow_datasets as tfds
from PIL import Image
from io import BytesIO
import tarfile, os, json

def write_sample(tar, key, pil_img, label, meta=None):
    """
    将单条样本写入一个已打开的 .tar 分片中。
    - key: 样本唯一键（用于生成 000001.jpg / 000001.cls / 000001.json）
    - pil_img: PIL.Image 格式的图像
    - label: 整数标签（或可转为 int 的字符串）
    - meta: 可选的字典元信息，将以 JSON 形式写入
    """
    # 1) 写入图像 => {key}.jpg
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")  # 将 PIL 图片编码为 JPEG 二进制
    ti = tarfile.TarInfo(f"{key}.jpg")
    ti.size = len(buf.getvalue())     # TarInfo 需要提前知道尺寸
    tar.addfile(ti, BytesIO(buf.getvalue()))

    # 2) 写入标签 => {key}.cls
    lb = str(int(label)).encode()     # 统一转为字节串，内容为类别整数
    ti = tarfile.TarInfo(f"{key}.cls")
    ti.size = len(lb)
    tar.addfile(ti, BytesIO(lb))

    # 3) 可选：写入元信息 => {key}.json
    if meta is not None:
        jb = json.dumps(meta).encode("utf-8")
        ti = tarfile.TarInfo(f"{key}.json")
        ti.size = len(jb)
        tar.addfile(ti, BytesIO(jb))


def tfds_split_to_wds(name, split, out_dir, data_dir, shard_size=50000):
    """
    将 TFDS 的某个 split（如 'train'/'test'）转换为 WebDataset 的 .tar 分片集合。

    参数：
    - name: TFDS 数据集名称（如 'cifar10' / 'my_dataset'）
    - split: 要导出的切分（如 'train' / 'test' / 'val'）
    - out_dir: 输出 .tar 分片的目录
    - data_dir: TFDS 数据所在目录（构建后的 TFRecord 存放路径）
    - shard_size: 每个 .tar 分片包含的样本数量上限（到达上限就切新分片）
    """
    os.makedirs(out_dir, exist_ok=True)

    # 加载 TFDS 数据集为 (image, label) 形式，并做轻度预取以提升转换吞吐
    ds = tfds.load(
        name,
        split=split,
        data_dir=data_dir,
        as_supervised=True  # -> (image, label)
    ).prefetch(1024)

    # 初始化分片编号与计数器
    shard_idx, in_shard = 0, 0
    tar = tarfile.open(os.path.join(out_dir, f"{name}-{split}-{shard_idx:06d}.tar"), "w")

    # 遍历样本流（tfds.as_numpy 将 TFDS 样本转为 numpy）
    for i, (img, label) in enumerate(tfds.as_numpy(ds)):
        key = f"{i:09d}"  # 使用 9 位零填充的顺序编号作为样本 key（便于排序与一致性）

        # 将 numpy 的 HWC uint8 图像转成 PIL，再写入 tar（同时写 label / meta）
        write_sample(
            tar=tar,
            key=key,
            pil_img=Image.fromarray(img),
            label=int(label),
            meta={"split": split}  # 可携带少量辅助信息，方便后续调试
        )
        in_shard += 1

        # 达到分片上限：关闭当前 .tar，开启下一个分片
        if in_shard >= shard_size:
            tar.close()
            shard_idx += 1
            in_shard = 0
            tar = tarfile.open(os.path.join(out_dir, f"{name}-{split}-{shard_idx:06d}.tar"), "w")

    # 数据遍历结束，关闭最后一个分片
    tar.close()
```

> 用法示例:
> 
> tfds_split_to_wds(name="my_dataset", split="train", out_dir="/data/wds/my_dataset", data_dir="/data/tfds_store", shard_size=50_000)


### 4.2 用 WebDataset 在 PyTorch 中训练（含分布式切分）

```python
# pip install webdataset torchvision
import os
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms

# 路径与超参数（与导出的wds路径一致）
# 训练分片（brace expansion），根据分片数量调整右边界
TRAIN_URLS = "/data/wds/my_dataset/my_dataset-train-{000000..000123}.tar"
VAL_URLS   = "/data/wds/my_dataset/my_dataset-val-{000000..000009}.tar"

BATCH_SIZE = 256           # 大 batch 建议配合多进程与足够带宽
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
IMAGE_SIZE = 224           # 示例：224 输入
LABEL_NAMES = ["cat", "dog"]  # 仅供注释参考（训练直接用 int 标签）

# 定义图片增强
tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])


def parse_sample(sample):
    """
    获取一个数据样本，可以在该函数中实现数据预处理
    WebDataset 每个样本是一个 dict：
      sample["jpg"] : PIL.Image
      sample["cls"] : bytes/int（这里我们保存的是文本整数，如 b"0" 或 b"1"）
      sample["__key__"] : 可选，样本键
      sample["json"] : 可选，元信息
    """
    img = tfms(sample["jpg"]) # 此处转为了Tensor格式
    y_raw = sample["cls"]
    # .cls 通常是 bytes（如 b"0"），也可能已被读取为 int；统一转 int
    if isinstance(y_raw, (bytes, bytearray)):
        y = int(y_raw.decode("utf-8"))
    else:
        y = int(y_raw)
    return img, y


# 从 WebDataset 到 PyTorch Iterable
def make_wds_loader(urls, parser, batch_size=BATCH_SIZE, shuffle_buf=10_000, training=True):
    ds = (
        wds.WebDataset(urls, shardshuffle=True)  # 分片级随机化
          .split_by_node()                      # 多机切分（DDP 多节点）
          .split_by_worker()                    # DataLoader 多 worker 切分
          .shuffle(shuffle_buf if training else 0)  # 样本级小缓冲区打乱（验证可为 0）
          .decode("pil")                        # 将图像解码为 PIL.Image
          .map(parser)                          # 变换为 (img, label)
          .batched(batch_size, partial=True)    # 打包为批次；partial=True 保留不满批
    )
    loader = DataLoader(
        ds,
        batch_size=None,                # 已在 .batched 完成组批
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True,
        pin_memory=True                 # 若使用 GPU 训练建议打开
    )
    return loader

train_loader = make_wds_loader(TRAIN_URLS, parse_sample, training=True)
val_loader = make_wds_loader(VAL_URLS, parse_sample, training=False)

# 训练/验证（示例）
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(num_classes=len(LABEL_NAMES)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

def train_one_epoch(epoch):
    model.train()
    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True) # 在parse_sample中已将其转为Tensor格式
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"[Epoch {epoch}] step={step} loss={loss.item():.4f}")

@torch.no_grad()
def evaluate():
    model.eval()
    total, correct = 0, 0
    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)
        logits = model(images)
        pred = logits.argmax(dim=1)
        total += targets.numel()
        correct += (pred == targets).sum().item()
    acc = correct / max(1, total)
    print(f"[Eval] acc={acc:.4%}")

for epoch in range(1, 6):
    train_one_epoch(epoch)
    evaluate()
```

**在 DDP/多机多卡中的要点**

* **DDP 初始化**（`torchrun`/`torch.distributed.launch`）后再创建 `dataset`，使 `.split_by_node()` 感知世界大小与 rank。
* **随机性**：每轮（epoch）可通过 `wds.ResampledShards` 或重建 `WebDataset` 实例来实现分片级重采样；或设置 `epoch` 相关种子影响 `.shuffle(buf)` 的顺序。
* **I/O 吞吐**：适当增大 `num_workers`、`prefetch_factor`、`bufsize`（`.shuffle(bufsize)`），并保证**网络或磁盘带宽**充足。
* **远程存储**：`urls` 可用 `https://.../shard-{...}.tar` 或 `s3://...`，支持**流式**训练。
