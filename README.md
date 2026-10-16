# Build a Tiny Transformer

通过逐阶段亲手实现，学习 GPT 风格的字符级文本生成器。目前 **Milestone 1/2 已完成，Milestone 3 为待完成的多头因果注意力练习**，尚未实现完整模型或训练。

默认学习方式是：阅读目标和提示 → 实现核心计算 → 运行测试 → 观察输出 → 解释原理，再进入下一阶段。当前不提供练习的完整参考答案。

## 环境与运行

需要 Python 3.12 和 `uv`，从仓库根目录运行：

```bash
uv sync --locked
uv run pytest -m "not exercise"
uv run ruff check .
```

依赖包括 NumPy、CPU 版 PyTorch，以及开发工具 pytest、Ruff。首次安装需要访问包索引；后续练习不需要 GPU 或下载数据。PyTorch 的 CPU 索引已在 `pyproject.toml` 中配置；依赖版本记录在 `uv.lock`。缩进使用四个空格，可用 `uv run ruff format .` 格式化。

源码位于 `src/transformer/`，测试位于 `tests/`，示例语料位于 `data/toy.txt`。安装会以可编辑方式提供 `transformer` 包；可用 `uv build` 构建分发包，产物位于 `dist/`。

## Milestone 1：从文本到张量（已完成）

语言模型在每个位置预测下一个 token。本项目的 token 是一个字符，包括空格和换行。字符表将字符映射到整数 ID；这些 ID 是类别标签，不能把 ID 的数值大小直接当成字符的含义。

`src/transformer/data.py` 已实现：

- `CharacterTokenizer.from_text`：构建排序、去重的字符表，拒绝空语料。
- `encode` / `decode`：字符与 ID 互相转换，拒绝未知字符和越界 ID，支持空序列。
- `make_batch`：有放回地随机采样起点，返回 CPU 上形状为 `[B, T]` 的 `torch.long` 输入与目标，支持指定随机数生成器。

例如 `hello` 在上下文长度为 4 时：

```text
input x:  h e l l
target y: e l l o
```

目标向后错一位，每个位置都有自己的预测任务。序列有 `N` 个字符、上下文长度为 `T` 时，需要 `T + 1` 个字符组成一组输入与目标，因此合法起点共 `N - T` 个。

```bash
uv run pytest tests/test_data.py tests/test_setup.py
uv run python -m transformer.inspect_data
uv run python -m transformer.inspect_data --text path/to/file.txt
```

检查入口默认 `B=16`、`T=32`，打印词表大小、张量形状和一组解码结果。`data/toy.txt` 是很小的原创故事，适合学习数据流程；流畅生成不是这一语料的目标。

复盘：为什么目标要错一位？为什么 token ID 使用 `torch.long`？当 `N=10`、`T=4` 时有哪些合法起点？

## Milestone 2：单头因果自注意力（已完成）

这一阶段让每个位置根据当前上下文，选择应该读取哪些位置的信息。暂不训练语言模型，也不实现多头注意力。

### 输入与接口

`src/transformer/attention.py` 中的 `CausalSelfAttentionHead.forward` 已实现，并通过单头注意力全部测试。下面保留接口和推导，供复习和后续复用。

```python
head = CausalSelfAttentionHead(embedding_dim=6, head_dim=3)
output, attention_weights = head(x)
```

`x` 是浮点特征，**不是 Milestone 1 的整数 token ID**。演示使用随机特征帮助观察计算；后续会由字符嵌入和位置嵌入产生这些特征。本阶段不要求把两阶段直接接成模型。

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `x` | `[B, T, C]` | 每个位置的输入特征 |
| `Q`, `K`, `V` | `[B, T, D]` | 查询、键、值的投影结果 |
| `scores` | `[B, T, T]` | 每个 query 与各个 key 的匹配分数 |
| `attention_weights` | `[B, T, T]` | 每行是一个位置读取其他位置的比例 |
| `output` | `[B, T, D]` | 根据注意力权重汇总的信息 |

`B` 为批量大小，`T` 为实际序列长度，`C=embedding_dim`，`D=head_dim`。两个构造参数必须是正整数；输入必须是三维浮点张量，`T >= 1` 且最后一维为 `C`，否则抛出 `ValueError`。输入与模块参数的设备、浮点类型应匹配，沿用 PyTorch 的使用规则。

### 核心计算与步骤

Q 表示“当前位置想找什么”，K 表示“每个位置提供怎样的匹配线索”，V 表示“匹配后实际读取的内容”。三者来自同一个输入序列的不同可学习投影，因此称为自注意力。

数学流程如下；`Kᵀ` 只交换最后两个维度：

```text
Q = query(x), K = key(x), V = value(x)
scores = QKᵀ / √D
weights = softmax(causal_mask(scores), dim=-1)
output = weights V
```

实现采用下面的计算顺序，可逐步检查形状：

1. 通过已提供的 `query`、`key`、`value` 层生成 Q/K/V。
2. 计算每个 query 与所有 key 的点积，除以 `sqrt(head_dim)`。思考 head_dim 增大时，点积分数的尺度会如何变化。
3. 根据当前 `T` 和 `x.device` 创建遮罩，在 softmax **之前**把未来位置的分数设为负无穷。可查阅 `torch.ones`、`tril`、`masked_fill`。
4. 沿最后一维，也就是 key 位置，执行 softmax。
5. 使用权重对 V 加权求和，返回输出和权重，保留计算图以支持反向传播。

不要使用 `nn.MultiheadAttention` 或 `scaled_dot_product_attention` 代替这些步骤。本阶段不加入 dropout。

### 因果遮罩

对四个位置，下表中 `1` 表示允许读取，`0` 表示禁止读取；行是 query，列是 key：

```text
        key 0 1 2 3
query 0     1 0 0 0
query 1     1 1 0 0
query 2     1 1 1 0
query 3     1 1 1 1
```

保留对角线：当前位置可以读取自己。比如输入 `h e l l`，第二个位置预测下一个 `l` 时，只能读取 `h e`，不能读取输入中的后续字符。遮罩后的未来分数为负无穷，softmax 后对应权重为零；每行允许位置的权重之和为一。

### 验证与复盘

```bash
uv run pytest -m milestone2
uv run python -m transformer.inspect_attention
```

检查入口打印形状、注意力矩阵、每行权重之和及未来权重是否全为零。Milestone 2 的全部测试已通过，`milestone2` 标记保留用于回归检查，`exercise` 标记已移除。

复盘问题与简答：

1. **Q、K、V 为什么采用不同的投影？为什么输出汇总的是 V？** Q/K 学习匹配关系，V 学习被汇总的内容；不同投影让两种职责可分别学习。
2. **为什么除以 `√D`，而不是 `√C`？** 在各分量近似独立且方差稳定时，点积方差随 D 增长，缩放可避免 softmax 过度尖锐；点积发生在 D 维，因此使用 D。
3. **为什么遮罩要在 softmax 之前？把未来分数设成零会怎样？** 先设为负无穷，才能让未来位置权重为零；设成零仍会产生正的指数权重。
4. **为什么 softmax 沿最后一维，而不是沿 query 维？** 每个 query 都要在所有允许的 key 之间分配读取比例，因此沿最后一维归一化。
5. **如果移除遮罩，下一字符预测会如何泄露答案？** 训练输入中的后续字符包含当前位置的预测目标，不加遮罩就可能直接读取答案。
6. **为什么只有一个位置时，输出等于该位置的 V？** 唯一允许位置的权重为 1，因此输出就是自身的 V。

## Milestone 3：多头因果注意力与输出投影（当前练习）

目标：复用你完成的单头模块，让多个头独立计算，再拼接和融合结果。每个头都有独立的 Q/K/V 参数，读取的是同一个完整输入 `x`，而不是把输入特征先分成几份。独立参数提供了学习不同关系的能力，但不保证各头训练后必然形成不同分工；当前演示也没有训练。

### 输入与接口

在 `src/transformer/attention.py` 中完成 `MultiHeadCausalSelfAttention.forward`：

```python
attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2)
output, attention_weights = attention(x)
```

构造函数和输入检查已提供。`embedding_dim=C`、`num_heads=H` 必须是正整数，并满足 `C % H == 0`，否则抛出 `ValueError`。每头宽度自动取 `D = C // H`，因此 `H × D = C`。

`self.heads` 使用 `nn.ModuleList` 注册 H 个独立的单头模块，使参数能被优化器、保存操作和设备/类型转换识别；`self.projection` 是无偏置的 `C → C` 线性层。输入仍须是 `[B, T, C]` 浮点特征，`T >= 1`，设备与类型匹配模块参数。

| 阶段 | 形状 | 操作目标 |
| --- | --- | --- |
| 共享输入 | `[B, T, C]` | 每个头读取同一个完整输入 |
| 每个头的输出 | `[B, T, D]` | 各自独立执行因果注意力 |
| 拼接后的特征 | `[B, T, H × D] = [B, T, C]` | 按头顺序保留全部特征 |
| 最终输出 | `[B, T, C]` | 输出投影混合不同头的特征 |
| 每个头的权重 | `[B, T, T]` | 每个头自己的匹配关系 |
| 返回的全部权重 | `[B, H, T, T]` | 新增 head 轴，保留各头信息 |

拼接保留每个头的特征块；求和或平均会把这些块合并，丢失头的对应位置，并把特征宽度保持为 D。输出投影则在拼接后学习如何混合不同头的信息，且保持最终宽度 C。权重只是按头堆叠，不经过输出投影，也不跨头归一化。

### 练习步骤与提示

1. 依次调用 `self.heads` 中的各头，收集每个头返回的输出和权重。不要重新实现单头计算。
2. 用 `torch.cat` 沿已有的特征维拼接输出，得到 `[B, T, C]`。不要沿序列维拼接，也不要把头求和或平均。
3. 将拼接结果送入 `self.projection`，得到最终输出。
4. 用 `torch.stack` 在新的头维上堆叠权重，得到 `[B, H, T, T]`，按 `ModuleList` 顺序返回 `(output, attention_weights)`。

注意 `cat` 合并已有维度，`stack` 新增一个维度。保留计算图，不要在 forward 中使用 `detach` 或 `no_grad`。本阶段不使用封装好的注意力接口，不加入 dropout、嵌入、残差连接或训练循环。

### 检查顺序与预期结果

```bash
# Milestone 1/2 回归测试与第三阶段已提供的基础行为：应通过
uv run pytest -m "not exercise"

# 当前阶段：基础检查通过后，在未实现的 forward 处失败
uv run pytest -m milestone3 -x

# 只检查待实现的核心计算，逐个解决失败
uv run pytest -m "milestone3 and exercise" -x

# 观察 C=8、H=2、T=4 时每个头的注意力
uv run python -m transformer.inspect_multihead

# 完成练习后检查全部测试与三个入口
uv run pytest
uv run python -m transformer.inspect_data
uv run python -m transformer.inspect_attention
uv run python -m transformer.inspect_multihead
uv run ruff check .
```

只有 Milestone 3 的核心计算测试保留 `exercise` 标记；骨架阶段这些测试和全量测试会因 `NotImplementedError` 失败。不要删除、跳过或改成预期失败来绕过测试。新检查入口在未完成时以状态码 1 退出并提示编辑位置；完成后打印形状、各头矩阵及行和、未来权重检查。

如果已有虚拟环境且当前无法联网，可用 `uv run --no-sync ...` 运行检查，避免重新同步依赖。

完成标准：

- 全部测试、Ruff 和修改文件的格式检查通过，三个检查入口正常。
- 输出为 `[B, T, C]`，权重为 `[B, H, T, T]`；各头独立满足因果性与概率规则。
- 固定参数示例验证头的顺序、拼接方向和输出投影；修改未来输入或另一个 batch 样本不影响原有因果边界。
- `H=1` 且输出投影为单位矩阵时，与原单头结果一致。
- 长度为一和变化的序列长度正确，输入、各头 Q/K/V 与输出投影均获得有限梯度。

复盘问题（完成后再回答）：

1. 为什么各头读取同一个完整输入，而不是各自只读取 C/H 个输入特征？
2. 为什么拼接输出使用 `cat`，堆叠权重使用 `stack`？各自选择哪个维度？
3. 如果把各头输出平均，会丢失什么信息？形状如何变化？
4. 为什么拼接后还需要输出投影？它会跨序列位置混合信息吗？
5. 为什么增加多个头和输出投影后，仍然不会读取未来信息？
6. 独立参数是否保证各头学到不同关系？如何观察它们的行为？

## 后续阶段

完成并复盘 Milestone 3 后，再加入字符/位置嵌入、Transformer 块和残差连接，然后学习连续的训练/验证划分、固定批次过拟合检查，以及带温度的生成和检查点保存。

初始完整模型计划使用 64 维嵌入、4 个注意力头、2 个预归一化 Transformer 块；这些属于后续目标，尚未实现。
