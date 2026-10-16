# Build a Tiny Transformer

通过逐阶段亲手实现，学习 GPT 风格的字符级文本生成器。目前 **Milestone 1 已完成，Milestone 2 为待完成的单头因果注意力练习**，尚未实现完整模型或训练。

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

## Milestone 2：单头因果自注意力（当前练习）

这一阶段让每个位置根据当前上下文，选择应该读取哪些位置的信息。暂不训练语言模型，也不实现多头注意力。

### 输入与接口

在 `src/transformer/attention.py` 中完成 `CausalSelfAttentionHead.forward`。构造函数、三个可学习的无偏置线性层和输入检查已提供。

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

按下面顺序实现，每一步都先检查形状：

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

### 检查顺序与预期结果

```bash
# 已完成的数据处理、环境检查，以及新模块的构造和输入检查：应通过
uv run pytest -m "not exercise"

# 当前阶段：先通过基础检查，然后在未完成的 forward 处失败
uv run pytest -m milestone2 -x

# 只检查待实现的计算；逐个解决失败
uv run pytest -m "milestone2 and exercise" -x

# 观察固定随机种子的四个位置
uv run python -m transformer.inspect_attention

# 完成练习后的最终检查
uv run pytest
uv run ruff check .
```

骨架阶段，计算测试会因 `NotImplementedError` 失败，这是明确保留的练习入口；全量 `pytest` 暂时也会失败。不要删除、跳过或改成预期失败来绕过测试。`exercise` 标记只用于未完成练习，`milestone2` 包括本阶段全部测试；Milestone 1 测试已移出练习分类。

注意力检查入口在未完成时以状态码 1 退出并提示编辑位置；实现完成后打印输入/输出形状、注意力矩阵、每行权重之和以及未来权重是否全为零。

完成标准：

- 全部测试和 Ruff 检查通过，两个检查入口正常运行。
- 输出和权重形状正确，权重非负、行和为一、未来权重为零。
- 手算示例一致，修改未来特征不会影响过去输出，不同 batch 样本互不影响。
- 同一实例可处理不同序列长度；长度为一时只读取自身的 V。
- 输入与 Q/K/V 参数均能获得有限梯度。

测试中的手算示例使用 `C=3`、`D=2`，帮助区分该除以哪一个维度；先自己计算再对照预期。

复盘问题：

1. Q、K、V 为什么采用不同的投影？为什么输出汇总的是 V？
2. 为什么除以 `√D`，而不是 `√C`？
3. 为什么遮罩要在 softmax 之前？把未来分数设成零会怎样？
4. 为什么 softmax 沿最后一维，而不是沿 query 维？
5. 如果移除遮罩，下一字符预测会如何泄露答案？
6. 为什么只有一个位置时，输出等于该位置的 V？

## 后续阶段

完成并复盘 Milestone 2 后，再学习多头注意力与输出投影。之后加入字符/位置嵌入、Transformer 块、连续的训练/验证划分、固定批次过拟合检查，以及带温度的生成和检查点保存。

初始完整模型计划使用 64 维嵌入、4 个注意力头、2 个预归一化 Transformer 块；这些属于后续目标，尚未实现。
