# `self.draft_model_runner.forward` 推理流程分析

## 概述

本文档详细分析在 EAGLE + MTP 场景下，`self.draft_model_runner.forward` 的完整推理流程。这是 draft model 生成候选 tokens 的核心过程。

## 1. 调用入口

在 `eagle_worker.py` 的 `draft_forward` 方法中：

```683:685:python/sglang/srt/speculative/eagle_worker.py
            logits_output, _ = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            )
```

### 1.1 上下文

这是在 draft 生成循环中的一步：

```636:693:python/sglang/srt/speculative/eagle_worker.py
    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        
        # ... 准备数据 ...
        
        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            # 选择 top-k tokens
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            
            # ... 记录结果 ...
            
            if i == self.speculative_num_steps - 1:
                break

            # 设置输入
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states  # 关键：更新 hidden states

            # 运行 forward
            logits_output, _ = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            )
            
            # 处理输出
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            hidden_states = logits_output.hidden_states  # 更新 hidden states 用于下一步

        return score_list, token_list, parents_list
```

**关键点**：
- `forward_batch` 包含 `spec_info`（类型为 `EagleDraftInput`）
- `spec_info.hidden_states` 在每次迭代前被更新
- `forward_batch.input_ids` 是当前步骤要预测的 token IDs
- `forward_batch.positions` 是位置信息

## 2. ModelRunner.forward 方法

### 2.1 入口方法

```1756:1785:python/sglang/srt/model_executor/model_runner.py
    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> Tuple[Union[LogitsProcessorOutput, PPProxyTensors], bool]:
        self.forward_pass_id += 1

        with get_global_expert_distribution_recorder().with_forward_pass(
            self.forward_pass_id,
            forward_batch,
        ):
            output = self._forward_raw(
                forward_batch,
                skip_attn_backend_init,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()
        return output
```

### 2.2 _forward_raw 方法

```1787:1838:python/sglang/srt/model_executor/model_runner.py
    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> Tuple[Union[LogitsProcessorOutput, PPProxyTensors], bool]:
        can_run_cuda_graph = bool(
            forward_batch.forward_mode.is_cuda_graph()
            and self.graph_runner
            and self.graph_runner.can_run(forward_batch)
        )
        if can_run_cuda_graph:
            ret = self.graph_runner.replay(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            return ret, can_run_cuda_graph

        # For MLP sync
        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.prepare_mlp_sync_batch(self)

        if forward_batch.forward_mode.is_decode():
            ret = self.forward_decode(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode.is_extend():
            ret = self.forward_extend(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        # ... 其他模式 ...

        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.post_forward_mlp_sync_batch(ret)

        return ret, can_run_cuda_graph
```

**关键点**：
- 在 draft forward 中，`forward_mode` 是 `DECODE`
- 因此会调用 `forward_decode` 方法

### 2.3 forward_decode 方法

```1680:1697:python/sglang/srt/model_executor/model_runner.py
    def forward_decode(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            self.attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )
```

**关键点**：
- 如果 `skip_attn_backend_init=False`，会初始化 attention backend 的元数据
- 调用 `self.model.forward()`，传入：
  - `input_ids`: 当前要处理的 token IDs
  - `positions`: 位置信息
  - `forward_batch`: 完整的 forward batch（包含 `spec_info`）

## 3. DeepSeekV3 MTP 模型 Forward

### 3.1 模型结构

对于 DeepSeekV3 MTP，`self.model` 是 `DeepseekV3ForCausalLMNextN` 实例：

```126:164:python/sglang/srt/models/deepseek_nextn.py
class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        # if not set, model load will be broken in DeepseekV3ForCausalLM load_weights()
        self.pp_group = get_pp_group()
        self.determine_num_fused_shared_experts("DeepseekV3ForCausalLMNextN")

        self.model = DeepseekModelNextN(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
```

### 3.2 DeepseekModelNextN.forward

这是核心的 MTP 模型 forward：

```80:123:python/sglang/srt/models/deepseek_nextn.py
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=(
                input_embeds.device if input_embeds is not None else input_ids.device
            ),
        )

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        if hidden_states.shape[0] > 0:
            hidden_states = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )

        residual = None
        with get_global_expert_distribution_recorder().disable_this_region():
            hidden_states, residual = self.decoder(
                positions, hidden_states, forward_batch, residual, zero_allocator
            )

        if not forward_batch.forward_mode.is_idle():
            if residual is not None:
                hidden_states, _ = self.shared_head.norm(hidden_states, residual)
            else:
                hidden_states = self.shared_head.norm(hidden_states)

        return hidden_states
```

## 4. 详细数据流

### 4.1 输入准备

在 `draft_forward` 循环中，每次迭代前：

1. **选择 Top-K Tokens**：
   ```python
   input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
       i, topk_p, topk_index, hidden_states, scores, self.topk
   )
   ```
   - `input_ids`: 当前步骤要预测的 token IDs（shape: `[batch_size * topk]`）
   - `hidden_states`: 上一步的 hidden states（用于下一步）

2. **更新 ForwardBatch**：
   ```python
   forward_batch.input_ids = input_ids
   forward_batch.positions.add_(1)  # 位置递增
   spec_info.hidden_states = hidden_states  # 更新 spec_info 中的 hidden states
   ```

### 4.2 Token Embedding

在 `DeepseekModelNextN.forward` 中：

```python
if input_embeds is None:
    hidden_states = self.embed_tokens(input_ids)
else:
    hidden_states = input_embeds
```

- 将 `input_ids` 转换为 embeddings
- Shape: `[batch_size * topk, hidden_size]`

### 4.3 拼接 Hidden States（MTP 核心）

```python
hidden_states = self.eh_proj(
    torch.cat(
        (
            self.enorm(hidden_states),  # Token embeddings 的归一化
            self.hnorm(forward_batch.spec_info.hidden_states),  # Target model 的 hidden states
        ),
        dim=-1,
    )
)
```

**关键点**：
- `forward_batch.spec_info.hidden_states` 来自 target model 的 hidden states
- 这是 MTP 的核心：将 token embedding 和 target hidden states 拼接
- `eh_proj` 是一个线性层：`Linear(2 * hidden_size, hidden_size)`
- 输出 shape: `[batch_size * topk, hidden_size]`

### 4.4 Decoder Layer

```python
hidden_states, residual = self.decoder(
    positions, hidden_states, forward_batch, residual, zero_allocator
)
```

- `decoder` 是一个 `DeepseekV2DecoderLayer`
- 包含：
  - Attention 层（使用 `forward_batch` 中的 KV cache）
  - MoE/MLP 层
- 输出 shape: `[batch_size * topk, hidden_size]`

### 4.5 最终归一化

```python
if not forward_batch.forward_mode.is_idle():
    if residual is not None:
        hidden_states, _ = self.shared_head.norm(hidden_states, residual)
    else:
        hidden_states = self.shared_head.norm(hidden_states)
```

- 应用 RMSNorm
- 输出 shape: `[batch_size * topk, hidden_size]`

### 4.6 Logits 处理

在 `DeepseekV3ForCausalLMNextN.forward` 中：

```python
return self.logits_processor(
    input_ids, hidden_states, self.lm_head, forward_batch
)
```

- `lm_head`: 将 hidden states 投影到词汇表大小
- `logits_processor`: 处理 logits（应用 temperature, top-p 等）
- 输出 `LogitsProcessorOutput`，包含：
  - `next_token_logits`: shape `[batch_size * topk, vocab_size]`
  - `hidden_states`: shape `[batch_size * topk, hidden_size]`

### 4.7 返回处理

在 `draft_forward` 中：

```python
logits_output, _ = self.draft_model_runner.forward(
    forward_batch, skip_attn_backend_init=True
)

# 处理输出
probs = torch.softmax(logits_output.next_token_logits, dim=-1)
topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
hidden_states = logits_output.hidden_states  # 用于下一步
```

- 对 logits 应用 softmax 得到概率分布
- 选择 top-k 个候选 tokens
- 保存 hidden states 用于下一步迭代

## 5. 关键数据结构

### 5.1 ForwardBatch

```python
@dataclass
class ForwardBatch:
    forward_mode: ForwardMode  # DECODE
    input_ids: torch.Tensor  # [batch_size * topk]
    positions: torch.Tensor  # [batch_size * topk]
    spec_info: EagleDraftInput  # 包含 hidden_states
    # ... 其他字段 ...
```

### 5.2 EagleDraftInput

```python
@dataclass
class EagleDraftInput:
    topk_p: torch.Tensor  # [batch_size, topk] - 概率
    topk_index: torch.Tensor  # [batch_size, topk] - token indices
    hidden_states: torch.Tensor  # [batch_size, hidden_size] - Target model 的 hidden states
    # ... 其他字段 ...
```

## 6. 完整流程图

```
draft_forward 循环 (i = 0 to speculative_num_steps-1)
    │
    ├─> select_top_k_tokens()  # 选择 top-k tokens
    │   └─> input_ids, hidden_states
    │
    ├─> 更新 forward_batch
    │   ├─> forward_batch.input_ids = input_ids
    │   ├─> forward_batch.positions.add_(1)
    │   └─> spec_info.hidden_states = hidden_states
    │
    ├─> draft_model_runner.forward(forward_batch)
    │   │
    │   ├─> ModelRunner.forward()
    │   │   └─> ModelRunner._forward_raw()
    │   │       └─> ModelRunner.forward_decode()
    │   │           └─> model.forward(input_ids, positions, forward_batch)
    │   │
    │   └─> DeepseekV3ForCausalLMNextN.forward()
    │       │
    │       ├─> DeepseekModelNextN.forward()
    │       │   ├─> embed_tokens(input_ids)  # Token embedding
    │       │   ├─> eh_proj(cat([enorm(embeds), hnorm(spec_info.hidden_states)]))  # MTP 核心
    │       │   ├─> decoder(positions, hidden_states, forward_batch)  # Decoder layer
    │       │   └─> shared_head.norm(hidden_states)  # 归一化
    │       │
    │       └─> logits_processor(input_ids, hidden_states, lm_head, forward_batch)
    │           └─> 返回 LogitsProcessorOutput
    │
    └─> 处理输出
        ├─> softmax(logits) -> probs
        ├─> fast_topk(probs) -> topk_p, topk_index
        └─> hidden_states = logits_output.hidden_states  # 用于下一步
```

## 7. 关键特性

### 7.1 MTP 的核心机制

- **双输入设计**：MTP 模型同时接收：
  1. Token embeddings（当前要预测的 tokens）
  2. Target model 的 hidden states（来自上一轮验证）

- **拼接投影**：通过 `eh_proj` 将两个输入拼接并投影到相同维度

- **轻量级结构**：只使用一个 decoder layer，相比完整的 target model 更轻量

### 7.2 Hidden States 传递

- **来源**：`spec_info.hidden_states` 最初来自 target model 的验证阶段
- **更新**：在 draft forward 的每一步，使用上一步的输出更新 `hidden_states`
- **传递**：通过 `forward_batch.spec_info` 传递给模型

### 7.3 批量处理

- **Batch Size**：`batch_size * topk`（每个序列有 topk 个候选分支）
- **并行计算**：所有候选分支并行计算，提高效率

## 8. 总结

`self.draft_model_runner.forward` 的推理流程：

1. **输入准备**：从 `EagleDraftInput` 获取 hidden states，准备 input_ids
2. **Token Embedding**：将 input_ids 转换为 embeddings
3. **MTP 拼接**：将 token embeddings 和 target hidden states 拼接并投影
4. **Decoder 处理**：通过一个 decoder layer 处理
5. **归一化**：应用 RMSNorm
6. **Logits 生成**：通过 lm_head 生成 logits
7. **输出处理**：返回 logits 和 hidden states，用于下一步迭代

这个流程充分利用了 DeepSeekV3 的 MTP 能力，通过轻量级的 draft model 快速生成候选 tokens，然后由 target model 验证，实现了高效的推测解码。






