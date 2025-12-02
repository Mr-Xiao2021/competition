# MoE模型昇腾迁移优化技术报告

## 评测结果

| 评测指标 | 平均得分 |
|---------|---------|
| 峰值显存得分 | 100 |
| Prefill时延得分 | 109.3774     |
| Decode时延得分 | 360.0786     |
| **总分** | **189.8187** |

## 优化模型

本项目针对以下两个MoE（Mixture of Experts）模型进行了昇腾NPU适配与性能优化：

1. **DeepSeek-MoE-16B-Chat** - 深度求索开源的MoE大模型
2. **Qwen1.5-MoE-A2.7B-Chat** - 通义千问开源的MoE大模型

---

## 核心优化技术

### 1. MindSpore算子适配优化

#### 1.1 使用mint算子替代ops算子

将原始的`ops`算子替换为Ascend硬件亲和的`mint`算子

```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x1 = x[..., : x.shape[-1] // 2]
    # x2 = x[..., x.shape[-1] // 2 :]
    x1, x2 = ops.split(x,x.shape[-1]//2,dim=-1)
    return ops.cat((-x2, x1), dim=-1)
```


#### 1.2 使用mint.narrow替代切片操作

```python
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            # self.cos_cached[:seq_len].to(dtype=x.dtype),
            # self.sin_cached[:seq_len].to(dtype=x.dtype),
            ops.narrow(self.cos_cached, 0, 0, seq_len).to(dtype=x.dtype),
            ops.narrow(self.sin_cached, 0, 0, seq_len).to(dtype=x.dtype),
        )
```

**收益**：mint.narrow避免切片操作的额外内存拷贝。

---

### 2. FlashAttention优化

```python
else: # prefill开启
    # 融合后，采用mindspore的融合算子，flash_attention_score
    sparse_mode = 0
    if attention_mask is not None:
        attention_mask = ~attention_mask

    if self.is_causal:
        sparse_mode = 3
    global_attn_mask = ops.ones(2048, 2048, dtype=mindspore.bool_).triu(diagonal=1)
    attn_output = mindspore.ops.flash_attention_score(query_states, key_states, value_states,
head_num=self.num_heads, input_layout='BNSD', real_shift = None,padding_mask = None, attn_mask=global_attn_mask,scalar_value=1/math.sqrt(self.head_dim), keep_prob=1-self.attention_dropout,pre_tokens = 2147483647, next_tokens = 2147483647, inner_precise = 0,drop_mask = None, prefix = None, actual_seq_qlen = None, actual_seq_kvlen = None,sparse_mode=sparse_mode)
```



---

### 3. MoE路由与专家计算优化

#### 3.1 Qwen2-MoE: decode优化

```python
if routing_weights.shape[0] == 1:
            # 遍历激活的 top-k 专家
            final_hidden_states = ops.zeros((batch_size * sequence_length, hidden_dim), dtype=mindspore.float32)
            flat_topk_idx = selected_experts.view(-1)
            # idt = ops.zeros(1,dtype = mindspore.int64)
            for i in range(self.top_k):
                expert_idx = flat_topk_idx[i].item()
                weight = routing_weights[0, i].to(mindspore.float32) # no item, no precision loss
                expert_layer = self.experts[expert_idx]
                final_hidden_states += expert_layer(hidden_states).to(mindspore.float32).mul(weight)
            final_hidden_states = final_hidden_states.to(hidden_states.dtype)
```

#### 3.2 DeepSeek-MoE: decode优化

**Decode阶段**

```python
@no_grad()
    def moe_infer_decode(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = ops.zeros_like(x)
        for i in range(self.num_experts_per_tok):
            expert_id = flat_expert_indices[i].item()
            weight = flat_expert_weights[i].item()
            expert = self.experts[expert_id]
            expert_out = expert(x)
            expert_cache += expert_out * weight
        return expert_cache
```



---

## 优化效果分析

### Prefill阶段优化

| 优化项 | 技术手段 | 预估收益 |
|-------|---------|---------|
| FlashAttention | 硬件加速 | 40-60% |
| mint算子替换 | 底层优化 | 10-20% |


### Decode阶段优化

| 优化项 | 技术手段 | 预估收益 |
|-------|---------|---------|
| 分场景MoE策略 | 减少padding | 40-50% |




---

## 关键技术总结

1. **算子层优化**：替换mint算子，充分使能昇腾NPU加速计算。
2. **注意力优化**：集成FlashAttention，加速prefill阶段推理能力。
3. **MoE优化**：针对decode场景进行优化。

