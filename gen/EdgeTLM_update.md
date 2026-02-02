灵感来自于论文FlexOlmo，https://arxiv.org/abs/2507.07024，代码https://github.com/allenai/FlexOlmo。
A. FlexOlmo 论文核心思想总结

FlexOlmo  提出了一种新型语言模型架构，其核心目标是解决两大问题：
分布式训练 (Distributed Training)：允许不同的数据所有者（例如，拥有不同私有数据集的公司）在不共享数据的情况下，各自训练模型的不同参数。 
数据灵活性 (Data-Flexible Inference)：在推理时，可以根据需求（如数据许可、隐私要求）灵活地包含或排除某些专家模块，而无需重新训练。 



为了实现这一目标，FlexOlmo采用了以下关键技术：
MoE 架构与独立专家： 模型的主体是一个混合专家（MoE）架构 。但与传统MoE不同，它的每个专家（Expert，即FFN模块）都是独立训练的。
“公共锚点” (Public Anchor) 训练法： 这是FlexOlmo最核心的创新。为了让独立训练的专家最终能够协同工作，所有专家都基于一个共享的、冻结的“公共模型” (M_pub) 进行训练 。在训练单个专家 M_i 时，系统会构建一个临时的“双路MoE”，只包含M_pub（冻结）和M_i（可训练） 。所有专家都学会了与同一个M_pub锚点协调，从而间接地学会了彼此协调。 


“单行路由” (Domain-Informed Router)： 路由器的训练同样是分布式的。每个专家 M_i 都关联着一个单独的路由向量（即“单行”）r_i 。这个 r_i 是在训练 M_i 时，在上述的“双路MoE”中与 M_i 的参数一同被微调的 。它的初始值可以来自对该专家领域数据的嵌入平均值。 


免训练合并 (Joint-Training-Free Merging)： 训练完成后，最终的全局模型是通过简单地将所有独立训练的专家模块 M_i 和它们各自的路由行 r_i 拼接（concatenate） 在一起而形成的 。这个过程不需要任何昂贵的联合训练。

负偏置 (Negative Bias)： 论文明确提到，为了帮助模型在合并后更好地在多个专家（而不仅仅是M_pub）之间进行竞争，为每个专家引入了一个负偏置项 b_i 。这使得决策边界更接近专家数据点 ，确保专家只在有强信号时才被激活，从而更具选择性。 

B. FlexOlmo 对您项目架构的启发
FlexOlmo的思想几乎完美地映射到了您的项目需求上，并为您当前的架构演进提供了坚实的理论基础。

** justification for Removing HA-LoRA -> Adopting Frozen TLM-BASE: 您提到之前的HA-LoRA（硬件无关模块）“不稳定”。这很可能是因为它是一个可训练的共享模块，导致在与HS-LoRA（硬件专属模块）同时训练时产生冲突和“灾难性遗忘”。FlexOlmo的“公共锚点”  策略提供了完美的解决方案：将共享部分（即您的 TLM-BASE）完全冻结。TLM-BASE不再是一个需要学习的组件，而是变成了一个所有专家赖以协调的稳定锚点**，这彻底解决了不稳定的问题。



LoRA 即“专家” (LoRA as "Expert"): FlexOlmo将FFN层定义为专家 。您的设计则更进一步：将一个 LoRA 模块 视为一个“低秩专家”。这个见解非常精妙，因为LoRA本身就是对BASE的一种“修正”，这与FlexOlmo中专家M_i作为M_pub的“领域补充”的思想完全一致。


单行路由向量的引入 (Single-Row Routing): FlexOlmo的“单行路由”r_i  是您为每个LoRA配备一个路由向量的直接理论来源。您的 r_i 就像FlexOlmo的r_i一样，是LoRA专家的“语义指纹”，用于在推理时与输入x进行匹配度计算。

独立训练与即插即用 (Independent Training & Plug-and-Play):
 FlexOlmo的核心价值在于支持在封闭数据集上独立训练 。这正是您希望实现的目标：“每个人都可以用自己的数据训练一个lora+一行路由向量”。您的新架构完美地复刻了这一点。任何人都可以使用冻结的TLM-BASE作为锚点，在他们自己的（例如V100）硬件数据上独立训练出一个{LoRA_V100, r_V100, b_V100}专家单元，然后将其贡献给一个无需重新训练的、不断扩充的全局专家库。


偏置与门控机制 (Bias & Gating Mechanism): 
FlexOlmo的“双路MoE”训练 和“负偏置”b_i 启发了您的门控设计。您采用的“隐式base=0对比（Sigmoid门控）”是对FlexOlmo“双路Softmax”的一种更简洁、数学等价的实现。您采纳的b_i \le 0约束，更是直接应用了FlexOlmo的偏置思想 ，这是确保您的门控g(x)不会“无脑全开”的关键机制之一。

总结：您的新架构（Frozen TLM-BASE + Gated LoRA Experts）可以被视为FlexOlmo思想在“低秩适应（LoRA）”领域的一次精彩应用。您通过引入“稳定锚点”解决了HA-LoRA的不稳定性，通过引入“单行路由”和“门控机制”实现了与FlexOlmo同等的数据灵活性和可扩展性。

新思路：
> 本版替换旧版“HA+HS 双轨”实现指南；旧版可参考历史文件。

## 概述

- **目标**：在冻结的 `TLM-BASE` 上，以“可插拔 LoRA 专家 + 单行路由”实现模块化优化与可审计合并。  
- **亮点**：免联合训练、本地可训练并上架、Top-K 稀疏推理、随时退出/撤销。

## 目录结构（建议）

```
project/
  experts/
    <org>/<hw>/<tag>/
      adapter_model.bin
      adapter_config.json
      router.json           # { "r": [...], "b": -0.7, "tau": 2.0, "meta": {...} }
      metrics.json          # 可选
  src/
    gating.py               # GatedLoRAExpert, BasePlusExperts
    train_single_expert.py  # 单专家训练脚本（配对：BASE vs BASE+LoRA）
    infer_multi_experts.py  # 多专家 Top-K 推理
    latency_cache.py        # BaseLatencyCache + 抽查刷新
    proxy_score.py          # (可选) 代理收益模型
```

## 关键前向（训练期：单专家）

\[
y = y_{\text{base}} + g(x)\cdot \Delta y,\quad 
g(x)=\sigma\!\Big(\frac{r^\top x+b}{\tau}\Big),\ b=-\mathrm{softplus}(\beta)\le0
\]

- `y_base`：`base(x).detach()`（完全冻结基座）  
- `Δy`：LoRA 的输出增量  
- `r`：单行路由向量；`b`：负偏置；`τ`：温度（可退火/可学习）

### 推荐初始化
- LoRA：`B=0`、`A` Kaiming/Normal（初始增量为 0，前向稳定）  
- `r`：用硬件/领域嵌入的**均值**作起点  
- `b`：负值（如 `-1.0`）；`τ`：2.0 起步逐步退火到 1.0/0.5

## 训练循环（兼容你的 SFT 闭环）

```python
# train_single_expert.py
for step, batch in enumerate(loader):
    x, y_star = batch["x"], batch["y"]
    with torch.no_grad():
        y_base = base(x)               # 冻结
    delta = lora(x)
    s = (x @ r) + (-F.softplus(beta))  # b<=0
    g = torch.sigmoid(s / tau)
    y = y_base + g * delta

    loss = L_task(y, y_star)
    if use_gain:
        gain = (lat_base(batch) - lat_base_plus_lora(batch))  # 真机/代理/缓存+抽查
        loss = loss + λ_gain * F.relu(gain - margin)

    loss = loss + λ_r * (r.norm()**2) + λ_H * entropic_reg(g)
    loss.backward()
    opt.step(); opt.zero_grad()
```

- **仅优化**：LoRA(A/B)、`r`、`beta`（与可选的 `tau`）  
- **不优化**：任何 Base 参数（emb/attn/ffn/ln/head 全冻结）

## 合并与推理（中心侧：多专家 Top-K）

```python
# infer_multi_experts.py
@torch.no_grad()
def infer_topk(x, base, experts, K=2, tau=1.0, mask=None):
    yb = base(x)                         # 冻结
    scores, deltas, names = [], [], []
    for name, e in experts.items():
        if mask and not mask.get(name, True): continue
        b = -F.softplus(e.beta)
        s = (x @ e.r) + b                # [N]
        d = e.lora(x)                    # Δy
        scores.append(s); deltas.append(d); names.append(name)
    # BASE 的隐式 0 分通道
    S = torch.stack([torch.zeros_like(scores[0])] + scores, dim=0)  # [1+E,N]
    # Top-K 截断（忽略 BASE 这一行）
    topk_vals, topk_idx = torch.topk(torch.stack(scores), k=min(K, len(scores)), dim=0)
    mask_mat = torch.zeros_like(S, dtype=torch.bool)
    for i, idx in enumerate(topk_idx): mask_mat[1:,:,][(idx, torch.arange(S.size(1)))] = True
    S = torch.where(mask_mat, S, torch.full_like(S, -1e9))
    w = F.softmax(S / tau, dim=0)        # [1+E,N]
    y = yb + sum(w[i+1].unsqueeze(-1)*deltas[i] for i in range(len(deltas)))
    return y, {"weights": w, "names": ["BASE"]+names}
```

- **BASE 通道**的 0 分“兜底”保证未被专家解释的成分回落到 Base。  
- 支持 **mask** 精准退出/隔离专家（行级屏蔽）。

## BaseLatencyCache 与代理收益

- 为每类算子/张量句子缓存 **BASE 延迟**（按形状/算子签名键控）  
- 训练时计算 `gain = lat(BASE) - lat(BASE+LoRA)`：
  - 优先用**代理模型**估计（快速）
  - 每 N 步**抽查真机**校准（可靠）
  - `L_gain = λ·ReLU(gain - margin)` 只在“明显有益”时鼓励更大门控

## 指标与看板（最小集）

- 门控分布直方图（`g` 或 `w`）  
- 专家激活热力图（token/样本维度）  
- 端到端加速与回退率（BASE vs BASE+LoRA）  
- 专家健康度（贡献度、覆盖度、冲突率）

## 分发协议（产物规范）

```
experts/<org>/<hw>/<tag>/
  adapter_model.bin
  adapter_config.json
  router.json        # { "r": [...], "b": -0.7, "tau": 2.0, "meta": {...} }
  metrics.json       # { "train_loss": ..., "gain@p50": ..., "gain@p90": ... }
```

- 聚合端：扫描 → 加载 LoRA → 拼接路由矩阵 → 推理  
- 治理：按 org/hw/tag 白名单或黑名单，对路由矩阵行做屏蔽

## 迁移旧版（HA+HS）

- 删除 HA 分支；每个 HS → `{LoRA, r, b}` 配对训练；  
- 旧的“硬件路由表” → 专家屏蔽表（面向治理）；  
- 旧脚本的“联合前向” → 新脚本的“BASE + 门控增量”。

## 超参数建议（起点）

- `rank`: 8/16/32（依模型大小与预算）  
- `tau`: 2.0 → 1.0（线性退火）  
- `b`: 初值 -1.0（经 `-softplus` 参数化）  
- `λ_gain`: 0.1 ～ 0.5；`margin`: 3% ～ 8%（相对延迟提升）  
- `λ_r`: 1e-4；门控熵正则 `λ_H`: 1e-4  
- 优化器：AdamW，LoRA/路由分别可用不同 lr（如 1e-4 / 5e-4）

## 常见问题

- **Q：没有其他专家参与，路由能学起来吗？**  
  A：可以。我们把问题化成“BASE vs BASE+LoRA”两路对比，路由学“该开多大”，不需显式与他人竞争。

- **Q：冷启动会不会不稳定？**  
  A：不会。LoRA 初始增量为 0，前向=BASE；梯度通过连续门 \(g\) 直接到 LoRA，第一步就能学习。

- **Q：为什么要负偏置？**  
  A：让专家“更挑活儿”，只有当确实有收益时才显著打开；合并后稳定、可审计。


  阶段 1：冻结 BASE 并抽象专家接口

目的：把旧的 HA/HS 逻辑切到“Frozen TLM-BASE + 可插拔 LoRA 专家”的新接口，但不改变现有训练循环的其它部分。

新增 FrozenBaseWrapper

路径：src/python/tlm/modeling/frozen_base.py

要点：

base.eval()（禁用 dropout 等），param.requires_grad_(False)

forward() 返回 logits 或你习惯的中间隐表示；记得 .detach()

可选：暴露 get_router_feature(x)（例如取某层隐状态平均或硬件 embedding 作为路由输入 z）

# frozen_base.py
class FrozenBaseWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x, **kw):
        # 返回 logits（或你项目里约定的 “张量句子” 表示）
        return self.base(x, **kw)

    @torch.no_grad()
    def router_feature(self, hidden_states=None, hw_emb=None):
        # 任选一种。建议优先硬件/簇 embedding，其次某层隐状态均值
        if hw_emb is not None:
            return hw_emb
        assert hidden_states is not None
        return hidden_states.mean(dim=1)  # [B, H]


新增 专家接口与注册表

路径：src/python/tlm/modeling/experts/interface.py

内容：

GatedExpertMixin（定义 forward_delta(x) -> Δlogits、gate_inputs(...) -> z、gate_score(z) -> s、serialize/deserialize）

ExpertRegistry（register(name, expert) / load(path) / save(path)）

# experts/interface.py
class GatedExpertMixin(Protocol):
    def forward_delta(self, x, **kw): ...
    def gate_inputs(self, *, hidden_states=None, hw_emb=None): ...
    def gate_score(self, z): ...
    def serialize(self, path): ...
    @classmethod
    def deserialize(cls, path): ...

class ExpertRegistry:
    def __init__(self):
        self._experts = OrderedDict()
    def register(self, name: str, expert: GatedExpertMixin):
        self._experts[name] = expert
    def items(self):
        return list(self._experts.items())


训练脚本接入 FrozenBaseWrapper + ExpertRegistry

在 gen/MosLora/train_mt_moslora.py（或你当前入口）里，把原本直接调模型的地方改成：

y_base = frozen_base(x).detach()

暂时只挂一个“占位专家”（下一阶段实现真正的 LoRA 专家类）

先不动旧 HA/HS 的数据加载与日志，确保接口层变更可通过单测。

单元测试（最小）

tests/python/tlm/test_expert_registry.py：注册、保存、加载

tests/python/tlm/test_frozen_base.py：确认所有参数 requires_grad=False，forward() 无梯度，输出张量不参与反传。

阶段 2：实现 Gated LoRA 专家（单行路由 + 负偏置）

目的：把“BASE vs BASE+LoRA”的两路对比做成可训练的专家单元。

新增 GatedLoRAExpert

路径：src/python/tlm/modeling/experts/gated_lora.py

关键参数：LoRA(A/B)、r ∈ ℝ^h、beta（通过 b=-softplus(beta)≤0）、tau（温度）

初始化：B=0，A Kaiming/Normal；r 用硬件/领域 embedding 均值初始化；beta 使 b≈-1.0；tau=2.0

class GatedLoRAExpert(nn.Module, GatedExpertMixin):
    def __init__(self, lora_modules, r_dim: int, tau_init=2.0, b_init=-1.0):
        super().__init__()
        self.lora = lora_modules  # 你现有的 LoRA 封装
        self.r = nn.Parameter(torch.zeros(r_dim))       # 之后用均值向量覆盖
        self.beta = nn.Parameter(torch.tensor(0.))      # b = -softplus(beta)
        self.register_buffer("tau", torch.tensor(tau_init))
        with torch.no_grad():
            self.beta.copy_(torch.log(torch.exp(torch.tensor(-b_init)) - 1.))  # softplus^-1

    def forward_delta(self, x, **kw):
        return self.lora(x, **kw)  # Δlogits

    def gate_inputs(self, *, hidden_states=None, hw_emb=None):
        return hw_emb if hw_emb is not None else hidden_states.mean(dim=1)

    def gate_score(self, z):
        b = -F.softplus(self.beta)
        s = (z * self.r).sum(dim=-1) + b   # [B]
        return s / self.tau

    def serialize(self, path):
        # 保存 LoRA 权重 + r/b/tau
        ...

    @classmethod
    def deserialize(cls, path):
        ...


新增 汇总容器 BasePlusExperts

路径：src/python/tlm/modeling/base_plus_experts.py

训练期（单专家）：y = y_base + σ(s)*Δy

推理期（多专家）：Top-K + Softmax（第 0 通道是 BASE 的 0 分）

class BasePlusExperts(nn.Module):
    def __init__(self, base: FrozenBaseWrapper, registry: ExpertRegistry):
        super().__init__()
        self.base = base
        self.registry = registry

    def forward_single(self, x, *, hidden_states=None, hw_emb=None, expert_name:str):
        yb = self.base(x).detach()
        name, e = expert_name, dict(self.registry.items())[expert_name]
        z = e.gate_inputs(hidden_states=hidden_states, hw_emb=hw_emb)
        s = e.gate_score(z)                  # [B]
        g = torch.sigmoid(s)                 # [B]
        dy = e.forward_delta(x)
        return yb + g.unsqueeze(-1) * dy

    @torch.no_grad()
    def forward_multi(self, x, *, hidden_states=None, hw_emb=None, topk=2, tau=1.0, mask=None):
        yb = self.base(x)
        scores, deltas = [], []
        for name, e in self.registry.items():
            if mask and not mask.get(name, True): continue
            z = e.gate_inputs(hidden_states=hidden_states, hw_emb=hw_emb)
            s = e.gate_score(z)              # [B]
            d = e.forward_delta(x)           # Δlogits
            scores.append(s); deltas.append(d)
        S = torch.stack([torch.zeros_like(scores[0])] + scores, dim=0)  # 含 BASE 的 0 分
        idx = torch.topk(torch.stack(scores), k=min(topk, len(scores)), dim=0).indices
        mask_mat = torch.zeros_like(S, dtype=torch.bool)
        for i, idb in enumerate(idx):
            mask_mat[1:,:,][(idb, torch.arange(S.size(1)))] = True
        S = torch.where(mask_mat, S, torch.full_like(S, -1e9))
        W = F.softmax(S / tau, dim=0)        # [1+E, B]
        y = yb + sum(W[i+1].unsqueeze(-1) * deltas[i] for i in range(len(deltas)))
        return y, W

阶段 3：Loss 模块（含冷启动）

目的：把我们讨论的损失函数落成独立模块，便于训练脚本调用。

路径：src/python/tlm/training/losses.py

暴露：

compute_task_loss(logits, targets)（你现有的 SFT/CLM）

compute_gain_loss(I_pct, g_mean, step, warmup_steps, m_target, lambda_gain, eps=0.005)（冷启动屏蔽 + 退火门槛）

entropy_reg(g, lambda_H)、l2r_reg(r, lambda_r)

def compute_gain_loss(I_pct, g_mean, step, warmup_steps, m_target, lambda_gain, eps=0.005):
    M = ((I_pct >= eps) or (step >= warmup_steps))
    m_t = 0.0 if step < warmup_steps else anneal_linear(0.0, m_target, step-warmup_steps)
    return lambda_gain * float(M) * g_mean * F.relu(m_t - I_pct)

阶段 4：训练脚本改造（单专家本地训练）

目的：接入“全量真测”的收益，计算 I_%，按总损失训练 {LoRA, r, beta, (tau)}。

核心循环（伪代码）：

frozen = FrozenBaseWrapper(load_base(...))
reg = ExpertRegistry(); reg.register(args.name, GatedLoRAExpert(...))
system = BasePlusExperts(frozen, reg)

opt = AdamW([
    {"params": reg._experts[args.name].lora.parameters(), "lr": lr_lora},
    {"params": [reg._experts[args.name].r, reg._experts[args.name].beta], "lr": lr_router},
    # 可选：tau
], weight_decay=...)

for step, batch in enumerate(loader):
    x, y_star, hw_emb = batch["x"], batch["y"], batch["hw_emb"]
    # 1) 前向
    y_base = frozen(x).detach()
    delta  = reg._experts[args.name].forward_delta(x)
    z      = reg._experts[args.name].gate_inputs(hw_emb=hw_emb)  # or hidden_states
    s      = reg._experts[args.name].gate_score(z)
    g      = torch.sigmoid(s)
    y      = y_base + g.unsqueeze(-1) * delta

    # 2) 任务损失
    L_task = compute_task_loss(y, y_star)

    # 3) 你的“全量真测”结果（同批生成）
    # base_candidates, lora_candidates 都已实测
    lat_base_star = batch["lat_base_star"]      # min latency among BASE candidates
    lat_lora_star = batch["lat_lora_star"]      # min latency among BASE+LoRA candidates
    I_pct = (lat_base_star - lat_lora_star) / lat_base_star

    # 4) 门控收益（带冷启动）
    L_gain = compute_gain_loss(
        I_pct=I_pct.mean(), g_mean=g.mean(),
        step=step, warmup_steps=args.warmup,
        m_target=args.gain_margin, lambda_gain=args.lambda_gain
    )

    # 5) 正则
    L_reg = l2r_reg(reg._experts[args.name].r, args.lambda_r) \
          + entropy_reg(g, args.lambda_H)

    # 6) 总损失与反传
    L = L_task + L_gain + L_reg
    opt.zero_grad(); L.backward(); opt.step()


注意点

BASE 一定要 detach()（或整个 FrozenBaseWrapper.forward 用 @torch.no_grad()）

LoRA 初值：B=0 保证首轮前向稳定；梯度能通过 g*Δ 路径流到 LoRA

冷启动：warmup_steps 内将 L_gain 屏蔽或门槛置 0

超参起点：tau: 2.0→1.0 退火；b_init ≈ -1.0；λ_gain: 0.1~0.5；margin m: 3%~8%；λ_r=1e-4；λ_H=1e-4（前期负数鼓励高熵）

阶段 5：合并与推理（多专家）

目的：无训练合并；Top-K 稀疏路由；支持专家屏蔽/撤销。

路径：src/python/tlm/infer/infer_multi_experts.py

加载多个 {adapter_model.bin, adapter_config.json, router.json}，注册到 ExpertRegistry

前向调用 BasePlusExperts.forward_multi(x, hw_emb=...)

提供 CLI：

--experts_root、--topk、--mask "org/hw/tag:0"、--tau

输出每条样本的 weights（含 BASE 通道）、命中专家列表（便于审计）

阶段 6：专家产物规范与序列化

目录结构（与我们文档一致）：

experts/<org>/<hw>/<tag>/
  adapter_model.bin
  adapter_config.json
  router.json   # { "r": [...], "b": -0.7, "tau": 2.0, "meta": {...} }
  metrics.json  # { "train_loss": ..., "gain@p50": ..., "gain@p90": ... }


router.json 里保存 r（list）、b（标量）、tau（标量）、meta（硬件/许可证/作者等）

serialize/deserialize 对应落地

阶段 7：指标与看板（最小闭环）

训练期：

g 的分布（均值/直方图）

I_% 的分布与 L_gain 命中率（warmup 后）

推理期：

Top-K 权重热力图（含 BASE 通道）

专家激活频率 / 冲突率

端到端收益（BASE vs 多专家）

阶段 8：迁移旧 HS → 新专家

逐个 HS LoRA：

载入旧权重 → 包装成 GatedLoRAExpert

r 初始化为该簇/硬件的均值 embedding；b≈-1.0；tau=2.0

用你的“全量真测”+SFT 流，按阶段 4 做一轮配对训练（BASE vs BASE+LoRA）

导出 {adapter_*.bin, router.json}，注册到中心侧即可推理

你给的第 1 步清单，逐条反馈

FrozenBaseWrapper：✅ 正确；建议同时 eval() 与 requires_grad_(False)；forward 默认 @torch.no_grad()，或返回后 .detach()。

GatedExpertMixin & ExpertRegistry：✅ 很好；Mixin 里加上 gate_score、forward_delta 两个强约束方法。

训练脚本过渡：✅ 先通过接口跑通（哪怕专家暂时是 dummy），保留旧日志与评估，降低风险。

单测：✅ 必要；另外加一个“前向梯度流”测试：确认 y_base 无梯度，Δ 与 r/b 有梯度。

## TODO（代码实现）

- [ ] 调整 `gen_state.py` 日志输出，记录 Top-K 专家权重便于审计。
- [ ] 编写/更新单元测试（冻结基座、专家注册与序列化、门控前向梯度）。

## EdgeTLM 操作指南（2025-XX-XX）

### 1. 导出硬件专属训练语料

`prepare_edge_dataset.py` 会为样本注入硬件向量与基线延迟。**推荐使用 `for_gen_best` 产出的 `0_merge.json`**（可保留同一 workload 的多条 PPT 示范），并用 `--dedupe_mode keep_all` 维持“示范库随迭代增长”的特性：

```bash
# 只拥有 base 数据时：提供空的 lora 文件 + allow-missing-lora
touch /path/to/empty_lora.jsonl

# 4090（base → edge_sft）
python prepare_edge_dataset.py \
  --base-jsonl /path/to/0_merge.json \
  --lora-jsonl /path/to/empty_lora.jsonl \
  --output-jsonl /path/to/edge_sft_4090.jsonl \
  --merge_mode base_left \
  --merge_key repr \
  --dedupe_mode keep_all \
  --hardware-id 4090 \
  --embedding-json /path/to/hardware_embeddings_v4.json \
  --allow-missing-lora

# V100（同理）
python prepare_edge_dataset.py \
  --base-jsonl /path/to/0_merge.json \
  --lora-jsonl /path/to/empty_lora.jsonl \
  --output-jsonl /path/to/edge_sft_v100.jsonl \
  --merge_mode base_left \
  --merge_key repr \
  --dedupe_mode keep_all \
  --hardware-id v100 \
  --embedding-json /path/to/hardware_embeddings_v4.json \
  --allow-missing-lora
```

- `--merge_key repr` 实际使用 `(normalize(workload_repr), target_str)` 作为 key，**不会跨 target 混淆**，避免 shape 错配。
- `--dedupe_mode keep_all` 会输出**所有**示范记录，但 `lat_*_star` 采用同 key 的最小延迟（贴近原版“示范库不断增长”的行为）。
- 若只想保留每个 key 的单条最佳记录，可改 `--dedupe_mode min`。
- v4 embedding 内部使用 `nvidia/rtx-4090` 命名；脚本已做 4090 ↔ rtx-4090/a40 的兼容回退。

### 2. 训练单个 LoRA 专家

```bash
python train_edge_expert.py \
  --base-model-path /path/to/base_model \
  --tokenizer-path /path/to/tokenizer \
  --dataset-jsonl /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_sft_v100.jsonl \
  --output-dir /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_expert_v100 \
  --num-epochs 3 \
  --lambda-gain 0.0 \
  --warmup-steps 200
```

- 首轮训练建议 `lambda-gain=0` 或保留较长 `warmup_steps`，避免在缺失 `lat_lora_star` 的情况下错误驱动门控。
- 产物包括 LoRA 适配器、`router.json`、`metrics.json`，可直接放入专家目录 (`experts/<org>/<hw>/<tag>/` )。
- 也可使用一键脚本（支持迭代号/硬件/tag/超参覆盖）：`gen/scripts/run_train_edge_expert.sh`。
- 若要继续训练上一轮 LoRA，可传 `--init-expert-dir` 或设置 `EDGE_INIT_EXPERT_DIR` 指向上一轮专家目录。
- `run_train_edge_expert.sh` 默认在 `idx>0` 时回溯上一轮：
  - 若设置 `EDGE_PREV_EXPERT_TAG`，使用该 tag；
  - 否则若当前 tag 形如 `vN_gain`，自动回溯到 `v(N-1)_gain`（`v2_gain` 的上一轮为 `v1_init`）；
  - 否则默认 `v1_init`。
  如需关闭自动续训，设 `EDGE_RESUME_PREV=0`。
- 可用 `EDGE_LORA_GAIN_MARGIN` 覆盖 `gain_margin`（默认 0.05），用于控制 gain 何时开始生效。
- 为避免 `lat_base_star` 被历史 LoRA 测量抬高，`utils.json` 现在区分：
  - `measure_records_base`（只存 base 测量）
  - `measure_records_kv_lora`（只存 kv+LoRA 测量）
  - 仍保留 `measure_records`（历史兼容）
- `postprocess.py` 新增 `--record-mode base|kv_lora|all` 与 `--record-dir`；一键脚本已自动切换 base-only / lora-only，并生成 teacher(all)。

### 3. 确认 utils.json

`utils.py` 会优先读取环境变量 `TLM_DATA_ROOT`（默认 `/home/hehangshuai/workspace/tlm/tlm_dataset/gen`）。运行前请确保 `utils.json` 中记录了最新的测量路径；必要时参考 `run_iterative_postprocess.sh` 的写法手动追加。

### 4. 真机测量与增量训练

- 使用现有脚本（`gen_state.py` + postprocess）在目标硬件上生成新 schedule、采集 `lat_lora_star`。
- 重新运行 `prepare_edge_dataset.py` 追加 LoRA 延迟后，再次执行 `train_edge_expert.py`，逐步提升 `lambda-gain` 并调节温度。

## 实验操作建议

1. **整理基线数据**：针对 V100、Xavier、RTX4090、Xeon 分别运行 `prepare_edge_dataset.py`，生成带 `hw_emb` 与 `lat_base_star` 的 JSONL 文件。
2. **训练单专家**：使用 `train_edge_expert.py` 分别在四份 JSONL 上训练 LoRA 专家，首轮保持 `lambda-gain=0` 或较长 `warmup_steps`。
3. **整理专家目录**：将每个专家的 `adapter_model.bin`、`adapter_config.json`、`router.json`、`metrics.json` 放入 `experts/<org>/<hw>/<tag>/`，便于推理脚本读取。
4. **推理验证**：运行 `gen_state.py --edge_expert_dirs`，确认新的门控推理链路正常工作。
5. **真机测量**：按原流程执行生成与测量脚本，收集 LoRA 延迟并更新 `utils.json`。
6. **增量训练**：基于最新测量结果重新导出 JSONL，逐步提升 `lambda-gain`，进行后续迭代。




我们把每个 LoRA 的启用度建模为一个连续门控 
(z)∈[0,1]，它由“路由向量” 
 与当前条件 的匹配度经 Sigmoid/Softmax 得到。
训练阶段，
	​

 在“BASE vs BASE+LoRA”的对比监督（含真实收益 

）下，自动学成“这条 LoRA 在此条件下应当开启的强度”；
推理阶段，我们用
 排序并取 Top-K，即选择“最值得开启”的若干 LoRA，按其强度做加权合成。
这将“该不该开”与“开多大”统一在一个可微的连续门控里，既继承了 MoE 的选择性，又避免了硬开关的不稳定。

---

**硬件 OOV 与插值泛化方案（草案）**

- 背景问题（现状与痛点）
  - TLM 将“子图 ComputeDAG 文本头 + 目标硬件字符串（target）+ 调度步骤（张量语言）”拼接后做分词训练。遇到新硬件（例如未出现过的 CUDA 目标或 CPU 型号）时，target 字符串可能 OOV，被切成无意义的子词，导致 TLM-Base 在输入端就“读不懂硬件语义”。
  - 数据稀疏：当前仅有少数硬件（如 v100/4090/xavier/xeon）的样本；希望在不重训 Base 的前提下实现对新硬件的 Zero-shot/Few-shot 泛化。
  - 约束：模型输出仍必须是“张量语言”，不能改变范式或要求额外的自然语言解释。

- 目标（不改 Base，跨硬件泛化）
  - 冻结 TLM-Base，不改动其词表与 Transformer 参数。
  - 通过“连续硬件向量 → 可插值的嵌入注入”绕开 target 文本 OOV，使新硬件在 TLM 的输入空间中拥有稳定、可插值的语义表示。

- 两条协作思路（Teacher + Student）
  - 思路 A：桶化 Bucketing（Teacher 与兜底）
    - 把冗长且易变的 target 文本压缩成少量稳定“桶”词（例：<arch:Ampere> <warp:32> <bw:H>），用于：
      1) Teacher 路径的知识蒸馏监督（无 OOV、语义稳定）；
      2) 推理兜底：当新硬件明显落在已知硬件凸包之外时回退使用。
  - 思路 B：硬件 Embedding 注入 HW Embedding Injection（Student 与最终方案）
    - 在输入句子中，用特殊占位符 <HW> 替换原来的 target 文本；不依赖词表语义。
    - 用外部硬件向量 h 通过可插值的投影函数生成 e(h)，并将 e(h) 注入到 <HW> 的位置（inputs_embeds 覆盖）。
    - 投影函数采用 ProtoMix（原型混合）结构以保证“先验连续性”：
      - 设已知硬件原型向量 {h_k} 与可训练的“语义原型嵌入” {z_k}；
      - 相似度 α_k(h) = softmax(cos(h, h_k)/τ)；
      - e(h) = Σ_k α_k(h) · z_k（凸组合，天然可插值）。

- 训练策略（冻结 Base，蒸馏对齐 + 可选 CE）
  - 样本成对构造：
    - Teacher 文本：保持原 target 文本（或桶化后的稳定文本）；
    - Student 文本：相同句子但将 target 文本替换为单个 <HW>；同时提供硬件向量 h；
    - 均保留“ComputeDAG 文本头 + 步骤（张量语言）”的一致上下文。
  - 前向：
    - Teacher：冻结 Base，得到 logits_T（不反传）。
    - Student：冻结 Base，仅在 <HW> 位置注入 e(h)（由 ProtoMix 产生），得到 logits_S。
  - 损失：
    - 主：KD = KL(P_Teacher || P_Student)；
    - 辅：CE 到张量语言标签；正则：原型 z_k 的 L2、α 的熵正则、可选温度 τ 学习；
    - 插值蒸馏（可选）：对 h_λ = λ h_i + (1-λ) h_j，约束 P_S(h_λ) ≈ λ P_T(h_i) + (1-λ) P_T(h_j)，提升凸包内的平滑插值能力。

- 推理改造（与现有 gen_state 兼容）
  - 外部文件仍使用“记录版 JSON（i/r）”以便 TVM 复现与测量。
  - 在 gen_state 中，将“记录 → 句子前缀”的转换改为 Student 形式：
    - 删除/忽略原 target 文本，替换为单个 <HW>，构造 input_ids；
    - 由目标硬件向量 h 经过 ProtoMix 得到 e(h)，再用 inputs_embeds 覆盖 <HW> 位置；
    - 其余 token 使用 Base 的词嵌入；保持 attention_mask/eos 等参数一致；
    - 调用 model.generate(...) 生成后续步骤，最终仍写回记录版 JSON（便于测量与后处理）。
  - 兜底：当 α 的分布显示“凸包外”（如 α_max 过低/熵过高）时，回退到桶化 Teacher 文本推理或触发少量测量的 Few-shot 校准。

- 数据与实现落点（计划）
  - 数据集构造：
    - 保持现有 0_merge.json（Teacher 文本）不变；
    - 新增 Student 文本导出（同一条样本替换 target 为 <HW>），同时保存 hw_emb（来源于 Embedding/hardware_embeddings_v2.json 或 prepare_edge_dataset.py 导出的 JSONL）。
  - 代码路径（拟）：
    - gen/make_dataset.py 与 gen/make_dataset_utils.py：增加 Student 文本导出/加载分支；
    - gen/gen_state.py：增加 --hw_injection 与 --edge_embedding_path；在 gen_func 中走 inputs_embeds 路径注入 e(h)；
    - 新增 modeling/hw_injection.py：实现 ProtoMix（{z_k} 可训练、α_k(h) 由余弦相似/小线性 + softmax 产生，可选温度 τ）。
  - 兼容性：不改 Base 结构与输出范式；外部产物仍为记录版 JSON；与 Edge 专家与 LoRA 门控互不冲突。

- 验证与度量（建议）
  - 一致性（同硬件）：Teacher vs Student 的 KL 均值/分位，端到端生成/测量指标差异；
  - 插值实验：对 h_λ 的 KL/延迟曲线是否平滑；logits 的 Lipschitz 估计；
  - OOT 仿真：将某已知硬件当作“新硬件”隐藏 Teacher 文本，纯注入 Student 路径评估，再做 Few-shot 校准曲线；
  - 真新硬件：先 Zero-shot，凸包外检测后启用兜底/少量样本。

- 风险与边界
  - 新模型/新 TVM 版本引入新的 ComputeDAG 模板（新哈希）不影响流程：记录恢复依赖 TVM workload 注册表而非词表；
  - 极端外推硬件可能需要 Few-shot 校准；
  - inputs_embeds 路径需与 generate 的缓存/掩码参数仔细对齐（HF 支持，需在实现中单测）。

- 小结
  - 该方案在不重训 Base 的前提下，构建一条“连续、可插值”的硬件语义注入通路，解决 target 文本 OOV，引入 Teacher 蒸馏保证语义等价，并保留现有张量语言与 TVM 记录产物，适合作为 EdgeTLM 的跨硬件泛化基础能力。

---

**TLM 硬件泛化（OOV）解决方案：HwToken 注入（版本提案）**

- 项目目标与约束
  - 目标：当出现新硬件（其 target 文本 OOV）时，TLM‑Base 仍能理解“硬件语义”并生成正确的“张量语言”。
  - 约束：
    - 冻结 TLM‑Base，不能重训；
    - 兼容现有硬件向量资产 h（Embedding/hardware_embeddings_v2.json 等）；
    - 不改变输出范式（仍为张量语言）。

- 灵感来源（输入层注入 + 可训练对齐器）
  - CAN_LLM（ICLR'25）输入层拼接外部模态（GraphToken），与“冻结 Base”兼容。
  - MPnP（MobiCom'25）提出可训练对齐器（Aligner/投影器）将外部模态对齐到 LLM 可消费空间；其“深层注入（最后 N 层 KV 注入）”与本项目不兼容，暂不采用。

- 选型与总体思路
  - 注入位置：采用 CAN_LLM 的输入层注入（Prompt‑level），避免改动 Transformer 结构。
  - 绕开词表：不再喂入易 OOV 的硬件 target 文本，改为在句子中放置一个 <HW> 占位符（或沿用已有特殊标记如 [MASK]）。
  - 动态嵌入：<HW> 的向量不从 Base 词表取，而由“对齐器（Aligner）”基于外部硬件向量 h 动态生成，随后以 inputs_embeds 形式注入。
  - Aligner 方案：
    - ProtoMix（优先）：e(h)=Σ α_k(h)·z_k，α_k(h)=softmax(cos(h,h_k)/τ)。其中 h_k 为已知硬件原型，z_k 为可训练“语义原型嵌入”（维度=Base 词嵌入维度）。凸组合天然可插值，适合样本稀疏；可选温度/正则。
    - 备选/增强：线性投影 W·h 或 ProtoMix + 小残差（受正则约束），以防表达力不足。

- 训练策略（首选纯 CE，必要时再引入 KD）
  - 冻结 Base，仅训练对齐器（z_k、可选温度/小线性）。
  - 输入：
    - Student 文本：将原句中的硬件 target 文本替换为 <HW>，其余保持“ComputeDAG 文本头 + 步骤（张量语言）”不变；同时提供硬件向量 h。
  - 前向：
    - 用 Base.get_input_embeddings() 将 input_ids → embeds；定位 <HW> 位置，并以 e(h)=Aligner(h) 覆盖对应向量；作为 inputs_embeds 喂入 Base；
  - 损失：CrossEntropy 到张量语言标签（与现有 CLM/SFT 一致）；梯度仅更新对齐器参数。
  - 计划 B（可选）：若纯 CE 不稳，则引入 KD：Teacher（原/桶化 target 文本）与 Student（<HW> 注入）做 KL 蒸馏；Teacher 亦可用于凸包外兜底推理。

- 推理路径（与 gen_state 兼容）
  - 外部文件仍为“记录版 JSON（i/r）”，便于 TVM 复现/测量。
  - 在 gen_state 中的“记录 → 句子前缀”转换改为 Student 形态：
    - 把 target 文本位置替换为 <HW>；
    - 由目标硬件向量 h 经过对齐器得到 e(h)，以 inputs_embeds 注入；
    - 其余 token 使用 Base 的嵌入；保持 attention_mask/eos 参数一致；
    - 调用 model.generate(...) 生成后续步骤，最终仍写回记录版 JSON。

- 验证与风险管理
  - LOHO（留一硬件）：3 训 1 验，评估 Zero‑shot 与 Few‑shot；
  - 插值扫描：在两硬件间做 h_λ=λh1+(1−λ)h2，检查 e(h_λ) 的余弦平滑与模型 Loss/PPL 的平滑；
  - 增广与正则：对 h 加微噪声（邻域抖动）、对 z_k/LN/温度做正则，防过拟合 4 个硬件；
  - 兜底：α_max 过低/熵高判定“凸包外”，回退到桶化 Teacher 或触发少量测量校准。

- 实现落点（最小改动）
  - 数据：在现有 0_merge.json 基础上导出 Student 文本（target→<HW>）与 hw_emb；SFT 集可放入 edge_experts_sft_collections/<hw>/iterXX。
  - 代码：
    - 新增 modeling/hw_injection.py：ProtoMix 对齐器（{h_k},{z_k}, α_k(h), 温度 τ、可选 LN/正则）。
    - 训练脚本：train_hw_injection.py（冻结 Base，inputs_embeds 覆盖 <HW> 向量，纯 CE 训练）。
    - 推理改造：gen/gen_state.py 增加 --hw_injection 与 --edge_embedding_path，gen_func 中以 inputs_embeds 注入 e(h)。
  - 兼容性：不改 Base 结构、不改外部产物形态（记录版 JSON），与 Edge 专家/LoRA 门控独立。
• 本次改动

  - make_dataset_utils.py / make_dataset.py：  
    - 新增 `--emit_hw_student`、`--hw_token_placeholder`、`--hardware_embedding_path`，在生成 0_merge.json 时可额外输出 `text_student`、`hw_emb` 等字段；  
    - `json_to_token/input_to_tokens` 支持将 target 文本替换为占位符，并在启用 HwToken 占位时同时清空硬件参数数组，保证训练与推理侧 prompt 清洗方式一致。
  - modeling/hw_injection.py / modeling/__init__.py：  
    - 新增 ProtoMix 对齐器实现（包含构建原型矩阵的辅助函数），统一导出供训练与推理侧加载；  
    - 对齐器内增加原型正交正则（`get_ortho_loss`），鼓励语义原型在 embedding 空间中分散。
  - train_hw_injection.py：  
    - 初版实现冻结 TLM-Base、仅训练 ProtoMix 的 Hw 注入脚本（CLM CE + 硬件分类 + 正交正则），输出 `hw_aligner.pt`；  
    - 后续迭代引入 SupCon 风格对比学习与日志增强，实验结果表明在“Base 完全冻结”设定下，注入在 `gen_state+TVM` 链路中整体失败（全部 state invalid），主要只能学到“通用 GPU token + 略微区分 CPU”的粗粒度解，已在背景文档中记录为负例。  
  - train_hw_injection_lora.py（新）：  
    - 在冻结 Base 的前提下，为 Base 挂一套 LoRA，并与 ProtoMix 对齐器联合训练；  
    - 输入为 `text/text_student/hw_emb`，其中 `text_student` 已替换 target→HwToken 并去掉 8 个硬件参数整数；  
    - 损失组合为局部 LM CE（LoRA+Aligner）、硬件分类（`cls_head(hw_embed)`）、原型正交正则（`get_ortho_loss`），并保留对比学习（CTR）与蒸馏（KD）的接口（默认关闭），形成“LoRA+Aligner 适配新格式”的训练路径。  
  - gen/gen_state.py：  
    - 早期版本加入 `--hw_injection/--hw_aligner_path/--hw_token` 开关，在 worker 内加载 ProtoMix 并以 inputs_embeds 注入 HwToken，占位符文本仍由 `input_to_tokens` 构造；  
    - 在本次更新中，`input_to_tokens` 的 Hw 占位逻辑与 `text_student` 清洗保持一致（target→占位符 + 清空硬件参数数组），并通过 `prepare_hw_injection_context` 统一加载 `hw_aligner[+lora]`，确保推理 prompt 分布与训练侧一致。  
  - run_train_hw_injection_lora.sh：  
    - 新增运行脚本，使用多硬件 CLM 基座与 multi-tokenizer，对 `align_train_multi_merged/0_merge.json` 上的部分样本（可控 `sample_fraction`）进行 LoRA+ProtoMix 训练，输出 `{student（LoRA权重）, tokenizer, hw_aligner_lora.pt}`。

  后续可选操作

  1. 运行 `make_dataset.py ... --emit_hw_student --hw_token_placeholder [MASK]` 重新导出带 `text_student/hw_emb` 的 0_merge.json，并用 `postprocess_align_hw_mask.py` 确保硬件参数整数在 student 文本中被清洗掉。  
  2. 用 `train_hw_injection_lora.py` 在注入格式的 prompt 上训练 LoRA+ProtoMix，观察局部 LM/CLS/ORTH 的收敛情况，再结合 `debug_hw_aligner_mixture.py` 检查几何行为。  
  3. 在 `gen_state.py` 中，通过 `--hw_injection --hw_aligner_path ... --hw_token [MASK]` 尝试在新/未见硬件上启用 HwToken 注入，对比“原始 target 流 vs 注入+LoRA+ProtoMix”的行为与 TVM 合法性，逐步评估该方案在 EdgeTLM 全链路中的可用性与局限。

  11 月 26 日 update：
HwToken 注入方案阶段一：几何对齐 + 后层 LoRA（2025-11-26）

失败回顾（旧版 ProtoMix + 全层 LoRA）

早期版本采用 ProtoMix Aligner + 全 12 层 LoRA 的设计：在 text_student 中用 [MASK] [MASK] [MASK] [MASK] 替换原始硬件 target，再由 ProtoMix 从 hw_emb 生成注入向量，配合 LM / CLS / KD / ORTH 多任务训练。

实验现象：

即使在已见硬件（V100/4090/Jetson/CPU）上，gen_state.py --hw_injection 生成结果大量出现 “All states are invalid”，合法 schedule 近乎为 0；

debug_short_gen.py 在短程生成中出现 [SEP] 1 0 auto_unroll_max_step$512 ... 1 1 1 1 1 ... 等循环垃圾 token，说明 Base 的张量 DSL 语法已被破坏；

训练 log 中 LM 与 CLS 呈明显“跷跷板”：CLS 降、LM 就涨，反之亦然，KD 也难以稳定收敛。

事后分析：

LoRA 全层注入等价于“重写一个新的 TLM”，导致 Base 早期负责“合法性 + 语法”的低层表征被改坏；

Aligner 只有 LM/CLS/KD 的间接监督，且 KD 过早介入、Teacher/Student 输入分布差异较大，使对齐器学到的是“为了分辨硬件/CLS”而不是“生成原生硬件 embedding”；

8 个硬件整数仍暴露在 text_student 里，Student 可以“偷看数字”而不必真正利用 hw_emb，进一步削弱了 Aligner 的学习信号。

反思与设计原则

将任务拆成 “语法/合法性恢复” 与 “硬件 → 策略微调” 两个阶段，优先保证 Base 的 DSL 能力不被破坏；

让硬件语义主要通过 少量 HwToken 注入，而不是让 LoRA 去“强行记住硬件名字”；

给对齐器一个 显式几何锚点：它输出的 4 个 HwToken，要在 embedding 空间里尽量贴近 Base 过去看到的真实硬件子串表示，而不是只靠 LM/CLS 的间接约束；

LoRA 只负责在后几层“调策略”，而不参与早期对硬件字段的解析与句法建模。

新方法摘要（Stage-1 版本）

4 段 HwToken 语义重新定义

ARCH：架构身份（cuda -keys=... -arch=sm_xx 或 llvm -keys=cpu -mcpu=...）；

CONS：硬约束常数（以 -thread_warp_size 为锚点，之后的 8 个整数）；

MEM / HOST：当前版本先保守处理，JETSON 的 LLVM 段可视为 HOST，其余硬件视为 inactive，后续再逐步启用。

对齐器结构从 ProtoMix 简化为 4 个线性头

输入仍为全局 hw_emb（多维硬件特征）；

为 ARCH / MEM / CONS / HOST 各定义一个独立线性层，输出 4 个 d_model 维 HwToken 向量，在 text_student 中覆盖 [MASK] × 4 的 embedding；

线性对齐器对所有硬件共享，只在少数参数上学习“如何把通用硬件 embedding 投到 Base 的词嵌入空间”。

几何监督：预计算 Base 看到的“硬件子串目标”

新增 precompute_hw_token_targets.py，对每个硬件使用 规范化 target 串（只含硬件参数，不含张量 DSL），用 Base 的 embedding + mean pooling 得到：

E_ARCH_base(h)：从 cuda/llvm 开始到 CONS 段前的子串；

E_CONS_base(h)：以 -thread_warp_size 为锚点后紧随的 8 个整数；

E_MEM_base(h) / E_HOST_base(h)：当前仅对 Jetson 等有明确 Host 段的硬件启用，其余用 mask 跳过。

将上述结果保存为 hw_token_targets_v1.pt，训练时按 hw_name 查表。

在 train_hw_injection_lora.py 中为 4 个 HwToken 增加几何 loss：
总损失为 loss = lm + λ_geom * geom + λ_orth * orth + λ_cls * cls，其中 Stage-1 设 λ_geom≈1.0，λ_cls 极小，λ_kd=0。

LoRA 只挂在最后 4 层 MLP

通过模块名过滤仅在 transformer.h.8~11.mlp.{c_fc,c_proj} 上插 LoRA，前 8 层保持完全冻结；

这样 Base 的低层继续负责张量语言的合法性与基本调度语法，LoRA 只在高层调整不同硬件下的策略偏好。

训练与实验进度（Stage-1）

数据：使用 align_train_multi_merged_no_ints/0_merge.json，在 text_student 中用 [MASK]×4 替换 target，并彻底移除那 8 个硬件整数，避免 Student“偷看”原始约束；样本规模约 230 万条，多硬件混合。

训练配置：

LoRA：r=16, alpha=32, dropout=0.05，仅最后 4 层 MLP，约 49 万可训练参数；

loss：λ_lm=1.0, λ_geom≈1.0, λ_orth=0.1, λ_cls=0.01, λ_kd=0；

在线 debug_short_gen 使用 Stage-1 模型和新的 HwToken 注入路径，周期性抽样检查语法是否稳定。

当前观察：

总 loss 从 8.x 降至 3–4 区间，LM loss 虽仍有震荡，但不再出现明显“CLS 降 LM 飙”的极端对立，训练过程比旧方案稳；

下一步将以 gen_state.py --hw_injection 在已见硬件上系统评估合法率，并与 “无注入 + 纯 LoRA” 和 “原始 TLM-Base” 基线对比，确认 Stage-1 是否恢复了基本语法与合法性，然后再考虑 Stage-2 的轻量 KD / 簇级 CLS 设计。


11.30日
  当前做法：4 个 hw 向量，注入到 prompt 中的做法，通过多次检测，发现可能会
  - 4-MASK + patch 注入的推理实验显示：虽然能生成 token，但语法被严重破坏，模型倾向复读硬件参数/数字，TVM 全部 invalid。
  - 离线检查 aligner 输出几何位置正常（与 prototype/target 余弦高），问题集中在 4-MASK 句法/解码习惯：替换硬件子句后，模型把硬件串当成续写，合法 schedule 稀缺。
  - 已试 embedding patch + greedy 解码绕过 HF+PEFT inputs_embeds bug，生成不再为空但仍不合法，提示需要在训练/掩码/提示格式上进一步调整（如更强的 schedule 监督、首 token 约束等）。



12.2

EdgeTLM 当前问题与目标简要回顾

基础模型：OSDI’24 Tensor Language Model (TLM)，输入为「子图 + shape + TVM target 硬件串」，输出合法的 schedule DSL（SP/FSP/AN/RE/CHR…）。

现有能力：在训练过的硬件上，TLM-base 已经能稳定生成 TVM 可接受的 schedule（语法与性能都不错）。

核心痛点：新硬件（OOV target） 无法直接用原始 target 串表示，一旦把新硬件的信息直接塞进 prompt，容易出现：

词表 OOV / token 乱飞；

生成结果被硬件字符串污染（大量 cuda/llvm 参数出现在 schedule 段），TVM 全部 invalid。

我们的目标是：在 冻结 TLM-base 主体 的前提下，让模型能对未见过的硬件也生成语法合法、性能合理的 schedule。

旧方案回顾：4×[MASK] + hw_aligner 的问题

早期尝试：

做法：用 4 个 [MASK] 替换整段硬件 target，训练一个 hw_aligner + LoRA 把连续硬件向量映射到这 4 个位置的 embedding，然后再交给 TLM-base 继续生成。

问题：

aligner 很快学成了「硬件 token 模拟器」，输出几乎完全落在硬件参数的 embedding 子空间；

冻结的 TLM-base 会把这 4 个位置当作「强硬件提示」，继续在后续输出中补齐各种硬件串；

结果 schedule 段被严重“硬件化”，TVM 侧全部 invalid。

结论：直接用“覆盖硬件串 + 强行注入 embedding”的方式，会与 TLM 的预训练模式正面冲突，不适合作为长期方案。

新方案概览：Bucket + KV Aligner（类 MPnP 设计）

新的思路参考了多模态系统（例如 MPnP）中 “side-channel KV 对齐” 的做法，把问题拆成两条通道：

文本通道（prompt）只保留粗粒度信息：硬件类型 bucket

在 tokenizer 中新增少量 bucket token，例如：

[HW_GPU_HPC]：高性能 GPU（V100/A40/4090/A100…）

[HW_GPU_EDGE]：边缘 GPU（Jetson 等）

[HW_CPU_X86]：服务器 CPU

[HW_CPU_ARM]：ARM SoC

对训练语料：

text 字段保留完整的 canonical 硬件 target 串（给 teacher / baseline 用）；

text_student 字段对容易 OOV 的字段做桶化，例如：

-arch=sm_86 → -arch=[HW_GPU_HPC]

其它关键字段如 cuda -keys=cuda,gpu、-max_num_threads=1024 暂时保留不动。

这样，student 看到的是“带 bucket 的张量句子”，仍然有清晰的 CUDA/CPU 语法，只是 arch 这类 OOV 值被抽象成少数几类符号。

硬件信息主通道：KV side-channel + 对齐器

为每个硬件准备一个连续向量 hw_emb（由我们自己的硬件特征/聚类生成）。

设计一个 HwKVAligner：

输入：hw_emb ∈ R^d_hw；

输出：若干个 K_hw, V_hw 向量（可以看作 T_hw 个“硬件 memory slot”）；

在 TLM 后若干层的 self-attention 中，把这些 K_hw, V_hw 拼接到文本 K/V 后面，形成：

Q：仍然来自文本 hidden states；

K = [K_text; K_hw]，V = [V_text; V_hw]，并通过可学习 gate 控制硬件 KV 的影响强度。

训练时仅更新 HwKVAligner + 少量深层 LoRA（以及 bucket embedding），TLM-base 主体保持冻结。

直观理解：

bucket 负责在文本侧告诉模型“这是一类什么硬件”（HPC GPU / EDGE GPU / CPU）；

hw_emb + KV-aligner 在注意力空间中注入细粒度硬件差异（如 sm_70 vs sm_86、内存/warp 规模等）；

schedule 的语法和主干预测模式仍然由原始 TLM 负责，不被大幅扰动。

分阶段计划（简略版）

Phase 0：Canonical 化硬件 target（填平语法坑）

为 V100/A40/4090/Jetson/CPU 定义统一的 canonical target 模板；

修改 make_dataset / gen_state，保证所有新生成的数据都使用 canonical target；

对旧 4090 等“奇葩格式”样本做离线修补，验证 TLM-base 在 canonical 格式下 invalid 率仍然正常。

Phase 1：扩展 bucket 词表 + 构造 bucket 版训练数据

扩展 tokenizer / 模型词表，加入 [HW_*] bucket token，并用原始 arch token embedding 的平均值进行初始化；

为每条样本保留：

text：完整 canonical target（teacher 视角）；

text_student：仅将 -arch= 等字段替换为对应 [HW_*]，其它字段保留（student 视角）；

hw_emb：连续硬件表示，供 KV-aligner 使用。

Stage 0：仅微调 bucket embedding（小规模 CLM/KD 预热）

在 student 复制的模型上，使用 text_student 做一轮小规模 CLM/KD；

只更新新加的 bucket embedding（及对应 lm_head 行），冻结其它参数；

目标是让模型在“bucket 句子”输入下，行为尽量接近原始 TLM。

Stage 1：训练 KV-aligner + 深层 LoRA

在 Stage0 的 bucket-base 上挂上 HwKVAligner 和少量深层 LoRA；

输入：text_student + hw_emb，teacher 仍然是原始 TLM（输入 text）；

Loss 主要是 schedule 段的 LM CE + KD；

只训练 KV-aligner、LoRA 和少量 bucket embedding，使模型在保持语法稳定的前提下，学会利用连续硬件信息进行 OOV 硬件迁移。

这样分层之后，整个系统的职责划分更清晰：

TLM-base：负责张量语言与 schedule 语法（尽量不动）；

bucket token：解决 OOV、提供硬件类型级别的离散提示；

KV-aligner + LoRA：桥接连续硬件向量与 TLM 的内部注意力空间，驱动新硬件上的 schedule 迁移与微调。

[2025-12] Bucket 化硬件词表 + 数据构造链路重构（Phase 0.5）

这一阶段，我们对 TLM 的数据构造链路做了一次“分层重构”，目的是：

保证 TVM / AutoScheduler 侧始终使用 canonical 的硬件 target 字符串；

只在 TLM 文本通道 上引入少量离散的“硬件桶（bucket）控制词”，为后续 KV side-channel + LoRA 专家路由做准备。

具体修改如下：

扩展 tokenizer / 模型词表，引入硬件 bucket token

在原有 gen_tokenizer_multi_v1 的基础上，我们新增了 6 个 bucket token，并用已有硬件 token 的 embedding 做初始化：

[HW_GPU_HPC]：高性能 GPU（V100 / A40 / 4090 等），初始化为 -arch=sm_70 和 -arch=sm_86 embedding 的平均；

[HW_GPU_EDGE]：边缘/嵌入式 GPU（Jetson Xavier 等），初始化为 -arch=sm_72 embedding；

[HW_CPU_X86]：x86 服务器 CPU（Xeon），初始化为 -mcpu=skylake-avx512 embedding；

[HW_CPU_ARM]：ARM SoC（Jetson host / Raspberry Pi 等），初始化为 -mcpu=carmel embedding；

[MODEL_CPU_SERVER]、[MODEL_GPU_CONSUMER]：预留给后续的 model 级 bucket。

生成的新 tokenizer / 模型保存在：

gen_tokenizer_multi_v1_bucket

clm_gen_multi_v1_bucket_init

统一 canonical target 语法，修正 4090 异常格式

在 make_dataset.py 的 main() 中，我们为 4090 和 multi 平台补上了一套统一的 canonical target 模板：

V100 / A40 / 4090：统一为
cuda -keys=cuda,gpu -arch=sm_xx -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32

Xeon：统一为
llvm -mcpu=skylake-avx512 -model=xeon

ARM/Jetson/Raspberry Pi 等后续硬件也会遵循类似规则（mcpu 统一 bucket 为 ARM）。

所有 TVM 的 tvm.target.Target 对象都从这些 canonical 字符串构造，并在构造时去掉末尾重复的整数约束段，确保 AutoScheduler / TVM 解析稳定。

引入 canonical_to_bucket：只在构造张量句子时做 bucket 重写

在 make_dataset.py 中新增：

canonical_to_bucket(target_str: str) -> str：
只基于 canonical target 字符串中的 -arch / -mcpu 字段，做最小替换：

sm_70 / sm_86 → [HW_GPU_HPC]

sm_72 → [HW_GPU_EDGE]

skylake-avx512 → [HW_CPU_X86]

carmel → [HW_CPU_ARM]

input_to_tokens(..., use_bucket: bool = False)：
在将 task + state 转成 [compute_dag, json_line_i] 这种“张量句子”结构时：

先用 str(task.target) 拿到 canonical target；

若 use_bucket=True，则调用 canonical_to_bucket 把 arch/mcpu 替换成 bucket token；

然后再把这个 target 写回 json_line_i[0][1]，只影响喂给 tokenizer 的文本，不影响 TVM 内部存储的 task.target。

这意味着：

AutoScheduler 的 JSON 记录和 TVM 解析的 target 始终是 canonical 形式；

TLM 的输入句子可以在需要时看到 bucket 化后的 target 片段，例如：
cuda -keys=cuda,gpu [HW_GPU_HPC] -max_num_threads=1024 ...；

“硬件词表抽象（bucket）”被明确地限制在 TLM 文本通道这一层，不再反向污染 TVM。

后续工作（展望）

针对现有 4 个硬件，canonical_to_bucket 已经可以稳定地产生 bucket 化张量句子，用于后续的 bucket-base CLM 小微调；

对于未来新硬件（树莓派 / 新 GPU 等），我们计划在硬件 embedding JSON 中维护 hardware_name → bucket 的显式映射，并通过一个 build_bucket_target(hw_bucket) 的 helper 直接生成 bucket 版 target 文本：

TVM 仍然吃 canonical target；

TLM 看到的则是 [HW_GPU_HPC] / [HW_CPU_ARM] 级别的抽象 + KV side-channel 注入的连续硬件 embedding。

这套改造为下一阶段的 “bucket prompt + KV-aligner + 深层 LoRA 专家路由” 做好了铺垫，同时也保证了原始 TLM / TVM pipeline 的行为尽量保持不变，便于对比和回滚。

[2025-12-11] Phase 1 落地：实现 HwKVAligner 硬件旁路注入
1. 背景与动机
在 Phase 0.5 中，我们确立了 "Text Channel (Bucket) + KV Side-Channel (Aligner)" 的双通道架构。此前的实验表明，直接在 Prompt 中进行 Token 级的硬件嵌入注入（4x[MASK] 替换）会严重破坏 TLM-Base 的张量语言（DSL）语法，导致 TVM Invalid 率飙升。

本次更新旨在打通 KV Side-Channel，即通过在 Attention 层深处注入硬件信息，避免污染 Prompt 的句法结构，从而在保持 Base 语法稳定性的前提下，实现对新硬件特性的细粒度适配。

2. 核心实现内容
A. 硬件 KV 对齐器 (gen/hw_kv_aligner.py)

架构参考：复用了 mPnP (LiDAR) 的设计范式，实现了 HwKVAligner 模块。

数据流：

输入：24 维连续硬件向量（来自 hardware_embedding_generator_v4）。

投影：通过 MLP 将 24 维向量映射为 num_slots (默认 4) 个 Token 的 KV 形态。

注入机制：仅在模型的最后 backward_depth (默认 4) 层生效。通过可学习的 linker_weights (Softmax 归一化) 控制每一层注入的强度。

输出：生成兼容 HuggingFace 格式的 past_key_values，形状为 [2, B*beams, H, T_hw, d_head]，支持 Beam Search 复制。

B. 推理链路集成 (gen/gen_state.py)

解耦设计：改造了 prepare_hw_injection_context 和 gen_func。现在的推理链路支持解耦配置：

仅使用 Bucket Token（文本通道）；

仅使用 ProtoMix（旧 Token 注入，保留兼容性）；

[新增] 仅使用 KV Injection（硬件旁路）；

混合模式。

无感接入：在 gen_func 中，若检测到 kv_aligner 存在，会自动构造 past_key_values 并传入 model.generate()。对于 Base 模型而言，这相当于“预先看到”了一段不存在于文本中的硬件上下文。

C. 训练骨架 (train_hw_kv_aligner.py)

建立了一个专门的训练脚本骨架，确立了 Frozen Base + Trainable Aligner 的训练模式。

目前已打通 Forward/Backward 梯度链路，验证了 past_key_values 注入的可微性。

3. 解决的问题
语法保护：Prompt 仍然保持纯净的张量语言格式（仅含 Bucket Token），规避了 Token 注入导致的 DSL 语法崩坏问题。

细粒度适配：Bucket Token ([HW_GPU_HPC]) 只能提供粗粒度的类别信息，而 HwKVAligner 引入的连续向量（如 sm_86 vs sm_89 的差异）通过 Cross-Attention 机制在深层影响生成策略，弥补了 Bucket 的精度损失。

4. 下一步计划
联调训练：将 train_hw_kv_aligner.py 接入真实的 EdgeTLM 数据集，使用 lat_lora_star 或 Schedule 合法性作为监督信号。

超参探索：验证 num_slots=4 和 backward_depth=4 在 Tensor Program 生成任务上的最佳配置。

LoRA 协同：在 KV Aligner 跑通后，尝试与深层 LoRA (Last-N layers) 联合训练，完成 Phase 1 的最终形态。

KV 注入通路实现与调试记录（建议加入 update.md）
0. 目标与基本设定

目标： 在不改动/不重新训练 TLM-Base 的前提下，让模型在输入中接收“硬件条件”（尤其是新硬件/OOV 硬件），并能生成合法 schedule。我们采用两段式策略：

Bucket（离散化硬件 token）：把硬件字符串映射到有限桶（如 [HW_GPU_HPC]），保证 tokenizer 不 OOV。

KV 注入（连续硬件 embedding side-channel）：把硬件 embedding 通过一个小网络（HwKVAligner）映射成若干个 “prefix slots”的 past_key_values，在推理时注入到模型的 attention cache 中，从而提供更细粒度的硬件条件。

最终希望形成 “bucket + KV” 的 base 通路；后续再叠加 LoRA 路由 承接“优化知识复用/迁移”。

1. 推理侧实现里程碑：gen_state_debug_kv.py

我们将推理脚本扩展为支持：

--use_bucket：使用 bucket token 替换硬件字符串；

--use_hw_kv / --hw_kv_mode {noop, zero, real}：

noop：完全不走 KV 分支（baseline）

zero：注入全 0 的 KV（用于隔离“位置/缓存/实现”问题）

real：加载 HwKVAligner.pt + hardware_embeddings.json 注入真实 KV

--hw_kv_num_slots N：prefix slots 数（常见 1/2/4）

--pos_compensate：对 prefix_len 引入的“位置偏移”做补偿（关键）

多个调试开关：--debug_forward_trace / --debug_manual_greedy_steps / --debug_logits_compare / --debug_kv_stats / --debug_prefill_equiv_kv 等

同时实现了 KV 推理的核心逻辑：

prefill + last-token generate：先把 prompt 的前缀做一次 prefill（结合 HW past），再拿最后一个 token 进入逐步生成，确保与 HF generate 的 cache 机制兼容。

我们验证过：在 greedy 模式下，“直接 generate” 与 “prefill + last-token generate” 输出一致，说明这段拆分本身不是 bug 源头。

2. 早期现象：KV 一开就掉 valid（甚至 zero KV 也掉）

最典型症状：

noop 模式：9/9 valid

zero 或 real KV：经常掉到 4/9 valid（或 slot=2 时略有回升，例如 6/9）

输出形式“看起来格式正确”，但数值约束（如 SPC 因子乘积 == 轴长度）不满足，被 TVM 判 invalid。

这一步的重要结论是：不是 TVM 校验“太严格”，而是模型生成轨迹被扰动，导致数值更容易跑飞；尤其“zero KV 也掉 valid”强烈暗示问题不在 aligner 输出内容，而在KV 注入带来的位置/缓存处理。

3. 排除项与关键定位过程（我们做过的实验链）
3.1 不是 EOS / 不是长度预算不足

加了日志检查 new_tokens/hit_limit/ended_with_eos，发现“短输出”主要是 max_new_tokens 截断（比如被内部传入的 34 卡死），并非 EOS 提前结束。

增大 max_new_tokens 后输出会变长，但 invalid 仍会出现 → “短”不是根因。

同时我们也澄清了一个误会：模型生成段（比如 34 tokens）并不等于最终 schedule 的全部，SketchPolicy 会基于 state 做后续扩展，最终 JSON 里会出现大量 FU/AN/SP/PRS/PR 等步骤。

3.2 不是 “prefill 拆分” 的逻辑 bug

增加 [PREFILL-EQUIV-KV] 对比：KV 模式下 full-forward 与 split-prefill 的 top1 一致 → prefill 拆分不是主要问题来源。

3.3 核心定位：HF generate 的 position_ids 管理在 KV 场景下失效

我们在 model.forward 入口加 trace，观察到：

prefill 阶段显式传入了 position_ids（0..136）；

进入 generate loop 后，position_ids 变成 None，模型内部按 past_len 自行重算位置；

在 past_len>0（prefix_len=4）的情况下，这会把后续 token 的位置推进到错误的坐标系里，导致自回归轨迹逐步偏移，最终更容易生成 invalid 值。

这一点还被“手写 greedy 对照实验”进一步证实：

slots=4 + zero KV：手写 greedy 30 步，在第 14 步开始分叉

slots=0：30 步完全 match

slots=4 + pos_compensate(强制贯穿)：手写 greedy 30 步完全 match

=> 结论：问题主要来自“HF generate 的后续步 position_ids 未按我们期望推进”，而不是 KV 本身必然扰乱 attention。

4. 最终修复：将位置补偿贯穿到 generate 的每一步

我们对推理侧做了关键补丁：在 generate loop 的每一步都显式提供正确的 position_ids（让它从 prompt_len 开始递增，而不是让 HF 根据 past_len 自行推断）。

修复后的 forward trace 典型表现：

prefill：position_ids = 0..136，past_len=4

后续每步：position_ids = 137/138/139/...（不再是 None），attention_mask 与 past_len 同步增长

修复效果：

--hw_kv_mode zero/real + --pos_compensate：恢复到 9/9 valid

manual greedy 回归：30 步 match

这说明 KV 注入通路在“实现层面”已经被打通（cache/position 对齐正确，不再因为框架位置管理导致 invalid）。

5. 训练侧问题：HwKVAligner 曾出现“输出塌缩为全 0”

在推理脚本中加入 --debug_kv_stats 后观察到：

HW-VEC（4090/v100 embedding）是非零且有区分度

但 KV-STATS raw/post 为 全 0（nonzero=0/…），说明 aligner 输出本身塌缩成 noop

这解释了“real KV 注入效果看起来不明显”：通路通了，但注入内容可能是 0。

随后我们调整训练超参并观察到两种阶段：

阶段 A（容易塌缩）：较强正则/较小 kv_scale 初期会把 K/V 均值逐步压回 0

阶段 B（不塌缩）：关闭 l2_reg、提高 kv_scale_init、取消 warmup、放大 kv_scale_max、使用更合适的 scheduler 后，K/V 均值不再趋零，甚至逐步增大，训练是“有效注入”的

这里的关键经验是：“loss 很低”不等价于“注入学到了东西”，必须把 KV-STATS(raw/post) 作为一等公民指标加入训练/评估。

6. 本阶段我们“确定了什么”

Bucket + KV 注入的工程通路已打通：KV 注入不会天然导致 invalid；之前掉 valid 是位置/缓存管理问题（已修复）。

HF generate 在 past 注入场景下必须显式管理 position_ids：否则后续步 position 会漂移，导致自回归轨迹偏移并引发 invalid。

aligner 训练确实存在塌缩风险：需要专门的监控指标（KV stats、梯度 stats、kv_scale 动态）与合适的超参/正则策略。

当前系统距离“端到端迁移优化”只剩下两件事：

aligner 稳定地产生非零且有区分度的 KV（并在端到端评测中带来收益）

路由 LoRA（承接不同硬件/子空间的优化知识复用）

7. 代码更新清单（建议写进 update.md 的“改动记录”）

推理脚本：gen_state_debug_kv.py

新增 KV 注入模式（noop/zero/real）、slots、pos_compensate

新增调试：token budget、stop reason、forward trace、manual greedy、prefill equiv、kv stats、logits compare

关键补丁：generate loop 显式 position_ids 推进（修复 past 场景错位）

训练脚本：train_hw_kv_aligner.py

更安全的 label mask（找不到 schedule 起点就整行 -100）

batch 级过滤（全 -100 样本/全无有效样本则跳过）

KD mismatch 统计与阈值控制（超过阈值跳过该 batch 的 KD）

训练监控：kv mean/max、kv_scale、梯度统计等

工具/脚本：

check_schedule_kd_sanity.py：验证 schedule 段是否为 0、是否 full、student/teacher 长度 mismatch 等

watch_and_eval_hwkv_ckpts.sh：自动观察/评测不同 checkpoint 的生成/valid/延迟等（用于持续回归）

下一步建议（KV 线）

训练继续跑到 70k 合理（你已经观测到“从塌缩 → 不塌缩”的稳定态），但务必做两类回归：

每隔 N steps：debug_kv_stats（raw/post 的非零率、mean/max）

每隔 N steps：固定 workload 的端到端生成 valid + 简单 latency 指标（确认“非零 KV”带来可感知差异，而不是只变大但无收益）

端到端验证建议至少分三档对照：

bucket only（无 KV）

bucket + KV-zero（验证实现/位置无副作用）

bucket + KV-real（验证真实收益）

8. 端到端系统集成（KV + LoRA 路由 + 草图自动选择）

新增脚本：`gen_state_kv_lora.py`

目标：
- 将 bucket + KV 注入 与 LoRA 路由推理串起来，形成可执行的端到端通路；
- 支持用目标硬件 embedding 自动选择“最相近可用草图硬件”；
- 支持多 LoRA 专家路由（Top-K）。

实现要点：
- `--edge_expert_dirs` 支持多个 LoRA 专家目录；路由向量来自 `router.json`。
- `--edge_embedding_path` 默认 v4，但旧专家 `router.json.hardware_dim=29` 时需显式传 v2（否则会报维度不匹配）。
- `{hw}` 形式的 `--sketch_path` 将触发“自动草图硬件选择”；可用 `--sketch_hw_candidates` 限定候选集合。

已知问题（待修复）：
- 4090 v2 数据与 v1 数据对齐失败：`lat_lora_star` 与 `lat_base_star` 无法对应同一 workload。
  当前 `prepare_edge_dataset.py` 在 JSONL merge 模式下只用 `(workload_id, target_str)` 作为 key，
  但 `workload_id` 不含 shape，导致不同 shape 的样本被错误对齐。
  修复思路：将 key 升级为 `workload_repr`（包含 shapes），或直接用 `line`/`workload_repr+target` 作为对齐键，
  并用同一批 sketch/测量记录生成 v2 数据。

9. 泛化硬件评测脚本（2080Ti / 3090）

新增：
- `for_gen_eval_sketch_ansor`：仅生成 **bert_base / resnet_50 / mobilenet_v2** 标准形状的评测草图。
- `gen/scripts/run_eval_suite.sh`：一键生成四种 baseline（mix/official/4090-only/v100-only）。

说明：
- 我们的方法默认 **bucket + KV 注入 + LoRA**；官方 baseline 使用 `gen_state.py`（无 bucket/KV/LoRA）。
- 生成产物默认落到 `$RUN_ROOT/<hw>/iter00/eval_ansor_gen/`，需要手动搬运到测量服务器。

1.14、最新进展（以总结完的讨论成果为例）：
1) 当前进展：系统已经打通，开始进入“实验与对比”阶段

学生已经完成的关键进展：

已经把整个 pipeline 跑通：从测量数据 → 训练 LoRA（专家）→ 混合专家推理（LoRA-mix）→ 在目标硬件上生成 schedule 并测量。

目前已经有两个训练好的 LoRA 专家（锚点硬件）：

4090 LoRA

V100 LoRA

为了验证“对新硬件的泛化”，学生目前选择了能租到的新硬件作为测试目标：

3090

2080 Ti

当前实验主要看：在新硬件上，使用 base+单LoRA 或 LoRA-mix（混合专家） 能否超过（或接近）TLM 官方模型的 performance。

2) 当前观察到的实验现象（结果不太乐观，但有信息量）
2.1 2080 Ti 上的泛化效果

测了 3 个网络（学生举例提到：BERT-base / MobileNet / ResNet 等）。

以 TLM 官方模型在该硬件上的优化效果 作为 1.0 baseline（normalized）。

结果：混合专家 LoRA-mix 能比单LoRA稍好，但整体仍没明显超过官方模型。

在 BERT-base 上约到 0.96（提升约 9%但仍低于 1.0）

在 MobileNet / 另一个网络上更差（0.95 / 0.88 级别）

2.2 3090 上的泛化效果

同样 3 个网络。

结果类似：LoRA-mix 比单LoRA略好，但提升幅度不大，依然没超过官方 baseline。

2.3 回到“训练锚点硬件”4090 上检查（应当更强）

在 4090 上，你们的 LoRA 效果整体是比 base 强，且 LoRA-mix 进一步略有提升；

但出现一个“离群”现象：BERT-base 的某个总体指标看起来远超 base + 远超官方模型（高出 40%+ 级别）。

学生的解释/猜测：BERT-base 由很多子图组成（比如 9 张子图），其中某些 关键子图（例如 matmul）会被调用 20~30 次，如果这张关键子图被优化得特别好（比如达到 1.59x），会把端到端的总体指标显著抬高，导致“离群看起来特别夸张”。

3) 学生提出的三类“原因分析”（为什么泛化没超过 TLM 官方）

学生把原因归为三条主线（讨论中多次出现）：

原因 A：TLM 官方模型在“相近高端 GPU”上本来就不差

学生最初设想：TLM 官方模型只在 V100 上训，泛化到别的硬件会很差；

但现在发现：对 3090/4090/2080Ti 这种相近 GPU，官方模型即使不“认识硬件”，也可能生成“差不多的一套 schedule”，性能仍然比较强。

老师进一步强调：这可能不是“模型泛化更强”，而是最终生成的优化产物在这些相近 GPU 上本来就大差不差（因此你很难在这类硬件上用“泛化优势”打赢它）。

原因 B：你们自己的 LoRA 还没收敛（测量次数不足）

学生拿 TLM 论文的收敛定义做对比：论文里提到“测量每新增 2 万条，如果收益 <1% 持续 2 万条，则视为收敛”（学生用这话解释 Figure 9 里的那套 convergence/Profit 判据）。

对照你们自己的收敛图：

迭代到第 22 代总测量约 14 万次，但离 TLM 论文典型收敛量级（约 21 万）还有差距；

在最近 3 万条测量里仍能看到 ~8% 级别的提升趋势，说明还在下降中、还没稳定。

结论：你们现在可能还在“没跑够”的阶段，性能上还没到极限。

原因 C：混合专家路由“分不清硬件”，权重几乎不随硬件变

学生发现一个非常关键的现象：

用 LoRA-mix 给 4090 做推理时，权重约：4090 专家 ~0.54，V100 专家 ~0.38，base ~0.08

用 LoRA-mix 给 V100 做推理时，权重居然也差不多是这个比例（几乎不变）

学生解释：这来自你们当前的 embedding 区分度不足（4090 vs V100 cosine ~0.99） + 专家参数差异（r/b/τ）：

embedding 太像 → r·emb 差异很小

4090 专家训练更充分/迭代更多 → r/b/τ 更“占优势”

所以无论输入哪个硬件 embedding，softmax 后权重都类似 → 路由机制没体现出“硬件识别”

4) 老师的核心判断：最大的问题其实是“锚点选错了”（4090 vs V100太像）

老师把讨论拉回一个更 fundamental 的点：

4090、3090、V100 在你们的硬件分类里没有本质区别（至少不够异构）
→ 所以你现在做“跨硬件泛化”的实验，很可能只是测到“相近 GPU 之间 schedule 本来就接近”的现象；差几个百分点甚至可能接近实验误差、数据波动或收敛程度差异。

老师提出更合理的硬件分层框架（建议作为你们选锚点/选测试集的依据）：

高端 GPU（服务器/桌面强 GPU）

低端/边缘 GPU（Jetson 类）

高端 CPU（服务器/桌面 CPU）

低端/边缘 CPU（树莓派/RK3588 等）

并强调：你们方法的价值在于“少量锚点覆盖大量硬件”，因此锚点应该覆盖异构类别，而不是在同一类里挑两个很像的点。

5) 学生的顾虑：如果用“非常异构”的锚点，会让 KV/aligner 显得没用、设计显得多余

学生提出一个很真实的“论文叙事风险”：

你们有 KV 注入 / aligner / 统化句子等设计；

如果锚点选得太异构（比如 Canva/Jetson/CPU 这种），模型可能通过输入格式就能判断硬件类别，不需要 aligner 去补“被 mask 的关键硬件信息”；

这样一来：

性能可能更容易做得漂亮（因为官方模型跨域更差，你们更容易赢）

但你们的“KV 注入 + aligner”会显得像是“可有可无”，叙事上不优雅

老师的回应更偏“论文策略”：

在 motivation / case study 里，你可以用 “V100 模型直接去 Jetson 会很差” 作为示例，体现跨域困难；

但在真正的主要实验里，你完全可以：

给边缘类硬件也训练一个锚点 LoRA（比如 Jetson Orin）

再比较你们的“少锚点泛化”与对照方法

你们的贡献重点应落在：一个模型承载多个硬件知识 + 对未见硬件仍可路由泛化，而不是被迫让某个子模块“必须显得必要”。

6) 关于 baseline 怎么比：学生没想明白，讨论给出一个“折中可行”的方向

学生的困惑非常集中、也很关键：

只拿 TLM 官方 V100best 去比 Jetson/CPU：你们肯定赢，但会被 reviewer 说“不公平”

如果让 TLM 按官方流程在每个硬件上都训练到收敛：

成本极高（20 万测量、半个月级别）

而且它 full model 参数量远大于你们 LoRA（你们约 2% 参数），性能可能也会赢你们

结果叙事会变得尴尬：你们强调泛化，但单硬件最强可能打不过

讨论中形成的一个 折中 baseline 方案（学生觉得“tricky 但可能是平衡点”，老师也认为可暂用）：

用 TLM 官方模型（比如 V100 的）作为起点，

在目标新硬件上做“有限成本的 continue training / finetune / 续训”（用你们收集到的该硬件测量数据），作为对照曲线的一条线：

成本比“从头训满 20 万”小

至少能回应“你有没有让它适配新域”的质疑

同时不会把实验成本炸穿

你们的主线仍是：你们训练少量锚点 LoRA，就能覆盖更多硬件；而 TLM 需要每个硬件都付出训练成本。

7) 接下来要做的实验计划（讨论基本达成一致）
7.1 重新确定“主图”与实验模板

你们要画的主图，倾向于类似两种风格：

Pruner 风格：横轴测量次数，纵轴 latency（或 normalized performance），展示随着测量增长的收敛曲线；不同方法多条曲线对比。

TLM 风格：normalized performance / speedup，对多个模型、多个硬件做柱状/分布图。

讨论中更倾向：先以 Pruner 那种“收敛曲线 + 多硬件网格”作为主图模板（因为你们现在已经在画 convergence 曲线了，也最能解释成本/收敛）。

7.2 重新选硬件：不要再用 4090 vs V100 作为核心锚点组合

下一阶段你们要做的是：锚点覆盖异构类别，比如先选两类启动：

一个高端 GPU（4090 或 A100 任选其一）

一个边缘 GPU（Jetson Xavier/Orin）

再逐步把 CPU 类补齐（高端 CPU + 边缘 CPU）

学生最后给出了一个工程上可落地的方案草案：

保留当前比较成熟的 4090 LoRA

再训练一个 Jetson/Heavier（学生口中的硬件）上的 LoRA

泛化测试用 Orin（作为未见硬件）等

7.3 明确要验证“两层泛化性”

讨论中把泛化拆成两层（老师点出来后，学生明确“懂了”）：

泛化层 1：多锚点共存能力
一个统一模型/系统能同时承载多个硬件 LoRA 专家，并能通过路由把不同硬件“切开”。

泛化层 2：未见硬件泛化能力
对同一类里未训练的硬件（例如锚点是 Xavier，测试 Orin；锚点是 4090，测试 3090；锚点是高端 CPU，测试另一颗 CPU），依然能路由到合适专家并带来收益。

7.4 同步要做的工程修正

在推进主线实验的同时，你们也明确要“边做边修”：

继续把 LoRA 训练推进到更接近收敛（测量次数增加）

修 embedding 区分度不足的问题（否则 mix 权重不随硬件变）

观察并解释 outlier（BERT-base 的关键子图重复调用导致端到端夸张提升）

8) 学生的“新实验顾虑”也明确了（你可以直接当作下一步风险清单）

锚点换成 Jetson/CPU 后，aligner/KV 注入可能显得不必要（叙事风险）

TLM baseline 的公平比较很难：训满每硬件成本高且你们可能性能打不过；不训又会被说不公平

路由机制目前不工作：embedding 太像导致权重不变，mix 的“硬件感知”没体现（二、 当前面临的核心问题 (Key Issues)
通过对实验日志和 router.json 的手算分析，我们发现了以下**“反直觉”**现象及根因：
1. 专家选择失真（Cross-Hardware Weight Anomaly）：
    ◦ 现象：在 V100 硬件上生成时，4090 专家的权重反而比 V100 专家更高；且当切换到 4090 硬件时，两者权重几乎没有变化。
2. 硬件嵌入（Embedding）区分度极低：
    ◦ 根因：当前使用的 v4 版硬件向量，其 Cosine 相似度高达 0.997 以上。
    ◦ 维度霸权：向量中的 B 段（TVM 约束维度，如最大线程数、寄存器数等）数值很大（如 16、10）但在不同 GPU 之间完全一致。这些常量维度在点积运算中占据了绝对主导，掩盖了主频、显存带宽等真正具有区分度的特征。
3. 路由退化为“全局开关”：
    ◦ 由于输入 h 的区分度不足，路由向量 r 的训练实际上只学到了一个“专家整体强度”，而没有学到“硬件匹配度”。
    ◦ 这就导致了**“谁的参数幅度大谁就赢”**，而不是“谁最适合当前硬件谁就赢”。

--------------------------------------------------------------------------------
三、 诊断与改进方案 (Next Steps)
针对上述问题，我们计划从以下维度进行优化，以实现真正的“按需路由”：
• 特征工程优化：
    ◦ 降维或缩放：删除或显著缩小 8~11 维（TVM 常量约束）的权重，避免其主导打分。
    ◦ 标准化：对硬件向量进行标准化（LayerNorm 或 Z-score），平衡各维度对点积的贡献。
• 路由打分公式改进：
    ◦ 将目前基于点积的打分改为 余弦相似度（Cosine Similarity） 或 基于距离（Distance-based） 的打分，从而消除路由向量 r 绝对幅度的影响。
• 强化区分性训练：
    ◦ 引入对比学习（Contrastive Learning），强制要求专家在“原生硬件”上的得分显著高于“非原生硬件”。
总结建议： 当前的路由机制在工程通路上已经打通（Frozen Base + Top-K 推理），但受限于硬件描述向量的特征重叠，导致路由决策对硬件差异不敏感。后续工作的重点将从“流程实现”转向“硬件语义空间建模”的精细化调整。
💡 形象类比：现在的路由就像是在一群穿着一模一样制服（常量维度）的专家里选人，虽然他们胸前有不一样的姓名牌（微弱差异维度），但因为制服太显眼，路由最后只凭谁的个子高（向量幅度 r）来选人，而无视了他们是否真正专业对口。我们接下来的目标是让他们脱掉统一的制服，露出能体现专业特征的着装。）

测量成本巨大：收敛需要很长时间，实验周期压力大

当前 3090/2080Ti 的实验不容易赢：因为同类 GPU 上 TLM 方案天然就强，差距可能只在几个点内，容易被解释为误差/未收敛

LoRA Routing / Embedding Preprocess：从 pipeline 跑通到发现问题、再到当前方案演进
1) Pipeline 跑通后的第一轮现象：相似 GPU 上对比价值不高

我们最初把 2080Ti / 3090 作为“陌生硬件泛化目标”来做对比，但实际发现：同簇 GPU 上 TLM 本身就很强，性能差距可能只在几个点以内，很难形成清晰对照，也不利于把“routing/embedding 的贡献”讲清楚。

同时我们意识到：当前已训练的两个 LoRA anchor（4090 / V100）选得不理想——它们在我们设计的 embedding 空间里过于接近（或处于同一类形态），导致 routing 很难体现“硬件感知”。

2) 关键问题浮现：routing 机制“不工作”，权重几乎不随硬件变化

主要症状：对不同硬件输入，softmax 权重变化很小，或出现“v100 输入仍偏向 4090 expert”这类不符合直觉的现象。

初步原因猜测：

硬件 embedding 的可分性弱：许多维度在 GPU 内几乎不变（“常数段/近似常数段”），导致 dot 分数被这些维度主导。

每个 expert 的路由向量 r 缺乏可比性（无锚点）：在“单硬件数据训练 router”的设定下，r 很容易学成“把门开大”的尺度放大器（r 范数更大 → 更容易赢），而不是学成“区分方向”的匹配器。

评分函数用 dot 会把 routing 推向“选更强硬件/更大数值”的方向，而不是“选更匹配的硬件”。

3) A 级最小侵入改动：只改 routing 侧 embedding preprocess（不动 KV / 不重导数据）

为避免牵一发而动全身（KV aligner、注入链路、数据重导、已有 LoRA 知识都可能受影响），我们确定了 A 级修改原则：

数据不重导、原 v4 embedding 文件不改、KV 侧不动

仅在 routing 计算分数 前做统一 preprocess，并确保 训练/推理一致

旧 router 若无 preprocess meta，则要有 identity fallback（并提示警告/不允许混合时默认报错）

落地实现演进：

新增 embedding universe：hardware_embeddings_v4_universe.json
用于计算 preprocess 的 mean/std/mask，解决“训练数据里硬件种类太少 → std≈0 → z-score 不稳定”的风险。

preprocess 采用 zscore_mask，并加入保护：

std_floor + mask + clip + fallback(identity)（有效维度过少直接回退）

router.json 写入 meta.preprocess（type/source/std_floor/clip/mask_nonzero…）

混合专家时做一致性检查：preprocess 不一致默认报错，需显式 --allow-mixed-preprocess 才放行（并各用各的 preprocess 计算）

4) 预览结果：preprocess 确实提升了 embedding 的区分度，但 routing 仍可能偏置

我们用脚本预览到的趋势是明确的：

preprocess 前：GPU embedding cosine 非常接近（例如 4090 vs v100 接近 1）

preprocess 后：cosine 明显拉开（例如 4090 vs v100 从 ~0.997 降到 ~0.78，且部分组合甚至变负），说明 embedding 的“区分信号”被放大了。

但 routing 验收中仍出现关键现象：

dot 模式下：v100 输入可能仍偏向 4090 expert
根因经诊断更偏向：r 的范数差异主导（||r4090|| > ||rv100||），再叠加 r 方向高度相似（cos(r4090, rv100)≈0.95+），使得“谁 r 大谁赢”。

我们做了 score_mode 对照：

dot：不满足目标（尺度主导）

dot_over_rnorm：能让“各自硬件更偏向自己 expert”，但分离度很小、稳定性一般

cosine：更趋近平局，分离度更弱

5) 当前结论：问题不在 preprocess 是否生效，而在“dot 路由”的可比性与目标错位

已确认的关键发现：

h（embedding）差异是存在的（某些维度差异很大），但 dot 的主导维度往往落在“常数段/弱区分段”，导致差异没有变成稳定的排序翻转。

r 向量在不同 expert 之间 方向高度相似，主要差别来自 尺度/范数；在单硬件训练条件下这是“天然会发生”的不可辨识问题：router 更容易学“开门变大”而不是学“匹配方向”。

6) 待做测试与下一步路线（从最小侵入到结构性修改）

我们接下来要做的，是把 routing 的目标从“分数更大”对齐到“匹配更像”：

（A+）仍属低侵入：在推理侧采用去尺度 score + 可控温度

固定 score_mode=dot_over_rnorm，并加一个 temperature/scale 放大差异（注意：这只能放大“已有差异”，不能创造方向差异）。

验收标准从单次输出改为：在多 target、多次运行下仍稳定满足“各自硬件选自己 expert”的胜出关系。

（B）训练侧加入 r 的范数锚点/约束（让不同 expert 可比）

训练时对 r 加正则/weight_norm，使不同 expert 的 r 范数收敛到相近尺度，避免“门开大就赢”。

这比纯推理归一化更干净：推理逻辑保持简单，但训练要增加约束项与监控。

（C）原型相似度/CLiTE 风格：把 routing 改成“相似度匹配”

为每个 expert 定义一个原型 p_i（可用 emb 初始化/簇均值初始化），用 cos(h, p_i) 或 -||h-p_i|| 做路由分数，再叠加可学习强度标量。

目标显式变成“选更像的”，避免 dot 的“越大越赢”偏置。

以上三条路线可以并行做小实验：先用 A+ 快速稳定住行为，再决定是否上 B/C 做结构性解释更强的版本。

--------------------------------------------------------------------------------
四、Emb v4_universe 迁移进度 & Router-only/Aligner 现状

迁移目标：统一 hw_emb 坐标系（v4_universe + 固化 preprocess），保证数据生成、路由、KV 注入与推理一致。
我们此前定义了 7 步迁移路径（略）：
1) 统一 canonical embedding（v4_universe 作为 raw 真值）
2) 统一 preprocess 版本（固定 mean/std/mask/clip）
3) Router 侧 preprocess 对齐（训练/推理一致）
4) KV Aligner 侧 preprocess 对齐（训练/推理一致）
5) Router-only 校准
6) KV Aligner 重训
7) 推理回归与验收

当前进度：已进入第 3~4 步，并开始并行执行第 5~6 步（router-only 重训 + aligner 重训）。

Router-only 现象（校准后两次推理）：
- base 权重偏高（g_total 约 0.40~0.46）
- 竞争分布过尖（π 接近一边倒）
- 旧路由几乎不区分（r 方向高度相似），新路由区分过强（r 方向近乎相反）

结论：问题不在 preprocess，而在 router-only 训练目标“只学分开、不学强度”，再叠加正则导致 g 被动变小。

因此我们新增 4 项训练内改动（不手调 router.json）：
1) 强度监督（g*）：根据 I_pct 自动构造 g_target，显式监督 g_label，避免 base 权重过大。
2) 竞争平滑：对 π 增加熵正则或 label smoothing，防止分布过尖。
3) 可训练 T_comp：引入 competition_tau 的可训练参数，并对其加入弱先验，交由数据决定“分得多尖”。
4) 路由参数解耦：r = s * normalize(p)，方向负责分离、强度由 s/b 负责开门；避免 L2 把 g 顺手压小。

上述改动已落地到 router-only 训练与推理加载逻辑中，后续以训练结果为准，不再依赖手动调整 router.json。

补充：router-only 校准默认参数（已写死到训练脚本）
- g* 目标锚点固定为 0，尺度用 IQR（稳健尺度）；即 g_target = sigmoid(I_pct / (IQR * scale))。
- competition_tau 默认固定为 1.0（不训练）；如需训练需显式开启，并加强先验正则（默认 reg=0.1）。
- π 平滑默认启用：lambda_pi_entropy=0.12，label_smoothing=0.1。
（对应脚本：gen/train_router_calibration.py）

补充验证（Arch swap sanity）：
- 在同一条 workload 上，仅替换 text_full 中的 `-arch=sm_86` → `-arch=sm_70`，其余保持不变。
- 使用 bucket tokenizer + teacher 前向，统计 schedule token 上的差异：
  - mean|Δlogits| ≈ 7.5e-02 ~ 7.9e-02
  - KL ≈ 2.5e-03
- 结论：teacher 对 arch 有明显敏感度，swap‑KD 信号不是 0。当前训练不收敛更可能来自 loss/路径/权重设置，而非 emb 本身无效。
