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