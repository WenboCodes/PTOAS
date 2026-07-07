# PTO 虚拟微指令(`pto.vmi`)设计

[toc]

---
## Part 0 —— 背景介绍

### 0.0 定位

`pto.vmi`(PTO virtual micro instruction) 是夹在**上层编程模型**(如 TileLang `T.parallel`)与**底层 `pto.mi` 物理 ISA**（CCE）之间的中间层：它只暴露逻辑连续语义与逐元素计算意图，物理 SIMD 寄存器的 layout(交织/半区/宽窄摆放/part/pack/dist)由 `pto.as` 持有与传播，对用户不可见。完整动机见 [RFC-VPTO-Logical-Vector-ISA.md](RFC-VPTO-Logical-Vector-ISA.md)。

```
TileLang  T.parallel(N) { C[i] = cast<i32>(A[i]) + B[i] }   ← 用户：只有逻辑数据
   │  (直译，逐元素语义保持)
   ▼
pto.vmi      %w = pto.vmi.vcvt %a ; %c = pto.vmi.vadd %w, %b           ← 逻辑向量，layout 由编译器持有
   │  (pto-as：layout-assignment + lowering)
   ▼
pto.mi       vcvt EVEN/ODD + 两路 vadd + vstsx2 INTLV_B32        ← 物理：SIMD 寄存器交织细节
```

- **上层 → vmi**:`T.parallel` 的逻辑迭代空间】直译成 `pto.vmi` 的逻辑向量 op——逐元素计算 → Category A op,`T.cast` → 一条无 `part` 的 `pto.vmi.vcvt`,逻辑长度 `N` → `!pto.vmi.vreg<N×T>`,"全 active" 语义 → 自动补齐的尾部谓词。
- **vmi → pto.mi**:`pto.as` 做 layout 推断 + 协调 + 连续化,把逻辑向量 lower 成具体 `pto.mi` 指令(含 `part/pack/interleave/dist`)。`K=1` 时退化为零开销直通。

### 0.1 记号

| 记号 | 含义 |
|---|---|
| `V<L×T>` | `!pto.vmi.vreg<L×T>` 逻辑向量，`L` = 逻辑元素数 |
| `M<L>` | `!pto.vmi.mask<L>` 虚拟谓词 |
| `Ptr<T>` | `!pto.ptr<T,ub>` UB 指针 |
| `s` | 标量 |
| `[pmode]` | 可选治理谓词（governing mask），`{pmode = "merge"\|"zero"}`，默认 `zero` |
| `[dist-mode]` | 可选访问形态（vload/vstore 专用），`{dist-mode = "continuous"\|"unpack"\|"dintlv"\|"brc"}`，默认 `continuous` |

> **物理记号（附录）**：以下记号是 pto.as 内部的物理量，**不**出现在 surface 签名中，仅用于本文档的物理视图与 lowering 说明；surface 用户只需写 `L×T`，`K`/`E_v`/`BlockLane` 由 pto.as 持有。
>
> - `K`：一个逻辑 vreg 的物理后备数，`K = L·bitwidth(T) / 2048`（`K_raw < 1` 时按 `K = 1` 处理，见 §1.1）。
> - `E_v`：一个物理 vreg 的 lane 数（f32/i32 = 64，f16/bf16/i16 = 128，i8/fp8 = 256）。
> - `BlockLane`：硬件 32B 原子归约单元，每个物理 vreg = 8 个 BlockLane；每个 BlockLane 容纳 `32B / bitwidth(T)` 个 lane。

### 0.2 Category A/B/C —— 精确 lowering 契约

RFC §5 给了三类定义，这里展开成 pto.as 的可执行判据。三类的本质区分点是**op 是否修改、以及对寄存器 layout 做出何种假定**：

| 类别 | 对 layout 的关系 | pto.as 行为 | 输出 layout |
|---|---|---|---|
| **A. Layout-passthrough** | **不修改**寄存器 layout（做透传） | 每 `K` 个物理 reg 各 fan-out 一次该 `pto.mi` op；治理谓词按物理 mask 族逐 reg 配（必要时 `ppack/punpack`） | 与输入相同（layout 透传，含 parity/half 等轴一并透传） |
| **B. Layout-rewritable** | **有规则地修改**寄存器 layout | 沿**其他**轴 fan-out，对匹配轴实例化对应模式（`PART_EVEN/ODD`、`Bin_N0/N1`、`PK/UNPK`、`INTLV/DINTLV`…） | 消费或产生该轴 |
| **C. Contiguous-required** | **对物理 layout 有强假定**（需 stride-1 连续视图，且无匹配模式可原地满足） | 在该 op **之前**先自动插 `.contiguous()` 物化（`INTLV`/`pack`/搬移）把寄存器 layout 转成 continuous，再做连续 op | 扁平 chunk（`is_contiguous`） |

> **C 类的强假定**：C 类 op 无法在任意寄存器 layout 上原地执行，它假定输入已是 continuous。因此 pto.as 在 C 类 op 之前会**显式插入 layout 物化**，把上游的 parity/half/sub_part 等非连续轴拍平成 continuous 再喂给该 op——这是从 A/B 类（可带 layout 运算）跨越到 C 类（强假定连续）的唯一过渡点。

### 0.3 谓词传播总则

1. **governing mask 伴随数据轴**：`[pmode]` 在 lowering 时随数据 fan-out 到每个物理 reg；mask 族 `G` 必须与数据族对齐（f32/i32→b32，f16/bf16/i16→b16，i8/fp8→b8）。跨族（如 widen i16→i32）时 mask 族也变，pto.as 用 `ppack/punpack` 或重新 `pset` 生成对应族 mask。
2. **inactive lane 行为**：`pmode="zero"`（默认）inactive lane 写 0；`pmode="merge"` 保留目的原值。reduce 类 op 的 inactive lane 见 §2.3。
3. **A5 load 不可谓词化**：`vload` 的尾部谓词不能挂在 load 上，必须迁移到消费侧 op 或 store。`vstore` 在 A5 上可谓词化。
4. **tail mask 物化**：`pset "PAT_ALL"` / `pge "PAT_VLn"` / `plt %rem` 三件套覆盖全 active、头尾 active、数据相关尾三种模式（见 §10）。

---

## Part 1 —— 虚拟类型形式化

### 1.1 `!pto.vmi.vreg<L×T>`

- **合法性**：完整向量的 `L · bitwidth(T)` MUST 是 2048bit/256B 的整数倍；小于 256B 的 compact/partial vreg 允许存在，其物理后备仍按 1 个 256B vreg 分配。
- **物理后备**：令 `K_raw = L·bitwidth(T) / 2048`。完整向量使用 `K = K_raw` 个 `!pto.mi.vreg<E_v×T>`；`K_raw < 1` 时按照 `K=1` 处理，低 `L` 个逻辑槽有效，其余物理槽不属于该逻辑值。`K=1` 且非 partial 时 vmi.vreg 与 pto.mi.vreg 一一对应。
- **`#layout`**：surface 源码省略，由 pto.as 填充。layout 是**编译器内部属性**，不进入用户类型签名（用户只写 `L×T`）。
- **合法 `L` 与 dtype**：

| T | bits | E_v | L 必须是 … 的2的幂次倍数 |
|---|---|---|---|
| f32 / i32 | 32 | 64 | 64 |
| f16 / bf16 / i16 | 16 | 128 | 64 |
| i8 / fp8 | 8 | 256 | 64 |

### 1.1.1 常用 vreg 逻辑/物理视图

本小节对每种常用 vreg 给三张图：**逻辑视图**、**物理视图（连续）**、**物理视图（非连续）**。`fp16/fp32` 对应本文类型名 `f16/f32`。
每个物理 vreg 固定为 `256B = 2048bit = 8 × 32B BlockLane`。图中数字范围是逻辑 lane id；
`pad/undef` 表示该物理槽不属于逻辑值，consumer 必须通过逻辑长度/谓词忽略。
非连续 layout 不改变 `V<L×T>` 的逻辑顺序，只改变逻辑 lane 到物理 reg/lane 的映射：
`parity(EVEN/ODD)` 是 stride-2 奇偶交织；`sub_part(P0~P3)` 是 8-bit 结果在 4B group 内的 byte 槽位，
只用于 fp8/i8 carrier，不是 fp16/fp32 的原生 lane 轴。

| 逻辑类型 | 逻辑字节数 | `K_raw` | 分配的物理 vreg | 每个物理 vreg 的有效槽 |
|---|---:|---:|---:|---|
| `V<256×fp8>` | 256B | 1 | 1 | 256 个 fp8 lane 全有效 |
| `V<256×fp16>` | 512B | 2 | 2 | 每个 128 个 fp16 lane 全有效 |
| `V<256×fp32>` | 1024B | 4 | 4 | 每个 64 个 fp32 lane 全有效 |
| `V<64×fp16>` | 128B | 1/2 | 1 | 低 64 个 fp16 lane 有效，高 64 个无效 |
| `V<64×fp8>` | 64B | 1/4 | 1 | 低 64 个 fp8 lane 有效，高 192 个无效 |

#### `V<256×fp8>`：1 个物理 reg（K=1）

**逻辑视图**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x254 │ x255 │
└────┴────┴────┴─────┴──────┴──────┘
                  256 lane
```

**物理视图（连续）** — 1 个物理 reg，每个 BlockLane = 32B = 32 个 fp8 lane

```text
   BL0          BL1                  BL7
┌─────────────┬─────────────┬───┬─────────────┐
│ x0 ... x31  │ x32 ... x63 │...│x224 ... x255 │
└─────────────┴─────────────┴───┴─────────────┘
                   P0 (256B)
```

#### `V<256×fp16>`：2 个物理 reg（K=2）

**逻辑视图**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x254 │ x255 │
└────┴────┴────┴─────┴──────┴──────┘
                  256 lane
```

**物理视图（连续）** — 2 个物理 reg，每个 BlockLane = 32B = 16 个 fp16 lane

```text
   BL0           BL1               BL7
┌──────────┬──────────┬───┬──────────┐
│ x0..x15  │ x16..x31 │...│x112..x127 │
└──────────┴──────────┴───┴──────────┘
                  P0 (256B)

   BL0           BL1               BL7
┌──────────┬──────────┬───┬──────────┐
│x128..x143│x144..x159│...│x240..x255 │
└──────────┴──────────┴───┴──────────┘
                  P1 (256B)
```

**物理视图（非连续, parity EVEN/ODD）** — 偶数 lane 摆 P0、奇数 lane 摆 P1（如 `DINTLV_B16` load 或 `vdintlv` 后保留奇偶态；两 reg 各 128 lane 全有效）

```text
   P0 (EVEN)                                   P1 (ODD)
┌────┬────┬────┬─────┬──────┬──────┐  ┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x2 │ x4 │ ... │ x252 │ x254 │  │ x1 │ x3 │ x5 │ ... │ x253 │ x255 │
└────┴────┴────┴─────┴──────┴──────┘  └────┴────┴────┴─────┴──────┴──────┘
   128 lane 偶数有效                       128 lane 奇数有效
```

> 还原连续态: `INTLV_B16(P0, P1) → [x0 x1 x2 x3 ... x255]`。

#### `V<256×fp32>`：4 个物理 reg（K=4）

**逻辑视图**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x254 │ x255 │
└────┴────┴────┴─────┴──────┴──────┘
                  256 lane
```

**物理视图（连续）** — 4 个物理 reg，每个 BlockLane = 32B = 8 个 fp32 lane

```text
   BL0       BL1               BL7
┌───────┬───────┬───┬───────┐
│ x0..7 │ x8..15│...│x56..63│
└───────┴───────┴───┴───────┘
            P0 (256B)

   BL0        BL1              BL7
┌────────┬────────┬───┬────────┐
│x64..71 │x72..79 │...│x120..127│
└────────┴────────┴───┴────────┘
            P1 (256B)

   BL0          BL1             BL7
┌──────────┬──────────┬───┬──────────┐
│x128..135 │x136..143 │...│x184..191 │
└──────────┴──────────┴───┴──────────┘
            P2 (256B)

   BL0          BL1             BL7
┌──────────┬──────────┬───┬──────────┐
│x192..199 │x200..207 │...│x248..255 │
└──────────┴──────────┴───┴──────────┘
            P3 (256B)
```

**物理视图（非连续, parity EVEN/ODD）** — 偶数 lane 摆 P0/P2、奇数 lane 摆 P1/P3（典型来源：`V<256×fp16> → V<256×fp32>` 加宽后保留 parity；4 reg 各 64 lane 全有效）

```text
 P0 (chunk0 EVEN)   P1 (chunk0 ODD)    P2 (chunk1 EVEN)   P3 (chunk1 ODD)
┌────┬────┬─────┐ ┌────┬────┬─────┐ ┌──────┬──────┬─────┐ ┌──────┬──────┬─────┐
│ x0 │ x2 │x126 │ │ x1 │ x3 │x127 │ │ x128 │ x130 │x254 │ │ x129 │ x131 │x255 │
└────┴────┴─────┘ └────┴────┴─────┘ └──────┴──────┴─────┘ └──────┴──────┴─────┘
    64 lane            64 lane            64 lane            64 lane
```

> 还原连续态: `INTLV_B32(P0, P1) → [x0..x127]`、`INTLV_B32(P2, P3) → [x128..x255]`，再按 chunk 顺序拼回。

**物理视图（非连续, P0/P1/P2/P3）** — 4-way stride-4 交织：每 4 个逻辑元素各落一个 reg（`x0,x4,...` → P0；`x1,x5,...` → P1；`x2,x6,...` → P2；`x3,x7,...` → P3）；4 reg 各 64 lane 全有效（对应 sub_part / part_T 4-way 轴）

```text
   P0                   P1                   P2                   P3
┌────┬────┬─────┐  ┌────┬────┬─────┐  ┌────┬────┬─────┐  ┌────┬────┬─────┐
│ x0 │ x4 │x252 │  │ x1 │ x5 │x253 │  │ x2 │ x6 │x254 │  │ x3 │ x7 │x255 │
└────┴────┴─────┘  └────┴────┴─────┘  └────┴────┴─────┘  └────┴────┴─────┘
     64 lane              64 lane              64 lane              64 lane
```

#### `V<64×fp16>`：1 个 partial 物理 reg（K=1, 低 64 lane 有效）

**逻辑视图**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x62  │ x63  │
└────┴────┴────┴─────┴──────┴──────┘
                  64 lane
```

**物理视图（连续）** — 1 个物理 reg，低 64 lane 有效，每个 BlockLane = 16 个 fp16 lane

```text
   BL0          BL1         BL2          BL3          BL4   BL5   BL6   BL7
┌──────────┬──────────┬──────────┬──────────┬──────┬──────┬──────┬──────┐
│ x0..x15  │ x16..x31 │ x32..x47 │ x48..x63 │      │      │      │      │
└──────────┴──────────┴──────────┴──────────┴──────┴──────┴──────┴──────┘
<------------- 128B logical payload -------------><---- 128B outside logical value ---->
                          P0 (256B)
```

**物理视图（非连续, part EVEN/ODD）** — 单个 `V<64×fp32> → V<64×fp16>` 窄化 carrier：64 个有效 fp16 摆在 128 物理 lane 的偶/奇位

```text
   EVEN carrier（phys lane 0,2,...,126 有效）
┌────┬───┬────┬───┬─────┬─────┬───┬─────┬───┐
│ x0 │ _ │ x1 │ _ │ ... │ x62 │ _ │ x63 │ _ │
└────┴───┴────┴───┴─────┴─────┴───┴─────┴───┘
```

> 对比: fp8/i8 的 `sub_part(P0~P3)` 是 4B group 内的 byte 槽位（`[P0 P1 P2 P3] [P0 P1 P2 P3] ...`），与 fp16 的 part EVEN/ODD 不同轴，见 `V<64×fp8>`。

#### `V<64×fp8>`：1 个 partial 物理 reg（K=1, 低 64 lane 有效）

**逻辑视图**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x62  │ x63  │
└────┴────┴────┴─────┴──────┴──────┘
                  64 lane
```

**物理视图（连续）** — 1 个物理 reg，低 64 lane 有效，每个 BlockLane = 32 个 fp8 lane

```text
      BL0           BL1          BL2   BL3   BL4   BL5   BL6   BL7
┌─────────────┬─────────────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ x0 ... x31  │ x32 ... x63 │      │      │      │      │      │      │
└─────────────┴─────────────┴──────┴──────┴──────┴──────┴──────┴──────┘
<-- 64B logical payload --><--------------- 192B outside logical value --------------->
                          P0 (256B)
```

**物理视图（非连续, sub_part P0）** — 来自 `V<64×fp32> → V<64×fp8>` 的 `vcvt`：不连续放低 64B，而是把每个 4B group 的第 0 个 byte 填上有效 fp8（`PK4_B32` 抽取目标）

```text
P0: 256B fp8 carrier，按 64 groups × 4B 看，每 group 仅 P0 槽有效
┌───────────┬───────────┬─────┬────────────┐
│ x0  _  _  _│ x1  _  _  _│ ... │ x63  _  _  _│
│ P0 P1 P2 P3│ P0 P1 P2 P3│     │ P0 P1 P2 P3 │
└───────────┴───────────┴─────┴────────────┘
   grp0          grp1              grp63
```

> 该稀疏视图是 lowering layout，不改变 `V<64×fp8>` 的逻辑视图；`vstore` lower 到 `PK4_B32` 时按此抽取，写出连续 64B fp8。

### 1.2 `!pto.vmi.mask<L>`

- 虚拟谓词，物理后备 `K` 个 256-bit `!pto.mi.mask<G>`。
- `G` 由所修饰数据类型推导（与数据族对齐，见 §0.3-1），不出现在用户类型签名。
- **合法性**：`L` 与所修饰 vreg 的 `L` 一致；`K` 一致。

### 1.3 compact reduce 结果

reduce/broadcast 涉及"小于 256B 的 compact 标量向量"（如 `V<2×f32>`、`V<8×u16>`）。其第一根轴的维度与`group`相等。

---

## Part 2 —— `group` 语义形式化

### 2.1 定义

对 reduce op（`vcadd`/`vcmax`/`vcmin`）与广播 op（`vbrc`），`{group=C}` 中的 `C` 表示 **group 数（组数）**，而非每 group 的 lane 数：

- **reduce**：把 `V<L×T>` 的 `L` 个 lane 切成 **`C` 个 group**，每 group `L/C` lane，各产 1 个 compact scalar；输出 `V<C×T>`（`C` 个 scalar，低 `C` 槽有效）。
- **vbrc**：把 compact 输入 `V<C×T>` 的 `C` 个 scalar 分别广播回各自 `(L/C)`-lane group，输出 `V<L×T>`。

`C` 必须整除 `L`；**每 group 的 lane 数**为 `L/C`，其字节数 `W = (L/C)·bitwidth(T)/8` 决定与 BlockLane 边界的关系，从而决定 Category。

**reduce 结果的合法 compact vreg 类型**：reduce 输出 `V<C×T>` 是"小于 256B 的 compact 标量向量"（见 §1.3），`C` 只能取 **`1 / 2 / 4 / 8`**——即合法的 reduce 结果虚拟寄存器类型只有 `V<1×T>`、`V<2×T>`、`V<4×T>`、`V<8×T>` 四种。因此 op 属性 `{group=C}` 的 `C` 必须与返回类型 `V<C×T>` 的 `C` **严格一致**（`group=8` ⇄ 返回 `V<8×T>`，`group=4` ⇄ `V<4×T>`，以此类推）；二者失配是非法 IR，由 pto.as 在类型检查阶段拒绝。

### 2.2 `group` → Category 决策表

记 `W = (L/C) · bitwidth(T) / 8`（一个 group 的字节数，`L/C` 为每 group 的 lane 数），`BlockLane = 32B`。

| `W` 与 BlockLane 关系 | Category | pto.as lowering | 物理指令 |
|---|---|---|---|
| `W == 32`（group 恰为 1 BlockLane） | **B** | 每 BlockLane 独立 reduce，1 op per physical reg，无跨 reg 合并 | `vcgadd`/`vcgmax`/`vcgmin` |
| `W = k·32, k>1`（group 跨 k 个 BlockLane，对齐 BlockLane 边界） | **B**（fold + vcg） | 先 `(k-1)` 次 `vadd/vmax/vmin` 把 k 个 BlockLane fold 成 1 个 BlockLane 宽的中间值，再 `vcg*` | `vmax×(k-1)` + `vcg*` |


### 2.3 reduce 的 inactive lane 与 argmax

- **inactive lane**：`vcmax/vcmin` 视 inactive 为 `-INF/+INF`（fp）或类型字面 min/max（int）；全 inactive 时 `result[0]` 为该极值。`vcadd` 视 inactive 为 0。与 SPEC §10 一致。

  ```mlir
  %val = pto.vmi.vcmax %x, %m {group = C}
      : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<C×T>
  ```

### 2.4 典型场景：MX block-scale exponent reduce（bf16，group=8）

`V<256×bf16>` 取 shared exponent，每 32-element block 一个 max → `256/32 = 8` 个 group，故 `group=8`（每 group `256/8 = 32` lane）。bf16 `bitwidth=16`，`W=32·16/8=64B=2 BlockLane` → 落 §2.2 第二行（`k=2`，对齐 BlockLane，Category B）。

- load：pto.as 从下游 `group=8` 逆推，选 `DINTLV_B16`（parity）让 even/odd 各成一 vreg。
- `vand` 提 exponent：A 类透传，两 parity vreg 各 `vand`。
- fold：`(k-1)=1` 次 `vmax` 把 even/odd 两 vreg fold 成 1 个，parity 轴消失，每 BlockLane 16 lane 覆盖一个 32-element block 的 max exponent。
- `vcgmax`：每 BlockLane 取 max → 8 个 shared exponent（与 `group=8` 一致；8 BlockLane 各产 1 个，一一对应）。
- 输出：compact `V<8×bf16>`（`C=8` 个 shared exponent）。

#### 32×32 tile → `V<256×bf16>` 摆放

一个 32×32 的 bf16 tile 共 1024 个元素，需要 **4 个 `V<256×bf16>`**(每个 256 lane = 8 行 × 32 列)。按行优先切成 4 个 8 行 × 32 列的行片,每个 vreg 装一片:lane `n·32 + j` = tile 行 `row_off + n`、列 `j`(`n ∈ 0..7`,`j ∈ 0..31`;`row_off ∈ {0,8,16,24}` 对应 V0~V3)。

```text
                   32×32 bf16 tile
   col 0                              col 31
┌──────────────────────────────────────────────┐
│ row0   c0  c1  c2  ...  c30  c31              │
│ row1   ...                                   │   ←─ V0 (row 0–7)   8×32=256
│ ...                                          │
│ row7   ...                                   │
├──────────────────────────────────────────────┤
│ row8   ...                                   │
│ ...                                          │   ←─ V1 (row 8–15)  8×32=256
│ row15  ...                                   │
├──────────────────────────────────────────────┤
│ row16  ...                                   │   ←─ V2 (row 16–23) 8×32=256
│ ...                                          │
│ row23  ...                                   │
├──────────────────────────────────────────────┤
│ row24  ...                                   │
│ ...                                          │   ←─ V3 (row 24–31) 8×32=256
│ row31  ...                                   │
└──────────────────────────────────────────────┘
```

> **与 `group=8` reduce 的对应**:`group=8` 把 256 lane 切成 8 组、每组 32 lane。在 8×32 行优先摆放下,每 32 lane 恰好是**完整的一行**(32 element)。因此一个 vreg 的 8 个 group 就是它持有的 8 行(row 0–7),`vcgmax` 每行取一个 max → 8 个 shared exponent,与 8 行一一对应——这正是 MX block-scale "每 32-element block 一个 shared exponent"的语义(tile 的每一行即一个 32-element block)。

---

## Part 3 —— Group 1:Load / Store

`vload`/`vstore` 是逻辑访存。**`[dist-mode]` 显式声明访问形态**,默认 `continuous`(连续);
可选 `unpack`(加宽 unpack)、`dintlv`(解交织)、`brc`(广播)。物理 `dist` token
(`NORM_B*` / `UNPK_B*` / `DINTLV_B*` / `BRC_B*`)对用户不可见,由 pto.as **根据 UB 指针
的元素类型 `T` 推导出对应的 `B*` 后缀**(见下表),再实例化为具体 `pto.mi` 指令。
原 `dist` token 的 layout 推断(连续视图消费者反推)仍由 pto.as 内部完成(见配套文档 §3)。

> `[dist-mode]` 与 layout 推断的关系:`[dist-mode]` 是用户对**访问形态**的显式声明(怎么读/写 UB),
> layout 轴推断是 pto.as 对**寄存器侧摆放**的隐式决策(数据进入 vreg 后怎么摆)。两者正交:
> 即便 `[dist-mode=continuous]`,pto.as 仍可能为服务下游 reduce 而把 load lower 成 `DINTLV_B*`
> (产生 `parity` 轴)——此时物理 dist 与 surface dist-mode 不一致,是 pto.as 的合法优化。

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vload` | A | `Ptr<T>`, `[dist-mode]`, `[pmode]` | `V<L×T>` | i8–i32, f16/bf16, f32 |
| `vstore` | A | `V<L×T>`, `Ptr<T>`, `M`, `[dist-mode]`, `[pmode]` | — | i8–i32, f16/bf16, f32 |

### `[dist-mode]` 取值与指针类型 → 硬件 dist 推导

`[dist-mode]` 默认 `continuous`;`B*` 后缀由 `Ptr<T>` 的元素位宽决定(`T` 为 8/16/32-bit → `B8`/`B16`/`B32`)。

| `[dist-mode]` | 语义 | vload → 物理 dist | vstore → 物理 dist |
|---|---|---|---|
| `continuous`(默认) | 连续 stride-1 访问 | `NORM` / `NORM_B*` | `NORM_B*` |
| `unpack` | 加宽 unpack:窄源按 `T` 展开到更宽 lane | `UNPK_B*` | —(store 无 unpack) |
| `dintlv` | 解交织/交织:成对 even/odd 半区 | `DINTLV_B*`(dual load,`vldsx2`) / `BDINTLV` | `INTLV_B*`(dual store,`vstsx2`) |
| `brc` | 广播:标量/块复制进 vreg | `BRC_B*` / `BRC_BLK` | — |

> **指针类型 → `B*` 后缀**:`!pto.ptr<f32,ub>` → `B32`;`!pto.ptr<bf16,ub>` → `B16`;
> `!pto.ptr<i8,ub>` → `B8`。`continuous` 的 load 用元素宽度无关的 `NORM`,store 用 `NORM_B*`。
> `dintlv`/`brc`/`unpack` 的 `B*` 后缀同样由 `Ptr<T>` 推导。`dintlv` 物理上是 dual 形态(对应 `vldsx2`/`vstsx2`),
> 但 surface 仍以单条 `vload`/`vstore` + `[dist-mode=dintlv]` 表达**单输入/单输出**——load 出 1 个逻辑 vreg、
> store 进 1 个逻辑 vreg;EVEN/ODD 两半区的拆分/合成由 pto.as 在 lowering 时完成(逻辑视图始终连续)。

示例:

```mlir
// 连续 load(默认 dist-mode):UB → vreg
%v = pto.vmi.vload %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
//  ↑ pto.as:Ptr<f32> → B32,dist-mode=continuous → pto.mi.vlds {dist="NORM"}

// 连续 store:vreg → UB(带治理谓词,A5 上 store 可谓词化)
pto.vmi.vstore %v, %ub_out[%offset], %mask : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.mask<64>
//  ↑ pto.as:Ptr<f32> → B32,dist-mode=continuous → pto.mi.vsts {dist="NORM_B32"}

// 广播 load:标量/块复制进 vreg
%vb = pto.vmi.vload %ub[%offset] {dist-mode = "brc"} : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
//  ↑ pto.as:Ptr<f32> → B32,dist-mode=brc → pto.mi.vlds {dist="BRC_B32"}(每 lane = UB[base])

// 加宽 unpack load:窄源展开到宽 lane
%u = pto.vmi.vload %ub[%offset] {dist-mode = "unpack"} : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<64xf32>
//  ↑ pto.as:Ptr<bf16> → B16,dist-mode=unpack → pto.mi.vlds {dist="UNPK_B16"}

// 解交织 load(surface 单输出;lowering 时 pto.as split 成 EVEN/ODD 两个物理 reg)
%v = pto.vmi.vload %ub[%offset] {dist-mode = "dintlv"}
    : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
//  ↑ pto.as:Ptr<f32> → B32,dist-mode=dintlv → pto.vldsx2 "DINTLV_B32"
//    surface 一条 vload 出 1 个逻辑 vreg;pto.as 在 lowering 时把其物理后备 split 成
//    EVEN/ODD 两个半区 reg(parity 轴),逻辑视图仍为连续 64 个 f32。

// 交织 store(surface 单输入;lowering 时 pto.as 把 EVEN/ODD 两半区合成成 dual store)
pto.vmi.vstore %v, %ub_out[%offset], %mask {dist-mode = "dintlv"}
    : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.mask<64>
//  ↑ pto.as:Ptr<f32> → B32,dist-mode=dintlv → pto.vstsx2 "INTLV_B32"
//    surface 一条 vstore 进 1 个逻辑 vreg;其物理后备是 EVEN/ODD 两半区,
//    pto.as 在 lowering 时合成成 dual store 写回连续内存。

// tail/partial load:A5 上 load 不可谓词化,%mask 迁移到消费侧/store
%vt = pto.vmi.vload %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>   // 尾谓词在下游 vadd/vstore 生效
```

---

## Part 4 —— Group 2:index-gen

复制与索引物化。产生 `broadcast` 轴或索引向量;
直到 Category-B/C 边缘需要展开形式前,绝不展开成 `K` 份存储副本。

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vci` | A | `s`  | `V<E×i32>` (`{ASC/DESC}`) | i8–i32, f16, f32 |

示例:

```mlir
// vci:生成 [base, base+1, ...] lane 索引(ASC/DESC 由属性给出,%base 为起始标量)
%idx = pto.vmi.vci %base {order = "ASC"} : i32 -> !pto.vmi.vreg<64xi32>
```

---

## Part 5 —— Group 3:Eltwise compute


| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vadd` `vsub` `vmul` `vdiv` `vmax` `vmin` | A | `V<T>`, `V<T>` (或 bcast), `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 (`vdiv` f16/f32) |
| `vand` `vor` `vxor` | A | `V<T>`, `V<T>`, `[pmode]` | `V<T>` | i8–i32 |
| `vnot` | A | `V<T>`, `[pmode]` | `V<T>` | i8–i32 (bit-typed) |
| `vshl` `vshr` | A | `V<T>`, `V<T>` (向量 count), `[pmode]` | `V<T>` | i8–i32 |
| `vadds` `vmuls` `vmaxs` `vmins` `vshls` `vshrs` | A | `V<T>`, `s`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vabs` `vneg` `vrelu` | A | `V<T>`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vexp` `vln` `vsqrt` | A | `V<T>`, `[pmode]` | `V<T>` | f16, f32 |
| `vcmp` | A | `V<T>`, `V<T>`, `M` | `M` | i8–i32, f16/bf16, f32 |
| `vcmps` | A | `V<T>`, `s`, `M` | `M` | i8–i32, f16/bf16, f32 |
| `vsel` | A | `M`, `V<T>`, `V<T>`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vselr` | A | `V<T>`, `V<index>` | `V<T>` (permute) | i8–i32, f16/bf16, f32 |

示例:

```mlir
// 二元算术(逐 lane,带治理谓词)
%s = pto.vmi.vadd %a, %b, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
%m = pto.vmi.vmax %a, %b, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>

// 向量-标量(标量隐式广播)
%scaled = pto.vmi.vmuls %x, %scale, %mask : !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
%shifted = pto.vmi.vshrs %data, %c4, %mask : !pto.vmi.vreg<64xi32>, i16, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xi32>

// 一元算术/激活
%a = pto.vmi.vabs %v, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
%e = pto.vmi.vexp %v, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>

// 比较 → 谓词(第三个 M 为治理谓词,限定哪些 lane 参与比较;inactive lane 结果置 0)
%lt = pto.vmi.vcmp %a, %b, %m, "lt" : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.mask<64>

// 标量比较 → 谓词
%ges = pto.vmi.vcmps %a, %c0, %m, "ge" : !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<64> -> !pto.vmi.mask<64>

// 谓词选择:%mask 为真取 %x,否则取 %y
%out = pto.vmi.vsel %x, %y, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>

// 寄存器 gather/permute
%p = pto.vmi.vselr %x, %idx : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xf32>
```

---

## Part 6 —— Group 4:Broadcast

`vbrc` 是逻辑标量→向量/已归约→扇出广播(R6)。未分组形式便宜;分组形式(每 BlockLane
partial 扇回自身 lane)是难情形。

| op (form) | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vbrc` (未分组) | A | `s` | `V<L×T>` | i8–i32, f16/bf16, f32 |
| `vbrc` (`{group=C}`) | B | `V<C×T>` | `V<L×T>` | i8–i32, f16/bf16, f32 |

示例:

```mlir
// 未分组广播:标量/已归约值扇出到整条 vreg(寄存器内,无 UB roundtrip)
%bc = pto.vmi.vbrc %maxe : f32 -> !pto.vmi.vreg<64xf32>

// 分组广播:C=8 个 scalar 各扇回自身 (L/C)=8 lane group(无直接物理指令对应,实现由 pto.as决定)
%scaleb = pto.vmi.vbrc %maxe {group = 8} : !pto.vmi.vreg<8xf32> -> !pto.vmi.vreg<64xf32>
```


---

## Part 7 —— Group 5: reduce

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vcadd` (`{group=C}`) | B | `V<L×T>`, `[pmode]` | `V<C×T>` | i8–i32, f16, f32 |
| `vcmax` (`{group=C}`) | B | `V<L×T>`, `[pmode]` | `V<C×T>` | i16–i32, f16, f32 |
| `vcmin` (`{group=C}`) | B | `V<L×T>`, `[pmode]` | `V<C×T>` | i16–i32, f16, f32 |

示例:

```mlir
// 全数组求和归约(到标量)
%sum = pto.vmi.vcadd %x, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<1xf32>

// 全数组 max 归约(到标量)
%mx = pto.vmi.vcmax %x, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<1xf32>

// group 归约:group=8 → 256 lane 切 8 组(每组 32 lane),各取一个 max → 8 个 compact scalar
%maxe = pto.vmi.vcmax %exp {group = 8} : !pto.vmi.vreg<256xu16>, !pto.vmi.mask<256> -> !pto.vmi.vreg<8xu16>
```


---

## Part 8 —— Group 6:Convert (cvt)

一个逻辑 `vcvt`,其 *目标 dtype 即布局*。`pto.as` 展开为 dtype-specific cast 链 +
part/width staging + 匹配的 store distribution,并拖出谓词伴随。

**属性**:`{to=<dtype>, rnd=<R>, sat=<SAT>}`(`to` 可由返回类型推断,为显式冗余;`rnd`/`sat`
控制窄化时的取整与饱和)。`part`/`PART_*`/`PK`/`UNPK` **不出现在 surface**,由 pto.as 作为
内部 layout 轴(`parity`/`width`/`sub_part`)填充。

| op (form) | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vcvt` | B | `V<L×Tn>`, `[pmode]` | `V<L×Tm>` | (i8–i32, f8-f32)↔(i8–i32, f8-f32)|
| `vinterpret_cast` | A | `V<L×T>` | `V<L×T'>` | 任意bit 重解释,显式,无布局推断|


> **`vinterpret_cast`** —— bit 级重解释(`bitcast`)。**不**是 `vcvt`:不产生 parity/width
> 轴、无布局可推断、无 dtype cast 链,只是把同一组 bit 按新 dtype 读出。因此
> Category 留空、不带 `[pmode]`,刻意保留为显式操作(作者自行保证语义合法)。

示例:

```mlir
// 宽化 16→32(radix-2,parity EVEN/ODD 由 pto.as 展开)
%w = pto.vmi.vcvt %a, %mask : !pto.vmi.vreg<128xf16>, !pto.vmi.mask<128> -> !pto.vmi.vreg<128xf32>

// 窄化 32→16
%n = pto.vmi.vcvt %a, %mask : !pto.vmi.vreg<128xf32>, !pto.vmi.mask<128> -> !pto.vmi.vreg<128xf16>

// 量化 f32 → fp8
%q = pto.vmi.vcvt %s, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xfp8>

// bit 重解释(显式,不经 vcvt)
%r = pto.vmi.vinterpret_cast %a : !pto.vmi.vreg<64xf32> -> !pto.vmi.vreg<64xi32>
```


---

## Part 9 —— Group 7:SFU

特殊功能/领域加速器操作。混合类别:`chistv2` 产生 `half` 轴(B);sort 与
gather/scatter 是 Category-C tile/permute 操作;融合激活/算术操作是 Category-A
`vreg→vreg`。

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vhist` | B | `V<L×i*>` (bin idx), `[pmode]` | `V` (Bin_N0/N1 counts, half 轴) | i8–i32 (bin index) |
| `vgather` | C | `Ptr<T>`, `%idx`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vgatherb` | C | `Ptr<T>`, `%idx`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vscatter` | C | `V<T>`, `%idx`, `Ptr<T>`, `[pmode]` | — | i8–i32, f16/bf16, f32 |
| `vexpdif` | A | `V<f*>` (x), `V<f32>` (max) †, `[pmode]` | `V<f32>` | f16/f32 → f32 |
| `vaxpy` | A | `V<T>` (x), `V<T>` (y), `s` (α) †, `[pmode]` | `V<T>` | f16, f32 |
| `vlrelu` | A | `V<T>`, `[pmode]` | `V<T>` | f16, f32 |
| `vprelu` | A | `V<T>`, `s`/param, `[pmode]` | `V<T>` | f16, f32 |
| `vmull` | B | `V<i32>`, `V<i32>`, `[pmode]` | `V<i64>` (hi+lo, 2 reg; 产生 `width` 轴) | i32/u32 |
| `vmula` | A | `V<T>` (acc), `V<T>`, `V<T>` †, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |

示例:

```mlir
// 直方图/per-bin 计数
%h = pto.vmi.vhist %bin_idx, %mask : !pto.vmi.vreg<256xi8>, !pto.vmi.mask<256> -> !pto.vmi.vreg<256xi16>

// 索引 gather(B32 / byte)
%g = pto.vmi.vgather %src, %offsets, %mask : !pto.ptr<f32, ub>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
%gb = pto.vmi.vgatherb %src, %offsets, %mask : !pto.ptr<i32, ub>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<256> -> !pto.vmi.vreg<256xi32>

// 索引 scatter
pto.vmi.vscatter %v, %dest, %offsets, %mask : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<64>

// 融合 exp(x − max)(softmax)
%e = pto.vmi.vexpdif %x, %max, %mask, "EVEN" : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>

// 融合 α·x + y
%y = pto.vmi.vaxpy %x, %acc, %alpha, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>

// leaky / parametric ReLU
%lr = pto.vmi.vlrelu %x, %slope, %mask : !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
%pr = pto.vmi.vprelu %x, %alpha, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>

// 宽化 32×32→64 乘(产生 width 轴,hi+lo 两 reg)
%res = pto.vmi.vmull %a, %b, %mask : !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xi64>

// 融合乘加
%acc = pto.vmi.vmula %acc, %a, %b, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
```

---

## Part 10 —— Group 8:Predicate ops

`pset`/`pge`/`plt` 的 mask 族(`b8/b16/b32`)由返回类型 `M<L>` 所修饰的数据类型推导,**不**进 op 名
(即 `pset : !pto.vmi.mask<L>`,而非 `pset_b32`)。族必须与所修饰的数据族对齐。

| op | Mask in | In | Out | Datatypes |
|---|---|---|---|---|
| `pset` | gen | — (命名模式 `PAT_*`) | `M` | b8/b16/b32 |
| `pge` | gen | — (lane-count 模式 `PAT_VLn`) | `M` (tail) | b8/b16/b32 |
| `plt` | gen | `s` (i32, 如 `%rem`) | `M` (tail), `s`(next) | b8/b16/b32 |

示例:

```mlir
// 物化全 active / tail 模式 mask(gen,无输入)
%all  = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<16>
%tail = pto.vmi.pge "PAT_VL16" : !pto.vmi.mask<16>   // 前 16 个 lane active

// 数据相关 tail mask(从标量剩余计数生成)
%mt, %next = pto.vmi.plt %rem : i32 -> !pto.vmi.mask<16>, i32

```

---

## Part 11 —— Group 9:Data rearrange

寄存器内数据搬移与置换,不访问 UB。`vintlv`/`vdintlv` 是 **A 类** op:逐 lane、dtype
一致、输入与输出同 `L` 同 `T`,且**不改变 vreg 的 layout**。

使用场景常见于一个向量寄存器当中连续存储数据的：
* 实部 + 虚部
* value + index

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vintlv` | A | `V<L×T>`, `V<L×T>`, `[pmode]` | `V<L×T>`, `V<L×T>` | i8–i32, f16/bf16, f32 |
| `vdintlv` | A | `V<L×T>`, `V<L×T>`, `[pmode]` | `V<L×T>`, `V<L×T>` | i8–i32, f16/bf16, f32 |

示例:

```mlir
// 交织:两路源按 even/odd 合并成成对结果(逻辑:low/high 两半区)
// low  = {lhs[0], rhs[0], lhs[1], rhs[1], ...}
// high = {lhs[L/2], rhs[L/2], lhs[L/2+1], rhs[L/2+1], ...}
%lo, %hi = pto.vmi.vintlv %a, %b, %mask
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64>
      -> !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>

// 解交织:成对源按 even/odd 拆分(AoS → SoA)
// lo = {lhs[0], lhs[2], lhs[4], ...}   // even
// hi = {lhs[1], lhs[3], lhs[5], ...}   // odd
%even, %odd = pto.vmi.vdintlv %x, %y, %mask
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64>
      -> !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>
```

---

## Part 12 —— 端到端示例:Block MX Quant

同一份 Block MX Quant 量化逻辑,从上到下展开三层抽象:先看 TileLang 如何组织整个 kernel,然后看传统 pto.mi 在向量层暴露了多少硬件细节,最后看 pto.vmi 如何通过语义级算子消除这些负担。

### 12.1 TileLang 层:完整的 kernel 算法

TileLang 负责组织整个 block-mx quant kernel——block 划分、amax 归约、scale 生成,然后把量化执行路径交给下层:

```python
# ===========================================================================
# per_block_cast_kernel:Block MX Quant 的完整 kernel
#   - Grid: (ceil_div(num_tokens, block_m), ceil_div(hidden, block_k))
#   - Threads: 256
#   - 每个 block 处理 block_m × block_k 个元素
#   - block_k 方向上每次处理 256 个元素
# ===========================================================================
@T.prim_func
def per_block_cast_kernel(
    x: T.Tensor[(num_tokens, hidden), in_config.dtype],          # 输入:f16 矩阵
    out: T.Tensor[(num_tokens, hidden), out_config.dtype],       # 输出:fp8 矩阵
    out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],  # scale factor
):
    with T.Kernel(
        ceil_div(num_tokens, block_m),   # grid dim 0
        ceil_div(hidden, block_k),       # grid dim 1
        threads=num_threads              # 256 threads per block
    ) as (pid_x, pid_y):
    ...
        if:
            for i, j in T.Parallel(block_m, block_k):
                out[...] = x_fragment[i, j] * sf_fragment[i // num_per_tokens, j // num_per_channels]  # 3. 量化写回
        else:
          ...
```

### 12.2 传统 pto.mi(CCE) 写法:算法被物理细节淹没

同样的量化主路径——已有 reciprocal scale,把一块 `f16` 数据乘 scale 后转 `fp8` 写回——用底层 MI 指令直接编写时,所有物理细节都会暴露在代码中。

> 这里处理的是 TileLang 中 block_k 方向上的一次 256 元素的子块。

```mlir
// ===========================================================================
// Block MX Quant 量化执行路径 — 传统 MI 写法
// 处理 num_per_tokens × 256 的分块:load f16 → cvt f32 → mul scale → cvt fp8 → store
// 需要手工处理寄存器拆分、part 选择、mask 粒度、vor 合并等所有物理细节
// ===========================================================================
module attributes {pto.backend = "pto", pto.target_arch = "a5"} {
  func.func @ComputeY1ToFP8_fp16_e4m3_MI(
      %arg0: i16, %arg1: i16,
      %arg2: !pto.ptr<f16, ub>,           // xAddr:输入 f16 数据
      %arg3: !pto.ptr<f16, ub>,           // mxScale1ReciprocalAddr:reciprocal scale
      %arg4: !pto.ptr<f8E4M3FN, ub>,      // y1Addr:输出 fp8 数据
      %arg5: i16, %arg6: i16) attributes {pto.kernel} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %vl_half = arith.index_cast %arg6 : i16 to index
    %load_stride = arith.muli %vl_half, %c2 : index

    pto.vecscope {
      // -- 加载 scale:128 个 f16,dist 模式为 E2B_B16 --
      %scale_128 = pto.mi.vlds %arg3[%c0] {dist = "E2B_B16"}
        : !pto.ptr<f16, ub> -> !pto.mi.vreg<128xf16>

      %mask_b16 = pto.mi.pset_b16 "PAT_ALL" : !pto.mi.mask<b16>   // f16 转换用
      %scale_fp32 = pto.mi.vcvt %scale_128, %mask_b16 {part = "EVEN"}
        : !pto.mi.vreg<128xf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
      %mask_b32 = pto.mi.pset_b32 "PAT_ALL" : !pto.mi.mask<b32>   // f32 运算用
      %mask_b8  = pto.mi.pset_b8  "PAT_ALL" : !pto.mi.mask<b8>    // fp8 写回用

      %block_count = arith.index_cast %arg1 : i16 to index
      scf.for %i = %c0 to %block_count step %c1 {
        %offset = arith.muli %i, %load_stride : index

        // (1) 加载 256 个 f16 → DINTLV_B16 交错拆成 low(128) + high(128)
        %low, %high = pto.mi.vldsx2 %arg2[%offset], "DINTLV_B16"
          : !pto.ptr<f16, ub>, index -> !pto.mi.vreg<128xf16>, !pto.mi.vreg<128xf16>

        // (2) f16 → f32:low 和 high 各自再拆 EVEN/ODD,变成 4 个 vreg<64xf32>
        %cvt_low_even  = pto.mi.vcvt %low,  %mask_b16 {part = "EVEN"} : ... -> !pto.mi.vreg<64xf32>
        %cvt_high_even = pto.mi.vcvt %high, %mask_b16 {part = "EVEN"} : ... -> !pto.mi.vreg<64xf32>
        %cvt_low_odd   = pto.mi.vcvt %low,  %mask_b16 {part = "ODD"}  : ... -> !pto.mi.vreg<64xf32>
        %cvt_high_odd  = pto.mi.vcvt %high, %mask_b16 {part = "ODD"}  : ... -> !pto.mi.vreg<64xf32>

        // (3) 乘 reciprocal scale:4 条独立的 vmul,一一对应上面 4 路 f32
        %mul0 = pto.mi.vmul %cvt_low_even,  %scale_fp32, %mask_b32 : ... -> !pto.mi.vreg<64xf32>
        %mul1 = pto.mi.vmul %cvt_high_even, %scale_fp32, %mask_b32 : ... -> !pto.mi.vreg<64xf32>
        %mul2 = pto.mi.vmul %cvt_low_odd,   %scale_fp32, %mask_b32 : ... -> !pto.mi.vreg<64xf32>
        %mul3 = pto.mi.vmul %cvt_high_odd,  %scale_fp32, %mask_b32 : ... -> !pto.mi.vreg<64xf32>

        // (4) f32 → fp8:4 条 vcvt,分别打入 P0/P1/P2/P3 part
        %p0 = pto.mi.vcvt %mul0, %mask_b32 {part = "P0", rnd = "R", sat = "SAT"} : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %p1 = pto.mi.vcvt %mul1, %mask_b32 {part = "P1", rnd = "R", sat = "SAT"} : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %p2 = pto.mi.vcvt %mul2, %mask_b32 {part = "P2", rnd = "R", sat = "SAT"} : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %p3 = pto.mi.vcvt %mul3, %mask_b32 {part = "P3", rnd = "R", sat = "SAT"} : ... -> !pto.mi.vreg<256xf8E4M3FN>

        // (5) 合并:3 条 vor 把 P0~P3 拼回一个完整向量
        %merge01 = pto.mi.vor %p0, %p1, %mask_b8 : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %merge012 = pto.mi.vor %merge01, %p2, %mask_b8 : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %merged = pto.mi.vor %merge012, %p3, %mask_b8 : ... -> !pto.mi.vreg<256xf8E4M3FN>

        // (6) 写回
        pto.mi.vsts %merged, %arg4[%offset], %mask_b8
          : !pto.mi.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, !pto.mi.mask<b8>
      }
    }
    return
  }
}
```

**传统 pto.mi(CCE) 写法的痛点:**

这一层已经不是在描述算法了。每一个"为什么"的答案都不是"算法需要",而是"硬件长这样":

- 一个逻辑向量被拆成 `low` / `high`,原因是物理寄存器只有 128 宽
- `f16 -> f32` 时区分 `EVEN` / `ODD`,原因是类型转换时位宽减半
- `f32 -> fp8` 时分到 `P0`~`P3`,且每条指令需显式携带 `rnd`(舍入模式)、`sat`(饱和溢出)等硬件参数,用错会直接导致数值错误,原因是 fp8 打包需要匹配硬件 part 机制
- 多个 part 用 `vor` 拼回去,因为前面各步骤不得不拆
- 数据从加载开始就以交织(interleave)方式分布在多个寄存器中(`DINTLV_B16` → low/high → EVEN/ODD),原始元素的索引步长变成 4,开发者需要始终在脑中维持这套映射关系才能追踪每个元素
- 每种操作需要选对 mask 粒度(`b16` / `b32` / `b8`),因为不同数据类型的位宽不同,选错即 bug

**loop body 17 条指令,其中 12 条跟量化算法毫无关系,纯粹在描述硬件如何拆分和拼接数据。**

### 12.3 pto.vmi 写法:只写语义,不写硬件

同样的量化主路径——处理 `num_per_tokens × 256` 的分块——用 pto.vmi 编写时,所有物理细节由编译器自动处理:

```mlir
// ===========================================================================
// Block MX Quant 量化执行路径 — pto.vmi surface 语法
// 同样处理 num_per_tokens × 256 的分块,只需 5 个语义动作
// ===========================================================================
module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {

  func.func @ComputeY1ToFP8_fp16_e4m3_VMI(
      %dataLen: i16,                       // 输入数据总长度
      %block_count: i16,                    // 主循环迭代次数 = block_k / 256
      %xAddr: !pto.ptr<f16, ub>,           // 输入 f16 数据地址
      %mxScale1ReciprocalAddr: !pto.ptr<f16, ub>,  // reciprocal scale 地址
      %y1Addr: !pto.ptr<f8E4M3FN, ub>,     // 输出 fp8 数据地址
      %ubBlockSize: i16,                   // UB block 大小
      %vlForHalfNumber: i16)               // half number 的向量长度
      attributes {pto.kernel} {

    // -- 常量 --
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %vl_half = arith.index_cast %vlForHalfNumber : i16 to index
    %load_stride_y8 = arith.muli %vl_half, %c2 : index

    pto.vecscope {
      // =====================================================================
      // 阶段 1:加载 reciprocal scale,广播到 256 宽,转成 f32
      // =====================================================================
      %scale_f16 = pto.vmi.vload %mxScale1ReciprocalAddr[%c0]
        : !pto.ptr<f16, ub> -> !pto.vmi.vreg<8xf16>                // 加载 8 个 f16 scale
      %scale_f16_vec = pto.vmi.vbrc %scale_f16 {group = 8}
        : !pto.vmi.vreg<8xf16> -> !pto.vmi.vreg<256xf16>          // 广播到 256 宽
      %scale_fp32 = pto.vmi.vcvt %scale_f16_vec
        : !pto.vmi.vreg<256xf16> -> !pto.vmi.vreg<256xf32>        // f16 -> f32

      // =====================================================================
      // 阶段 2:逐块量化 — load f16 → cvt f32 → mul scale → cvt fp8 → store
      // 每次处理 256 个 f16 元素
      // =====================================================================
      %block_count_idx = arith.index_cast %block_count : i16 to index
      scf.for %i = %c0 to %block_count_idx step %c1 {
        %x_off = arith.muli %i, %load_stride_y8 : index
        %y_off = arith.muli %i, %load_stride_y8 : index

        // (1) 加载一块 f16 数据(256 个元素)
        %x_f16 = pto.vmi.vload %xAddr[%x_off]
          : !pto.ptr<f16, ub> -> !pto.vmi.vreg<256xf16>

        // (2) f16 -> f32
        %x_fp32 = pto.vmi.vcvt %x_f16
          : !pto.vmi.vreg<256xf16> -> !pto.vmi.vreg<256xf32>

        // (3) 乘 reciprocal scale
        %res_fp32 = pto.vmi.vmul %x_fp32, %scale_fp32
          : !pto.vmi.vreg<256xf32>, !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xf32>

        // (4) f32 -> fp8 (e4m3)
        %res_fp8 = pto.vmi.vcvt %res_fp32
          : !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xf8E4M3FN>

        // (5) 写回 fp8 结果
        pto.vmi.vstore %res_fp8, %y1Addr[%y_off]
          : !pto.vmi.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, !pto.vmi.mask<256>
      }
    }
    return
  }
}
```
