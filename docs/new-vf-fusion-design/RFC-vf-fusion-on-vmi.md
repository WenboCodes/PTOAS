# RFC:VF fusion 2.0 —— 基于 `pto.vmi` 的重构

[toc]

---

## 1. 背景

PyPTO2 框架下的 VF(向量函数)融合是一个**后端独立 pass**(`HiIPUVFLoopFusion` + `HiIPUVFLoadStoreElim`):它先生成各自独立的 tileop loop,再事后把它们合并、消除 store-load。这套方法在追求泛化性高性能时撞上了结构性困境,详见 [PyPTO2-vf-fusion-analysis.md](./PyPTO2-vf-fusion-analysis.md)。困境的根源不在 pass 的实现细节,而在**判据层**:

- 融合资格判断在 `CCE/LLVM IR` 做 —— 用 SCEV 算 trip-count、用 LLVM-IR 别名分析判 UB 重叠、用循环结构兼容性判可融。
- CCE层信息不足:CCE Intrinsic API 的 `vlds`/`vsts` 把地址折叠成标量 offset + POST_UPDATE,dist 模式、scatter、in-place、padding 混杂,别名分析在 MayAlias 时只能保守拒绝,错失大量合法融合。
- 循环结构不兼容:Reduce/Expand 与 element-wise 的循环遍历方向不同,事后融合时 trip-count/mask/stride 对不上。

本 RFC 提议:**用一个运行在 `pto.vmi` 逻辑层的 VF-fusion pass + mem2reg pass,替换整条 PyPTO2 在LLVM IR中做的 fusion pipeline**。`pto.vmi`(见 [PTO-vmi-design.md](./PTO-vmi-design.md))是夹在 TileOP 与物理 `pto.mi` 之间的逻辑向量 ISA:只暴露逻辑连续语义,物理 SIMD 寄存器 layout 由 `pto.as` 持有、对用户不可见。把融合判据从物理层上移到逻辑层,病根随信息层一起变 —— 这是本设计的核心论点。

### 1.0 架构总览

整条流水线的逻辑流:

```
TileOP       用户调用模板写独立 tileop
   │        (基于 pto.vmi op 的模板库,非独立 dialect)
   │  ── 模板展开 ──▶  每个 tileop → 一个 scf.for over N×VL + vmi ops;
   │                   tileop 间经 UB 传递数据
   ▼
pto.vmi      展开后的逻辑向量 IR —— 多个独立 tileop loop
   │        (layout 由编译器持有,对用户不可见)
   │  ── vf 融合 ────▶  判据层在 vmi(trip-count / 相邻 / 别名 / access-pattern);
   │                   合并兼容 tileop loop 为共享 loop;
   │                   mem2reg 消 tileop 间 UB store-load → vreg 直传
   ▼
pto.vmi      融合后的逻辑向量 IR —— 共享 loop
   │        (reduce ⊕ element-wise 同循环、tileop 间 vreg 直传)
   │  ── vmi lowering ─▶  pto-as 做 layout-assignment + lowering;
   │                       post-update / 地址模式等物理细节在此生成
   ▼
pto.mi       vcvt EVEN/ODD + 两路 vadd + vstsx2 INTLV_B32
             ← 物理:SIMD 寄存器交织细节
```


### 1.1 tileop 实现与融合的版本复杂度

PyPTO2 里,一个 tileop 要维护**多个实现版本**,根源是两个正交维度各分情况:

- **2D tile 合轴 vs 非合轴**:内层与外层合成单层(1D,合轴)还是保持 2D 嵌套。合轴循环结构统一、利于融合,但不利用 2D 局部性;非合轴反之。
- **末尾地址 post-update vs not**:地址推进用 post-update(边算边步进指针)还是每轮重算。post-update 省地址计算,但地址模式复杂、别名分析难。

两维度叉乘,每个 tileop 维护 2×2 起步的多版本;叠加精度/尾块/unroll 等维度后指数膨胀([PyPTO2-vf-fusion-analysis.md](./PyPTO2-vf-fusion-analysis.md) §4.3 难点 2"维度爆炸")。融合时编译器还要判两 tileop 版本是否兼容、该选哪个,复杂度进一步上升。

**VMI 的解法 —— tileop 变单层循环**:

Class 2 的 tile 约束(§4.1:surviving axis = VL)带来一个结构性改变:**tileop 在 VMI 下只有单层循环(`scf.for` over `N`),不再有 2 层 nested loop**。因为 inner 轴是 VL = 单个 vreg,不是循环轴;只有 outer(`N`)是循环。这直接让"合轴/非合轴"这个版本维度**根本不存在** —— PyPTO2 的 2D nested-loop 形态在 VMI 下消失,不存在"合轴 vs 非合轴"的选择。

> **例子(f32 dtype)**:
f32 的 bitwidth=32,一个物理 vreg 容量 = 256B/4B = **64 个元素**,故 `VL=64`。VMI 下,**只支持 inner 轴是 64 的 f32 tile**(如 `64×64`、`128×64`、`256×64`),其他 inner shape 的 f32 tile(如 inner=32、inner=100)**不支持**。inner 被钉死成单个 vreg(64 元素),不再是循环轴,故 tileop 天然单层循环;不存在"内层 32 要不要合轴"这类选择。要处理更多元素,撑大 outer(`N`),不撑 inner。

剩下"末尾地址 post-update vs not"这维度,也不上浮到 tileop/融合层 —— **pto-as 编译 pass** 在 lowering 时按需生成 post-update / 地址模式,这些是物理层细节,归 pto-as。

tileop 层只写一个语义版本(单层循环),合轴选择不存在、地址模式由 pto-as 承接 —— 这正是"判据层上移 + 物理细节下沉"的另一体现。

## 2. 问题二分:Class 1 与 Class 2

PyPTO2 的融合病根可分成两类,解决机制根本不同:

- **Class 1 —— 抽象不足**:融合失败是因为 `pto.mi` op 携带的语义不够(load/store 只会连续访存、cast 带全局寄存器副作用、dist 模式阻断 store-load 消除、tileop 版本爆炸...)。**VMI 更丰富的 op surface + 约束把这类问题在构造期消解** —— `vload`/`vstore` 带 dist-mode、`vcvt` 把 cast 链当 compiler-internal layout、mask 是一等值、tile 形状约束消除版本选择。
- **Class 2 —— 寄存器直传(register direct-pass)**:Reduce/broadcast ⊕ element-wise 融合失败,是因为 reduce 的结果/输入无法在一个静态寄存器上直传跨 element-wise 循环。**抽象丰富度解不了这类问题** —— 需要一个 tile-shape 约束,让 reduce 的 acc 塌成单个 vreg。

Class 1 靠"换 IR 层 + 加约束"自动消解;Class 2 靠"tile 约束"专门解决。两者的设计分别见 §3、§4。注意 §1.1 的 tileop 版本复杂度属 Class 1 —— 它是"实现细节上浮到融合层"的典型,由约束 + pto-as 下沉消解。

## 3. Class 1:VMI op surface 消解的 PyPTO2 病根

下表把 PyPTO2 分析文档逐条扫出的病根,对照 VMI 的解法。**"彻底"列诚实标注解的程度** —— 有的是根治,有的是兜底,有的其实是回避(归入开放问题)。

| PyPTO2 病根 | 出处 | VMI 解法 | 彻底? |
|---|---|---|---|
| 动态 trip-count,SCEV 算不出即拒 | §5.1.1 | 融合判据改判"两 loop 是否共享同一 `%valid_row` SSA 值"([ADR-0002](./adr/0002-static-n-dynamic-valid-data-via-mask.md)),不要求 SCEV 算常量 | 彻底 |
| Alias MayAlias 即拒(W-W/WAR) | §5.1.2 | shaped ptr + 多维 index 精确算 UB 区域,区域不重叠即可融([ADR-0003](./adr/0003-vload-vstore-multidim-index-shaped-ptr.md)) | 彻底 |
| Alias RAW:store 有 predicate、load 无 | §5.1.2 | 区域精确 + A5 load 不可谓词化仍靠消费侧 `[pmode]` 兜底 | 兜底 |
| Access-pattern SCEV 步长不兼容 | §5.1.3 | shaped ptr + 多维 index,步长从指针类型取([ADR-0003](./adr/0003-vload-vstore-multidim-index-shaped-ptr.md)) | 彻底 |
| LoadStoreElim 不支持 DINTLV dist | §5.2.3 | mem2reg 按 index location 消,不看 dist-mode([ADR-0004](./adr/0004-vmi-mem2reg-after-fusion.md)) | 彻底 |
| LoadStoreElim 标量 UB 阻断扫描 | §5.2.4 | mem2reg 只提升 vreg store-load,标量不阻断 | 彻底 |
| LoadStoreElim Preg 不匹配 | §5.2.3 | vmi的vload默认也不带mask | **待定** |
| TCast 全局寄存器副作用阻断融合 | §3.2.3/§5.2.4 | `vcvt` 把 `sat`/`rnd` 当 per-op 属性,不写全局寄存器 | 彻底 |
| TCast 位宽变化致元素数不一致 | §3.2.3 | `vcvt` 逻辑 `L` 不变,只 T 变;物理 K 变化归 pto.as | 彻底 |
| RowExpandBin vlds 在两层 for 间致 MEM_BAR | §3.2.4 | `vbrc` 是寄存器内广播 op,无 vlds-in-outer-loop 结构 | 彻底 |
| tileop 版本爆炸(合轴/post-update 叉乘) | §4.3 | tile 约束消合轴选择,pto-as 承接地址模式(§1.1) | 彻底 |
| Reduce/Expand 循环结构不兼容 | §3.3.1/§4.2 | Class 2 同循环(RowMax-style)/拆两循环+mem2reg(ColMax-style),见 §4 | 大体 |
| **TColSum Binary vs 非 Binary 精度建模** | §3.2.2 | reduce-A(element-wise+acc)统一写法,**不暴露精度旋钮** | 待定 |
| CodeGen 不感知 VF 融合 | §3.1.3/§4.1 | 上游 TileOP 模板库直接写 tileop,判据在 vmi 层 | 大体 |

> **诚实标注**:表中"Binary vs 非 Binary 精度建模"一行**不是收益** —— VMI 的 reduce-A 顺序累加绕开了 Binary/非Binary 的分叉,等于没提供精度旋钮,而非解决了精度建模。这归入 §10 开放问题。

## 4. Class 2:tile 约束与寄存器直传

### 4.1 核心契约:surviving axis = VL

让 reduce 的 acc 塌成单个 vreg,关键是一个 tile-shape 约束(详见 [ADR-0001](./adr/0001-tile-shape-n-times-vl.md)):

- **`VL` = 物理 256B vreg 的元素容量,随 dtype**:`VL = 256B / bitwidth(T)`(f32/i32→64,f16/bf16/i16→128,fp8/i8→256)。VL 不是自由选择,由 dtype 决定。
- **任何 reduce 的 surviving axis(acc 所在轴)必须等于 VL**,使 acc 恰为 `V<VL×T>` = 一个物理 256B vreg。

为什么 VL 必须绑 dtype:若 VL 是 dtype 无关的逻辑计数,surviving axis 可能逻辑上是 VL 却跨多个物理 vreg,寄存器直传立即破裂。VL 绑物理 vreg 容量,才让"surviving axis = VL"与"acc = 一个物理寄存器"重合。

**图示(以 f32、VL=64 为例)**:

```
tile = N × VL  (f32 → VL = 256B/4B = 64)

        inner axis = VL = 64 ──────────────▶
        ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐   row 0
        │x │x │x │..│  (一个物理 vreg = 64 个 f32)       │   ┐
 N      ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤   row 1  │
(outer) │x │x │x │..│                               │   │   ├── 每行一个 vreg
        ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤   row 2  │
        │  │  │  │  │  ...                          │   │   │
        ...                                           ...   │
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘   row N-1 ┘

两种 reduce 取向(acc 都是单 vreg,但生命周期不同):

  ColMax-style(沿 N 压、survive on inner=VL):
        acc = ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐   ← 一个物理 vreg
              │m0│m1│m2│..│  (64 个列 max,每列一个)       │     (V<64×f32>)
              └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
        acc 跨所有 N 轮累积,跑完才完整 → 不能同循环(见 §4.2)

  RowMax-style(沿 inner VL 压、survive on outer,要求 N=VL):
        acc = ┌──┬──┬──┬──┐
              │r0│r1│..│  │   ← VL 个行 max(逐行产出、当轮可消费)
              └──┴──┴──┴──┘
        每轮独立算出当行 max、当轮给 element-wise 用 → 可同循环(见 §4.2)

若 surviving axis ≠ VL(假设允许 inner=32,f32 仍 64/vreg):
        inner=32 → 每行只填半 vreg,acc 跨多个半 vreg → 寄存器直传破裂 ✗
        (这正是 VL 必须绑 dtype、surviving axis 必须等于 VL 的原因)
```

> **未来工作**:当前 VL 由 dtype 决定(每 dtype 唯一合法 VL)。未来可扩展为 `VL ∈ {64,128,256}` 自由选择(如 f32 tile 取 VL=128 跨 2 物理 vreg),用 **group reduce**(`{group=C}`,见 §4.5)把 acc 维持成单 vreg。当前 pass 不支持。

### 4.2 acc 生命周期决定能否 loop 融合

"surviving axis = VL"是 acc 成单 vreg 的**必要**条件,但**不充分** —— 同循环融合还取决于 **acc 的生命周期**:acc 是逐轮产出当轮可消费,还是要跨所有轮累积、跑完才完整。两种 reduce 取向:

- **ColMax-style**(沿 N 压、survive on inner=VL):acc = "VL 个列 max",每个列 max 要看遍所有 N 行才完整 → acc **跨所有轮累积、跑完才完整** → element-wise 要等 acc 完整 → **拆两循环**(循环1 reduce 出 max,循环2 `x-max`)。reduce 用 `vmax`(element-wise 指令,**代价低**)。
- **RowMax-style**(沿 inner VL 压、survive on outer,要求 `N=VL`):acc = "逐行 max",每轮在一行内部沿 VL 压出**当行 max**、当轮即可给该行 element-wise 用 → **同循环融合**(每轮 `load row i → reduce 出 m_i → x_i - m_i`)。reduce 用 `vcmax`(整行规约指令,**代价大**)。

**图示(以 f32、VL=64、N=64 为例)**:

```
━━━ ColMax-style:沿 N 压,acc 跨所有轮累积、跑完才完整 ━━━

   循环1 (reduce,跑完 N 轮 acc 才完整)      循环2 (element-wise,等 acc 完整)
        ┌──┬──┬──┬──┐ row 0                    ┌──┬──┬──┬──┐ row 0
        │x │x │..│  │   │  累加进 acc           │x-│x-│..│  │
        ├──┼──┼──┼──┤ row 1                   ├──┼──┼──┼──┤
        │x │x │..│  │   ▼                      │x-│x-│..│  │    拆两循环
        ├──┼──┼──┼──┤       ──────▶           ├──┼──┼──┼──┤    (acc 经 UB 传递)
        │..   ..  │ row N-1                    │..   ..  │
        └──┴──┴──┴──┘                          └──┴──┴──┴──┘
        acc = [m0 m1 ..] ← 跑完 N 轮才完整      每列减同列 max
        (用 vmax,element-wise,代价低)         (不融,但每条指令便宜)

   不能同循环:acc 跨所有轮累积,跑完才完整 ✗


━━━ RowMax-style:沿 inner VL 压,acc 逐轮产出、当轮可消费 ━━━

   同循环(每轮:load row i → reduce 出 m_i → x_i - m_i)
        ┌──┬──┬──┬──┐ row 0
        │x→│x→│x→│x→│   沿 inner 压出 m_0(当行 max)
        ├──┼──┼──┼──┤       ↓ 当轮消费
        │x→│x→│x→│x→│   x_0 - m_0  ← 同一轮算出、同一轮用
        ├──┼──┼──┼──┤
        │..   ..  │
        └──┴──┴──┴──┘
        每轮独立算出当行 max、当轮给 element-wise
        (用 vcmax,整行规约,代价大,但能融)

   能同循环:acc 逐轮产出、当轮可消费 ✓
```

> **关键洞察**:ColMax 和 RowMax 是个**两难权衡** —— ColMax 不融但用便宜的 `vmax`(element-wise),RowMax 能融但用昂贵的 `vcmax`(整行规约)。选哪种是 cost-model 问题,不能靠"能融就融"的规则。这归入 §12 开放问题。详见 [ADR-0001](./adr/0001-tile-shape-n-times-vl.md)。

### 4.3 结构性后果:tileop 单层循环

Class 2 的 tile 约束带来一个特别重要的改变:**tileop 的实现都变成一层循环(`scf.for` over `N`),不需要 2 层 nested loop**。因为 inner 轴是 VL = 单个 vreg,不是循环轴;只有 outer(`N`)是循环。这是寄存器直传的直接后果 —— inner 不再是循环维度,2 层嵌套的必要消失。PyPTO2 的 2D nested-loop 形态(及其"合轴/非合轴"版本选择,见 §1.1)在 VMI 下根本不存在。

### 4.4 二维 Tile 的动态性保留

tile 的动态性分两层保留(详见 [ADR-0002](./adr/0002-static-n-dynamic-valid-data-via-mask.md)),不要混为一谈:

| 层级 | 动态量 | Form A 怎么承接 | Form B 怎么承接 |
|---|---|---|---|
| row 级(跑多少行) | `valid row`(≤ N) | 跑满静态 N,超出行用 mask 屏蔽 | **trip-count 本身动态**(`to %valid_row`) |
| lane 级(末行 lane 尾) | 末行 partial VL | `plt(remaining)`,remaining 是 iter_args 计数器 | `plt(valid_row - %offset)` |

- **`N`** = tile 的 **buffer 容量**(静态,管分配),不是 trip-count。
- **`valid row`** = 实际有效行数(动态,管跑多少)。
- 两 form 都合法,Form B 优先(不跑屏蔽空迭代)。
- acc 的身份静态,**不是因为 N 静态**,而是因为 acc 是 `scf.for` `iter_args` SSA 值跨迭代复用 —— 与 trip-count 静态与否无关。
- 绕过 PyPTO2 §5.1.1 的机制:vmi pass 的 trip-count 一致性检查改判"两 loop 是否共享同一 `%valid_row` SSA 值"(trivially 一致,不需 SCEV)或 affine 可证等价。
- RowMax-style 下 `N=VL`,而 VL 是 dtype 决定的静态值,故 RowMax-style 的 N 静态=VL —— 但 `valid row` 仍动态,Form A/B 仍叠加其上。

### 4.5 范围边界:`{group=C}` compact 归约不参与同循环融合

`vcadd`/`vcmax`/`vcmin {group=C}`(沿单 vreg 内 lane 压,产 compact 标量向量 `V<C×T>`,C∈{1,2,4,8})是 Category-B op([PTO-vmi-design.md](./PTO-vmi-design.md) §2.2),**不参与** Class 2 的同循环 element-wise 融合 —— 其 acc 形状 `V<C×T>` 与 element-wise 的 `V<VL×T>` 操作数不对齐。它仍存在,经 `vbrc {group=C}` 广播回 `V<VL×T>` 才接 element-wise。**tiling 阶段被禁止产生需要它的 compact tile**。这是有意的、易逆转的范围限定,非架构锁。

> **注意 `vcmax` 重载**:`vcmax {group=C}`(本节,compact lane-group,出范围)与 `vcmax` 无 group 属性(整行规约,RowMax-style 用,在范围内、可同循环融合但代价大)是**不同 op** 共用一名。

## 5. fusion pass:判据层上移

vmi fusion pass 形态上与 PyPTO2 同构(事后融合独立 tileop loop),但判据从 `pto.mi` 物理层上移到 `pto.vmi` 逻辑层。五项资格判据对照:

| PyPTO2 检查 | vmi 层状态 |
|---|---|
| (1) trip-count 一致(SCEV) | 保留,MLIR 层;同 `%valid_row` SSA 值或 affine 等价([ADR-0002](./adr/0002-static-n-dynamic-valid-data-via-mask.md)) |
| (2) 相邻 | 保留,简化(无 mi 的 POST_UPDATE 副作用) |
| (3) guard 一致 | 消失(`scf.for` 无显式 guard) |
| (4) 依赖/别名 | 保留,但 MLIR 层 tile 分析替代 LLVM 层;W-W/WAR 根治,RAW 兜底([ADR-0003](./adr/0003-vload-vstore-multidim-index-shaped-ptr.md)) |
| (5) access-pattern 兼容 | 保留,shaped ptr + 多维 index([ADR-0003](./adr/0003-vload-vstore-multidim-index-shaped-ptr.md)) |

判据所需的结构化信息,来自两个 IR 设计(详见 [ADR-0003](./adr/0003-vload-vstore-multidim-index-shaped-ptr.md)):

- **`vload`/`vstore` 取多维 index 表达式**(`%a[%i, %j]`),IR 不折叠 —— 折叠会丢失分析所需的结构。
- **shape/stride 挂指针类型**(`!pto.ptr<T,ub,shape=[...],stride=[...]>`,memref 风格),index 只含归纳变量。
- access-pattern 分析只看归纳变量 + 静态 stride;`%valid_row` 只进 trip-count 检查,不进 access-pattern —— 两者正交。
- 物理地址折叠延迟到 `pto.as` lowering,不在 vmi IR 做。

## 6. mem2reg pass:消除 tileop 间 UB store-load

PyPTO2 把"合并 loop"和"消除 store-load"拆成两个 pass,且消除在融合**之前**跑 —— 融合后新暴露的 store-load 永远消不掉(§5.3 顺序病)。vmi 用**单个 mem2reg pass,在 fusion 之后跑**(详见 [ADR-0004](./adr/0004-vmi-mem2reg-after-fusion.md)):

- fusion 合并 loop → mem2reg 在同 loop 内把同 location 的 `vstore`→`vload` 提升为 vreg SSA 直传。
- 修了顺序病:消在融后,所有 pair 已同 BB。
- 解了 LoadStoreElim 的两保守病:按 index location 消(不看 dist-mode,DINTLV pair 可消)、只提升 vreg(标量 UB 不阻断)。**Preg 不匹配仍待定** —— vmi 的 `vload` 默认不带 mask(同 PyPTO2 的 load 无 predicate),这个不对称 vmi 没动(见 §3 表、§12 开放问题)。
- **能力边界**:提升"同 location 同形状"的整体 store→load,与 reduce 取向无关。RowMax-style(acc 同循环直传,无需 UB)和 ColMax-style(循环1 `vstore %acc`→循环2 外部 `vload`,整体)都可消;ColMax 的循环2 内部怎么消费提升后的值(按列对齐等)是 post-promotion SSA,不涉及 UB。
- **与 reduce acc 的关系**:RowMax-style acc 是 in-loop `iter_args`,不经 UB,mem2reg 不碰(正交);ColMax-style acc 跨循环经 UB,mem2reg **会**消。mem2reg 永不碰的是 in-loop acc 更新(每轮 `vcmax`/`vmax` 进 `iter_args`),那是寄存器 SSA。

## 7. 融合覆盖面

四类可融合场景:

1. **element-wise ⊕ element-wise**(基础):两 element-wise tileop loop 共享 `%valid_row`、相邻、access-pattern 兼容、依赖 RAW(shaped ptr 精确 + mask 兜底) → 同循环融合,中间数据经 mem2reg 走 vreg。
2. **RowMax-style reduce ⊕ element-wise**(acc 逐轮产出):同循环,每轮 `load row i → reduce 出 m_i → x_i - m_i`,acc 当轮产出当轮消费,无 `N×VL` UB 中间量。reduce 用 `vcmax`(整行规约,代价大)。
3. **ColMax-style reduce → element-wise**(acc 跑完才完整):拆两循环,acc 跨所有轮累积、跑完才完整,循环1 reduce 出 max、循环2 `x-max`,跨循环 acc 经 mem2reg 消。reduce 用 `vmax`(element-wise,代价低)。
4. **reduce → broadcast → element-wise**(跨段):reduce 跑完出 max(`V<VL×T>` 或标量),`vbrc` 把它扇回到 `V<VL×T>` 再给 element-wise。整条链跨多段、各段经 UB 传递,mem2reg 消同 location 同形状的整体 store→load(reduce→broadcast 的产出、broadcast→element-wise 的输入),数据走 vreg 直传。softmax 的 `RowMax → brc → x/max` 即此场景(ColMax-style reduce 产 max、brc 扇回、element-wise 消费)。

**部分融合**:非 all-or-nothing。一组兼容的 tileop loop 可融,其余留独立;每个独立组分别 lower。

## 8. 输入 IR:TileOP 模板库

vmi fusion 的上游是 **TileOP —— 一个基于 `pto.vmi` op 的模板库(非独立 dialect)**。用户调模板写独立 tileop,每个 tileop 展开成一个 `scf.for`(over `N×VL`)+ vmi ops。fusion pass 的输入是这组展开后的 vmi IR,识别其中的独立 tileop loop 为 IR pattern —— **不消费专门的 TileOP op**。TileLang 只是 vmi IR 的一个更高层生产者([PTO-vmi-design.md](./PTO-vmi-design.md) §12 示例),不是 fusion pass 的设计目标。

## 9. 验证与性能目标

- **正确性验证**:lit test(Before/Expected/Actual 模式,按项目 testing 规则)+ A5 board/remote 数值比对。重点:mask 传播、mem2reg 提升、ColMax/RowMax 路径选择。
- **性能目标**:以 softmax / MX-quant 为 benchmark,目标追平手写融合。融合失败时退化为各自 VF launch + UB round-trip(PyPTO2 现状性能)。

## 10. 实现里程碑

1. f32 + element-wise ⊕ element-wise + mem2reg
2. RowMax-style reduce ⊕ element-wise(同循环;reduce 用 `vcmax` 整行规约)
3. ColMax-style(拆两循环 + mem2reg 消跨循环;reduce 用 `vmax` element-wise)
4. 扩 dtype(f16/fp8)
5. group reduce 解绑 VL(未来工作)
6. ColMax vs RowMax cost-model 接入(开放问题)

## 11. 对 `pto.vmi` 的修改

本 RFC 落地时,需同步修改 [PTO-vmi-design.md](./PTO-vmi-design.md)(跨层同步项,见 `.claude/rules/cross-layer-sync.md`):

- **§1.1 VL 定义**:从"`L` 是 64 的倍数(dtype 无关)"收紧为"surviving axis = dtype 的 VL(256B/bitwidth)"。
- **§3 `vload`/`vstore` 签名**:从标量 offset `vload %ub[%offset]` 扩为多维 index `vload %a[%i,%j]`;`!pto.ptr` 加 shape/stride。
- ODS、C++ verifier、lowering pattern、tests 同步。

## 12. 开放问题

- **精度建模**:reduce-A(element-wise+acc)绕开了 Binary/非Binary 精度旋钮,等于回避了"何时可安全用低精度非Binary"的建模(§3 表中 TColSum 行)。当前保守用顺序累加;未来若需精度旋钮,需补精度建模能力。
- **ColMax vs RowMax 的 cost-model**:ColMax 不融但用便宜的 `vmax`(element-wise),RowMax 能融但用昂贵的 `vcmax`(整行规约)。选哪种是"不融+便宜指令" vs "融+贵指令"的权衡,需 cost-model 接入,不能靠"能融就融"的规则。当前 pass 暂不决策,留作开放问题。
- **group reduce 解绑 VL**:当前 VL 由 dtype 决定;未来用 `{group=C}` 让 VL 在 `{64,128,256}` 自由选择(§4.1 未来工作)。
- **RAW 的 mask 兜底**:A5 load 不可谓词化,RAW 安全性仍靠消费侧 `[pmode]` 正确屏蔽 padding。mask 传播是 load-bearing 的,任何 in-loop op 丢 mask 即尾块错误。

## 13. ADR 索引

- [ADR-0001](./adr/0001-tile-shape-n-times-vl.md) — tile 形状 `N×VL`,surviving axis 钉到 VL,acc 生命周期决定融合
- [ADR-0002](./adr/0002-static-n-dynamic-valid-data-via-mask.md) — buffer 容量 N 静态、valid row 动态,双层动态性
- [ADR-0003](./adr/0003-vload-vstore-multidim-index-shaped-ptr.md) — vload/vstore 多维 index,shaped ptr
- [ADR-0004](./adr/0004-vmi-mem2reg-after-fusion.md) — vmi mem2reg pass,在 fusion 之后
- [CONTEXT.md](./CONTEXT.md) — 本设计的术语表(glossary)
