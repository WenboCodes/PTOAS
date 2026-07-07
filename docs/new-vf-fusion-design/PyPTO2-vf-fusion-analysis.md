# **PyPTO2.0框架下VF融合泛化性难点分析**

## **一、总体结论**

VF融合的难点主要在于以下3部分需要**同时满足**融合要求，但3部分均有各自需要权衡的矛盾：

| **维度**           | **核心矛盾**                         | **待解决问题**                                               |
| ------------------ | ------------------------------------ | ------------------------------------------------------------ |
| **CodeGen**        | OP顺序/同步调度 与 VF融合优先 的权衡 | 如何判断调度的收益和VF融合的收益的大小？                     |
| **底层库实现**     | 通用性、可融合性与性能之间的权衡     | 什么实现是适合融合的？如何保证性能通用性的同时又能够提供可融合的实现？ |
| **编译器融合判断** | 精度和性能之间的权衡                 | 该不该融合？选什么OP实现版本？                               |

在目前框架下，VF自动融合要实现泛化性的高性能仍是巨大的挑战。

------

## 二、VF融合收益主要来源分析

经验上看，从汇编来分析VF融合最大收益来源于以下方面：

| **融合情况**            | **性能影响**                                | **备注**                           |
| ----------------------- | ------------------------------------------- | ---------------------------------- |
| **VF个数减少**          | 取决于减少的VF个数，减少2~3个VF以内影响不大 | 收益来自于VF Launch消耗减少        |
| **深融合+Loop内无同步** | 优化极大                                    | 收益来自于减少大量读写UB和乱序执行 |
| **深融合+Loop内有同步** | 劣化极大                                    | 甚至比不融合的性能还要劣化数倍     |

最优版本为：合轴+深度融合+Loop内无同步

------

## **三、以FA Softmax算子的VF融合优化实战看泛化性难点**

我们以FA的Softmax算子（TileShape [128, 64]）为例，分析识别出的优化出发点，以及对于这些优化点是否具有泛化性进行探讨。

### **3.1 CodeGen调整优化点——OP顺序与同步调度**

#### **3.1.1 初始CCE代码与融合分析**

初始CCE代码（输入Shape为[128, 64]）：

```auto
// [128, 64] -> [128, 64] element-wise
TMul<CastUse2Dim<0, 1>, float>(ubtensor_2, ubtensor_2, 0.0883883461); 
// [128, 64] -> [1, 64] reduce
TRowMaxLine<3>(ubtensor_4, ubtensor_2);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
// [1, 64] -> [128, 64] expand
TExpand<LastUse2Dim<0, 0>, 3>(ubtensor_8, ubtensor_4);
// [128, 64] -> [128, 64] element-wise
TSub<LastUse3Dim<0, 1, 1>>(ubtensor_2, ubtensor_2, ubtensor_8);
// [128, 64] -> [128, 64] element-wise
TExp<pto::ExpAlgorithm::DEFAULT, LastUse2Dim<0, 1>>(ubtensor_2, ubtensor_2);
// [128, 64] -> [1, 64] reduce
TRowSumLine<3>(ubtensor_17, ubtensor_2, ubtensor_18);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
// [128, 64] -> [128, 64] element-wise
TCast<CastUse2Dim<0, 1>, 0, pto::SaturationMode::OFF>(ubtensor_20, ubtensor_2);
```

OP顺序为：**TMUL → TCOLMAX → set/wait flag → TEXPAND → TSUB → TEXP → TCOLSUM → set/wait flag → TCAST**

逐项分析OP之间的可融合情况：

| **OP对**              | **融合可能性** | **原因**                                                     |
| --------------------- | -------------- | ------------------------------------------------------------ |
| TMUL → TColMax        | **不可融合**   | TColMax是列归约操作，需完整遍历数据才能得到全局最大值，融合会出错 |
| TExpand → TSub → TExp | **可融合**     | 均为[128, 64] shape，有上下依赖关系                          |
| TColSum → TCast       | **被同步阻断** | 中间有set_flag指令阻碍融合                                   |
| TColSum与TCast顺序    | **可优化**     | 两者输入均来自TExp输出，无固定顺序关系，但Shape突变会分割融合块 |

#### **3.1.2 优化策略详解**

**优化1：set_flag和wait_flag外提**

同步指令阻碍了TColSum和TCast之间的融合。将set_flag/wait_flag外提至融合OP群的上下界：

```auto
// 优化后：同步指令统一上提/下沉
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);   // 上提至融合群上界
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);    // 上提至融合群上界
// ... 可融合的OP群 ...
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);    // 下沉至融合群下界
```

**优化2：TMulS下放到ColMax后**

TMULS被TColMax阻挡，无法参与后面[128, 64]的融合。利用等价变换（TMuls + TColMax ≡ TColMax + TMuls），以多计算1次vmuls的代价换取128次vmuls参与融合的收益：

```auto
//==================== 修改前 ========================
TMulS<LastUse2Dim<0, 1>, float>(ubTensor_0, ubTensor_0, 0.0883883461);
TRowMaxLine<3>(ubTensor_4, ubTensor_0);
TExpand<LastUse2Dim<0, 0>, 3>(ubTensor_8, ubTensor_4);
TSub<LastUse3Dim<0, 1, 1>>(ubTensor_2, ubTensor_2, ubTensor_8);
TExp<pto::ExpAlgorithm::DEFAULT, LastUse2Dim<0, 1>>(ubTensor_2, ubTensor_2);

//==================== 修改后 ========================
TRowMaxLine<3>(ubTensor_4, ubTensor_0);
// 这里的计算64的元素，仅一次可以计算完
TMulS<LastUse2Dim<0, 1>, float>(ubTensor_4, ubTensor_4, 0.0883883461);
TExpand<LastUse2Dim<0, 0>, 3>(ubTensor_8, ubTensor_4);
// 用深融合的收益抵消掉多计算一次vmul的开销
TMulS<LastUse2Dim<0, 1>, float>(ubTensor_8, ubTensor_8, 0.0883883461);
TSub<LastUse3Dim<0, 1, 1>>(ubTensor_2, ubTensor_2, ubTensor_8);
TExp<pto::ExpAlgorithm::DEFAULT, LastUse2Dim<0, 1>>(ubTensor_2, ubTensor_2);
```

**优化3：TCast与TColSum对调位置**

Shape先从[128,64] reduce到[1,64]，再做[128,64]的element-wise操作，Shape发生突变会被分割为两块。调换顺序后：**TExpand → TSub → TExp → TCast → TColSum** 可以全融合。

修改后的等价CCE代码排布：

```cpp
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
// [128, 64] -> [1, 64] reduce
TRowMaxLine<3>(ubtensor_2, ubtensor_4);
// [1, 64] -> [128, 64] element-wise  (TMulS下放)
TMuls<CastUse2Dim<0, 1>, float>(ubtensor_2, ubtensor_2, 0.0883883461); 
// [128, 64] -> [128, 64] element-wise
TMuls<CastUse2Dim<0, 1>, float>(ubtensor_4, ubtensor_4, 0.0883883461); 
// [1, 64] -> [128, 64] expand
TExpand<LastUse2Dim<0, 0>, 3>(ubtensor_8, ubtensor_4);
// [128, 64] -> [128, 64] element-wise
TSub<LastUse3Dim<0, 1, 1>>(ubtensor_2, ubtensor_2, ubtensor_8);
// [128, 64] -> [128, 64] element-wise
TExp<pto::ExpAlgorithm::DEFAULT, LastUse2Dim<0, 1>>(ubtensor_2, ubtensor_2);
// [128, 64] -> [1, 64] reduce           (TCast调到TColSum前)
TRowSumLine<3>(ubtensor_17, ubtensor_2, ubtensor_18);
// [128, 64] -> [128, 64] element-wise
TCast<CastUse2Dim<0, 1>, 0, pto::SaturationMode::OFF>(ubtensor_20, ubtensor_2);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
```

#### **3.1.3 CodeGen优化泛化性评估**

| **优化点**             | **调整原因**       | **泛化性评估**     | **难点分析**                                                 |
| ---------------------- | ------------------ | ------------------ | ------------------------------------------------------------ |
| set_flag/wait_flag外提 | 阻碍融合           | **不一定能泛化**   | 需要评估set-wait flag外提可能造成的**调度空隙**的端到端性能劣化和VF融合收益的CostModel（PyPTO-VFTune）。Flag外提可能打破MTE2/V/MTE3三流水线的双缓冲调度，导致数据搬运和计算无法overlap，反而在端到端性能上劣化 |
| TMulS下放到ColMax后    | 让更多计算参与融合 | ***\*无法泛化\**** | 需要有**额外计算**和**参与VF融合收益**的复杂CostModel。这是一次基于经验的优化——仅当"多算1次vmuls的代价 < 128次vmuls参与深融合的收益"时才成立，且需要知道ColMax是融合的硬边界 |
| TCast与TColSum对调位置 | 更多融合           | **可以泛化**       | 相同Shape尽量放在一起，涉及到Shape变化的尽量放在OP群的首尾。这是可以规则化的算法 |

**OP顺序调整的核心难点**：

OP顺序的调整本质上是**NP困难的调度问题**：

  1. **OP间的等价变换并非自由**：需要保证语义等价，而判断语义等价需要考虑数据依赖、UB生命周期、同步语义等多重约束

  2. **变换的收益不可预测**：一次变换的收益取决于变换后是否创造了新的融合机会，而融合机会又取决于后续OP的实现细节（PTO-ISA层）

  3. **多目标冲突**：调度优化的目标（流水线overlap最大化）与融合优化的目标（同Shape OP聚合最大化）可能冲突

  4. **CodeGen阶段无法感知VF融合**：当前CodeGen生成OP序列时并不考虑VF融合友好性，导致生成的OP顺序天然不适合融合

### **3.2 库实现调整优化点——PTO-ISA版本的泛化性困境**

解决了OP顺序之后只能实现**理论上的可融合**，具体的融合细节优化还需要PTO-ISA库实现的配合。PTO-ISA是通用性和性能平衡的产物，必须兼顾：

 ● 尾块场景

 ● 可合轴场景

 ● 融合场景

 ● 高性能场景（如二分写法）

 ● 高精度场景

#### **3.2.1 TColMax改写——Shape特化 vs 通用性**

在明确ColMax无法融合的情况下，应该最大化单OP在特定Shape下的实现。PTO-isa通用的TColMax对[128, 64] Shape特化无法达到最优。

**为什么[128×64]这个shape下可以有特殊的优化实现？**

 ● column为64且数据大小为4Byte时，外层Loop正好为1，减少VLOOP指令和一层循环开销

 ● column为64且数据大小为4Byte时，一个VectorLength的寄存器正好可以保存，不需要计算Col维度上的偏移，且Reduce后的结果正好可以用一个寄存器承载

 ● 满轴且Shape一定为4的倍数，可以unroll为4路，充分利用寄存器并消除循环迭代依赖的情况下，还不用考虑尾块

 ● 由于提前知道Shape，地址计算在编译期完成可优化成VAG指令，减少开销

**泛化性评估**：**无法泛化**。无法满足所有Shape的最优性能，特化实现仅对此Shape有效。

#### **3.2.2 TColSum改写——高性能写法 vs 可融合写法**

TColSum的默认实现使用高性能二分写法，无法和其他element-wise的OP深融合。

**高性能二分写法**（不可融合）：

```auto
__VEC_SCOPE__
{
    for (uint16_t i = 0; i < rptTimes; ++i) {
        pReg = CreatePredicate<T>(sReg);
        srcP0 = src + 1 * SrcStride;
        srcP1 = src + 2 * SrcStride;
        vlds(dstVReg, src, elmPerRpt, NORM, POST_UPDATE);
        for (uint16_t j = 0; j < nLoop; ++j) {
            // 二分写法：同时load两个源，Reduce合并
            vlds(src0VReg, srcP0, 2 * SrcStride, NORM, POST_UPDATE);
            vlds(src1VReg, srcP1, 2 * SrcStride, NORM, POST_UPDATE);
            InstrOp::ReduceInstr(tmpVReg, src0VReg, src1VReg, pReg);
            InstrOp::ReduceInstr(dstVReg, dstVReg, tmpVReg, pReg);
        }
        vsts(dstVReg, dst, elmPerRpt, distValue, pReg, POST_UPDATE);
    }
} // end VF
```

**可融合写法**（性能更差但可融合）：

```auto
__VEC_SCOPE__
{
    for (uint16_t i = 0; i < rptTimes; ++i) {
        preg = CreatePredicate<T>(sReg);
        vbr((RegTensor<typename InstrOp::PadType> &)dstVReg, InstrOp::InitVal);
        for (uint16_t j = 0; j < (uint16_t)validRow; ++j) {
            vlds(srcVReg, src + i  elmPerRpt, j  SrcStride, NORM);
            InstrOp::ReduceInstr(dstVReg, dstVReg, srcVReg, preg);
        }
        vsts(dstVReg, dst + i * elmPerRpt, 0, distValue, preg);
    }
}
```

**泛化性评估**：***\*无法泛化\****。默认选择高精度的TColSum的Binary版本，此版本融合效果差，但在FA场景中可以选择TColSum非Binary版本仍然能保证精度正常。关键问题是：**在什么场景下可以安全使用低精度非Binary版本？这需要精度建模**。

#### **3.2.3 TCast改写——类型转换与全局寄存器的融合障碍**

TCast本身存在的不可融合问题：

  1. **全局寄存器的设置**：TCast会设置全局寄存器（如饱和模式等），融合后全局寄存器状态会影响其他OP

  2. **数据类型变化**：同样长度下元素数量变化（如float→half），导致element数量不一致，无法和后续依赖OP融合

本例子中由于TCast作为依赖链OP的末端，无需考虑②的问题。关于①的问题，由于FA-softmax中的全局寄存器采用默认即可，因此可以引入模板参数控制TCast在此场景下不设置全局寄存器，促进TCast与其他element-wise OP进行融合。

**泛化性评估**：**无法泛化**。由于涉及到元素位宽变化和全局寄存器的设置，TCast与后面的OP无法融合。

#### **3.2.4 TRowExpandBinary——循环结构不兼容导致深融合劣化**

PTO-isa中的RowExpandBin实现和其他element-wise的OP无法深融合，原因在于**vlds指令位于两层for循环之间**，融合后此vld和其他OP的访存指令间会插MEM_BAR，使性能大大劣化：

**不可融合形式**（vlds在外层循环，内层循环之前）：

```auto
__VEC_SCOPE__
{
    for (uint16_t i = 0; i < (uint16_t)(kValidRows); ++i) {
        uint32_t sreg = (uint32_t)(kValidCols);
        vlds(vreg1, src1Ptr, i * blockSizeElem, BLK);  // ← 外层循环内的vlds
        for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
            preg = CreatePredicate<T>(sreg);
            vlds(vreg0, src0Ptr, i  Src0RowStride + j  elementsPerRepeat, NORM);
            Op::RowExpandBinaryInstr(vreg2, vreg0, vreg1, preg);
            vsts(vreg2, dstPtr, i  DstRowStride + j  elementsPerRepeat, distValue, preg);
        }
    }
}
```

**可融合形式**（vlds移到最内层循环，但增加了冗余计算）：

```auto
__VEC_SCOPE__
{
    for (uint16_t i = 0; i < (uint16_t)(kValidRows); ++i) {
        uint32_t sreg = (uint32_t)(kValidCols);
        for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
            // vlds 移动到最内层循环可以与其他element-wise OP保持一致促进深融
            vlds(vreg1, src1Ptr, i * blockSizeElem, BLK);  // ← 冗余load！
            preg = CreatePredicate<T>(sreg);
            vlds(vreg0, src0Ptr, i * Src0RowStride + j * elementsPerRepeat, NORM);
            Op::RowExpandBinaryInstr(vreg2, vreg0, vreg1, preg);
            vsts(vreg2, dstPtr, i * DstRowStride + j * elementsPerRepeat, distValue, preg);
        }
    }
}
```

**核心矛盾**：可融合形式增加了repeatTimes倍的冗余vlds计算，性能一定是变差的。但不可融合形式的vlds位于外层循环内，融合后会在vlds和其他OP的访存指令之间插入MEM_BAR屏障，同样性能劣化。**两种方案都不是最优**。

#### **3.2.5 库实现优化泛化性总结**

| **优化点**  | **调整原因**          | **泛化性评估** | **核心矛盾**                                                 |
| ----------- | --------------------- | -------------- | ------------------------------------------------------------ |
| TColMax改写 | 单OP在此shape下非最优 | **无法泛化**   | Shape特化 vs 全Shape通用                                     |
| TColSum改写 | 使其参与深融合        | **无法泛化**   | 高精度Binary写法 vs 可融合非Binary写法，需要精度建模判断何时安全 |
| TCast改写   | 使其参与深融合        | **无法泛化**   | 全局寄存器/位宽变化 vs 融合友好性                            |

**PTO-ISA版本选择的困境**：每个PTO-ISA operator都需要在以下维度上做权衡，而这些维度之间往往是矛盾的：

通用性（支持所有Shape/尾块） ←→ 性能（特化实现/二分写法）
可融合性（统一循环结构）   ←→ 性能（最优循环结构）
高精度（Binary Reduce）     ←→ 可融合性（非Binary Reduce）
简洁性（无全局寄存器操作）  ←→ 功能完整性（全局寄存器控制饱和/溢出）

------

### **3.3 编译器调整优化点——融合判断的精度与性能权衡**

#### **3.3.1 支持ColReduce和ColExpand的OP融合**

**为什么ColReduce/ColExpand融合默认关闭？** 因为这两类OP的循环结构与element-wise OP不兼容：

 ● ColReduce的循环是**列优先遍历**，外层循环遍历列，内层做Reduce

 ● Element-wise的循环是**行优先遍历**，外层循环遍历行，内层处理列方向

 ● 融合后TripCount/Mask/Stride不一致，无法保证对同一段数据操作

**泛化性**：目前认为功能性是**可泛化**的，但泛化后的性能可能会受限。

#### **3.3.2 修复Expand指令UB overlap误判**

部分Expand指令被误判为UB overlap而无法融合。Expand操作将[1, 64]扩展为[128, 64]，需要读取源UB和写入目标UB，如果源和目标UB有overlap则无法融合。但某些场景下Expand的源数据在扩展过程中只读不写，实际上可以安全融合，编译器的Alias Analysis过于保守。

**泛化性**：修复Alias Analysis是**可泛化**的，但Alias Analysis的精度提升一直是编译器的经典难题。

**编译器优化泛化性总结**：

| **优化点**                  | **泛化性评估** | **难点**                                         |
| --------------------------- | -------------- | ------------------------------------------------ |
| ColReduce/ColExpand融合支持 | 部分泛化       | 需要解决循环结构不兼容问题，且默认关闭有精度考量 |
| 复合指令合并                | 可泛化         | 仅限于已知pattern，新pattern需人工添加           |
| Expand UB overlap误判修复   | 可泛化         | Alias Analysis精度提升是长期难题                 |
| Store-Load消除              | 可泛化         | 需要精确的UB生命周期分析                         |

------

## **四、VF融合难度深度剖析**

### 4.1 最优OP顺序的困难性

寻找最优OP顺序本质上是一个**排列优化问题**，其难点在于：

  1. **搜索空间巨大**：N个OP有N!种排列方式

  2. **约束复杂**：

 a. 数据依赖约束：OP必须在其输入数据的OP之后

 b. 同步语义约束：set-flag/wait-flag的位置影响调度正确性

 c. UB生命周期约束：融合不能改变UB的生命周期

  3. **目标函数**：OP顺序的微小变化可能导致融合机会的剧变（一个不可融合的OP插在中间就分割了整个融合群）

  4. **评估代价高**：每次调整OP顺序后，需要评估所有下游OP的融合可能性，这取决于PTO-ISA的实现细节



虽然最优OP顺序难以计算，但从实践中可以提炼出以下**可泛化的启发式规则**：

| **规则**                 | **原理**                                                     | **实现难度**           |
| ------------------------ | ------------------------------------------------------------ | ---------------------- |
| **相同Shape的OP聚合**    | 相同Shape的OP才可能融合，Shape突变是融合的硬分割线           | 中（需要Shape信息）    |
| **直接依赖的OP相邻**     | 上一个OP的输出是下一个OP的输入时，融合收益最大（消除UB读写） | 低（数据流分析）       |
| **Reduce OP靠后**        | Reduce改变Shape，放在融合群末尾避免分割                      | 中（需识别Reduce OP）  |
| **Expand OP靠前**        | Expand改变Shape，放在融合群开头可以将广播后的Shape用于后续融合 | 中（需识别Expand OP）  |
| **同步指令外提**         | set-flag/wait-flag外提到融合群边界                           | 高（需评估调度影响）   |
| **等价变换创造融合机会** | 如TMuls+ColMax ≡ ColMax+TMuls                                | 极高（需语义等价证明） |

**CodeGen阶段无法感知VF融合**是当前最大的结构性问题。如果CodeGen在生成OP序列时就能考虑VF融合友好性，则可以在源头上解决OP顺序问题，而不需要后端的复杂重排。

### **4.2 PTO-isa实现配合融合的难点**

即使假设OP顺序已经是最优的（即所有可融合的OP已经相邻排布），编译器做好融合仍然面临以下核心难点：

#### **难点1：不同OP在PTO-ISA实现的读写模式不统一**

**问题描述**：VF深融合要求参与融合的OP在循环结构上兼容——相同的TripCount、相同的Mask、相同的Stride。但不同OP的PTO-ISA实现天然有不同的循环结构：

| **OP类型**                      | **典型循环结构**     | **与ElemWise兼容性** |
| ------------------------------- | -------------------- | -------------------- |
| Element-wise (TADD, TMUL, TEXP) | 单层循环，按行处理   |                      |
| RowReduce (TROWSUM, TROWMAX)    | 单层循环，Reduce累加 | 输出Shape不同        |
| RowExpand (TEXPAND)             | 单层循环，广播源行   | 输入Shape不同        |
| ColReduce (TCOLSUM, TCOLMAX)    | 列优先遍历           | 循环结构根本不同     |
| RowExpandBin (TEXPANDADD)       | vlds在外层循环       | 循环结构不兼容       |
| TCast                           | 含全局寄存器设置     | 全局状态破坏         |

**具体不兼容案例**：

 ● **RowExpandBin**：vlds指令位于两层for循环之间（外层循环load广播源，内层循环计算），与element-wise OP的单层循环结构不兼容，融合后产生MEM_BAR

 ● **ColReduce**：循环是列优先遍历，element-wise是行优先遍历，即使Shape相同循环结构也不兼容

 ● **TCast**：除了循环结构外，还有全局寄存器设置的副作用，融合后会影响其他OP的全局寄存器状态

**编译器的处理**：当前编译器在PTO融合层面通过 `PTO_TYPE` 分类来简单判断是否可融合，但**不检查底层循环结构兼容性**。版本选择逻辑（`selectVersion()`）仅区分IMPL_1D和IMPL_2D，不区分具体的循环结构兼容性。

```auto
// PTOFusion.cpp selectVersion() 简化逻辑
if (all ops are ElemWise)
    select IMPL_1D;
else
    select IMPL_2D;
```

这意味着编译器在PTO层面认为可以融合的组合，可能在VF层面因为循环结构不兼容而无法深融合。

#### **难点2：与变ShapeOP融合时对实现极度敏感**

**问题描述**：Expand和Reduce OP是Shape变化的边界点，它们的融合处理比element-wise OP复杂得多：

**Reduce OP融合的难点**：

  1. Reduce将多维Shape压缩（如[128,64]→[1,64]），融合后同一个VF Loop内部分迭代写UB、部分迭代只累加不写，**必须精确判断哪次迭代是最后一次**才能正确输出

  2. Reduce的初始值设置（vbr指令）必须在循环之前，融合后需要保证vbr指令不被移到错误位置

  3. Reduce的非Binary版本精度较差，但Binary版本的循环结构不适合融合

**Expand OP融合的难点**：

  1. Expand将低维Shape广播到高维（如[1,64]→[128,64]），融合后源数据的load只发生一次，但需要参与多次计算，**循环结构天然不对称**

  2. Expand的源UB和目标UB可能有overlap，Alias分析容易误判

  3. RowExpandBin在循环结构上的vlds位置问题（如3.2.4节所述）

**Shape变化对融合分割的影响**：

[128,64] → ElemWise → ElemWise → Reduce → [1,64] → Expand → [128,64] → ElemWise
                          └─── 可融合 ───┘         └──── 可融合 ────┘
                                                    ↑
                                              Shape突变分割线

Reduce→Expand的Shape突变会强制分割为两个融合群。如果将Expand放到前面（如果逻辑允许），可以创造更大的融合群：

[128,64] → Expand → [128,64] → ElemWise → ElemWise → Reduce → [1,64]
           └──────────── 可融合 ────────────────────┘

这就是"ReduceOP尽量靠后，ExpandOP尽量靠前"的启发式原理。

### **4.3 编译器实现自动融合的难点**

#### **难点1：Alias分析的精度困境**

**问题描述**：VF融合需要判断两个OP之间是否存在数据依赖冲突（即UB空间的overlap），这依赖于Alias Analysis。但Alias分析一直是编译器领域的经典难题：

 ● **MustAlias**：确定有overlap → 不能融合（正确判断）

 ● **MustNoAlias**：确定无overlap → 可以融合（正确判断）

 ● **MayAlias/PartialAlias**：可能overlap → **保守处理为不能融合**（可能错失融合机会）

VF融合的Alias分析需要在编译期判断两个TileOP的UB buffer是否存在空间重叠，但PTO-ISA中TileOP的UB访问模式极其复杂，导致编译器在多种场景下无法做出精确判断。

**VF Loop Fusion中的精确依赖检查**（`HiIPUVFLoopFusion.cpp`）：

| **检查类型** | **说明**                                  | **保守程度** |
| ------------ | ----------------------------------------- | ------------ |
| W-W 别名     | FC0写 vs FC1写，PartialAlias/MayAlias拒绝 | 非常保守     |
| W-R RAW      | FC0写 vs FC1读，MustAlias时检查同P寄存器  | 中等保守     |
| R-W WAR      | FC0读 vs FC1写，MustAlias时检查有效依赖   | 中等保守     |

**结论**：PTO-ISA中TileOP的UB访问模式（POST_UPDATE步进、混合寻址模式、In-Place操作、Scatter写入、Size不匹配等）远超传统编译器Alias Analysis的处理能力。**当前编译器的Alias Analysis基于简单的UB首地址+大小区间判断**，无法处理上述任何一种复杂模式，只能在MayAlias时保守拒绝融合，错失大量合法融合机会。

#### 难点2：版本选择难度大

**版本选择的维度爆炸：**

每个PTO-ISA operator需要在多个维度上做选择，维度之间相互影响：

实现维度：
├── 循环结构：1D / 2D / PostUpdate变体
├── 精度模式：Binary / Non-Binary / 特殊算法(如ExpAlgorithm)
├── 地址模式：静态地址 / 动态地址 / BLK / NORM
├── 尾块处理：有尾块 / 无尾块 / Unroll
├── 合轴策略：合轴 / 不合轴
├── 全局寄存器：设置 / 不设置
└── 融合友好：可融合 / 不可融合

以TExp为例，仅算法选择就有多种：

TExp<pto::ExpAlgorithm::DEFAULT, ...>    // 默认算法
TExp<pto::ExpAlgorithm::HIGH_PRECISION, ...>  // 高精度算法

以TCast为例：

TCast<..., pto::SaturationMode::OFF>    // 不饱和
TCast<..., pto::SaturationMode::ON>     // 饱和截断

每个维度的选择都影响融合友好性和性能，维度组合是**指数级**的。

**版本选择与融合耦合**

核心原因：PTO-ISA的版本选择在**内联之前**（CodeGen阶段），而融合判断在**内联之后**（编译器后端）。这意味着：

  1. CodeGen选择了一个"高性能"版本（如Binary ColSum）

  2. 编译器后端发现这个版本不适合融合

  3. 但已经无法回退到"可融合"版本

当前编译器的处理：PTO融合层面的 `selectVersion()` 在**融合之后**修改版本选择：但这个修改只区分1D/2D，不区分更深层次的实现选择（如Binary vs Non-Binary）。

而且版本选择中还涉及到精度考量：

以TColSum为例：

| **版本**               | **精度**     | **融合友好性** | **性能** | **适用场景**        |
| ---------------------- | ------------ | -------------- | -------- | ------------------- |
| Binary（二分写法）     | 高精度       | 不可融合       | 高性能   | 大Shape、精度敏感   |
| Non-Binary（逐行写法） | 可能精度不足 | 可融合         | 性能较差 | 小Shape、精度不敏感 |

编译器如何判断"Non-Binary版本在当前场景下精度是否足够"？当前的编译器**完全没有**这种精度建模能力，因此只能保守选择高精度版本，牺牲融合机会。

#### **难点3：版本选择的CostModel缺失**

一个完整的PTO-ISA版本选择CostModel需要回答以下问题：

| **问题**                 | **需要的信息**                                               | **当前状态**                 |
| ------------------------ | ------------------------------------------------------------ | ---------------------------- |
| 是否应该融合？           | 融合收益（VF启动节省+UB读写减少）vs 融合代价（MEM_BAR+寄存器压力） | 仅有粗粒度参数               |
| 融合后选什么版本？       | 1D vs 2D的可融合性差异、性能差异                             | 仅区分ElemWise vs 非ElemWise |
| 是否值得为融合切换版本？ | "可融合版+融合收益" vs "高性能版+不融合代价"                 | 无此CostModel                |
| 精度是否可接受？         | 输入数据范围、误差传播链、目标精度                           | 无精度建模                   |

目前编译器假设融合起来**一定有收益**，但实际情况远比这复杂：

**融合不一定有收益的场景**：

  1. **融合个数过多**：融合太多OP会导致寄存器压力增大，可能触发register spill，反而劣化性能

 a. VF Loop Fusion中有寄存器压力检查：VMemRead数量之和超过`AvailableVregCount`(默认32)拒绝融合

 b. 不同P寄存器指令数超过`AvailablePregCnt`(默认7)拒绝融合

 c. 但这只是粗粒度检查，没有精确的CostModel

  2. **合轴版本(1D) vs 2D版本的选择**：

 a. 1D版本（IMPL_1D）：所有OP都是ElemWise时选择，循环结构统一，但可能无法充分利用2D数据的局部性

 b. 2D版本（IMPL_2D）：包含Reduce/Expand时选择，保持2D循环结构，但循环结构不兼容的风险更大

 c. 当前选择逻辑过于简单：只看是否全部是ElemWise

  3. **融合vs不融合的量化评估**：

 a. 节省1次VF启动 ≈ 140 cycles（`BenefitOfSavingVFLaunch = 140`）

 b. 但融合后如果插入MEM_BAR或register spill，损失可能远超140 cycles

 c. 当前代价模型参数过于粗糙：

CostOfTermOpsInVF = 15;
CostOfMemoryOpsInVF = 15;
BenefitOfSavingVFLaunch = 140;

  4. **深融合vs浅融合的收益差异**：

 a. 浅融合：只合并VF启动，OP间仍通过UB传递

 b. 深融合：OP间通过寄存器传递，消除UB读写

 c. 深融合收益远大于浅融合，但深融合对循环结构兼容性要求更高

 d. 当前编译器无法量化深融合vs浅融合的收益差异

------

## **五、VF Loop Fusion与LoadStoreElimination的限制深度分析**

### **5.1 VF Loop Fusion的限制**

VF Loop Fusion的融合判断是一个**严格的瀑布式检查链**，按顺序执行，以下任一环节失败即终止：

#### **5.1.1 融合准入硬性约束**

**1. Trip Count必须完全一致**（`HiIPUVFLoopFusion.cpp:1138-1160`）

使用SCEV比较两个loop的backedge taken count。如果SCEV无法计算（`SCEVCouldNotCompute`），直接拒绝。这意味着**动态trip count的loop无法融合**——而很多PTO-ISA的loop是运行时才知道迭代次数的（如尾块处理）。

**2. 必须相邻**（`HiIPUVFLoopFusion.cpp:1171-1177`）

FC0的ExitBlock必须是FC1的Preheader（非guarded loop），或FC0的NonLoopBlock必须是FC1的EntryBlock（guarded loop）。**中间有任何BB都阻止融合**——即使BB只有少量标量指令也不行（注释中提到未来可能处理non-empty exit blocks，但当前未实现）。

**3. Guard必须一致**（`HiIPUVFLoopFusion.cpp:1191-1209`）

两个loop要么都有guard，要么都没有。都有guard时，guard的compare指令必须identical。**guard条件略有不同的两个loop无法融合**。

#### **5.1.2 依赖检查——最核心也最保守的环节**

`dependencesAllowFusion`（`HiIPUVFLoopFusion.cpp:1594-1712`）是融合判断的核心，也是**最保守的环节**：

**W-W检查**（两个loop都有写，line 1602-1618）：

FC0.VMemWrites × FC1.VMemWrites
→ PartialAlias/MayAlias → 拒绝（OverlappingAddressAccess）

实际上很多场景下两个loop写同一块UB的不同部分是安全的（比如写不同的行），但MayAlias直接拒绝。

**W-R RAW检查**（FC0写，FC1读，line 1620-1647）：

FC0.VMemWrites × FC1.VMemReads
→ MayAlias/PartialAlias → 拒绝
→ MustAlias + 不同Preg → 拒绝（RawDifferentPregHazard）
→ MustAlias + 相同Preg → 允许

**RawDifferentPregHazard是精确性问题的根源**：store有predicate控制有效写入范围，load没有predicate参数（A5硬件限制——vlds无predicate），如果store和load的predicate不同，load会读到store写的padding数据。但当前检查**只在MustAlias时检查Preg**，实际上即使MustAlias + 相同Preg也不一定安全（因为vlds无predicate，总是读整个向量宽度的数据，包括无效padding元素）。

**R-W WAR检查**（FC0读，FC1写，line 1650-1695）：

FC1.VMemWrites × FC0.VMemReads
→ MayAlias/PartialAlias → 拒绝
→ MustAlias → 调用hasValidDependency()检查是否存在有效的读依赖链

`hasValidDependency`（line 1717-1741）检查"FC0的Read是否有对应的Write，且FC1的Write也有对应的Read，且这些Read/Write之间MustAlias"。这是试图识别"FC0写了中间结果到UB，FC1读这个中间结果"的场景——但这种检查方式过于保守，很多合法的依赖链无法通过。

**控制流依赖**（line 1702-1709）：

```cpp
for each instruction in FC1:
  for each operand:
    if def is in FC0 → 拒绝
```



#### **5.1.3 Access Pattern兼容性检查——VAG地址生成的物理约束**

`hasCompatibleAccessPattern`（`HiIPUVFLoopFusion.cpp:1466-1591`）比较两个loop中MustAlias访存指令的**SCEV AddRec步长**：

```cpp
// 对每对MustAlias的访存指令，提取VAG offset的SCEV AddRec表达式
// 比较所有嵌套层的step recurrence
for (size_t L = 0; L < Steps0.size(); ++L) {
    if (Val0 != Val1) → 拒绝（IncompatibleAccessPattern）
}
```



这是**物理约束而非逻辑约束**：融合后两个loop的VAG地址生成必须共用相同的地址增量模式。如果两个loop访问同一块UB但stride不同（比如一个按行步进64个f32，一个按列步进128个f32），融合后VAG无法同时生成两种stride。

**对融合的具体影响**：

 ● Reduce + Expand无法通过此检查：Reduce按行步进写结果到[1,64]，Expand按行步进从[1,64]读取并广播——虽然读写的UB相同，但Reduce的写步进与Expand的读步进不同（Reduce写完一行就结束，Expand读一行重复多次）

 ● 两个不同Shape的element-wise OP如果使用不同的RowStride，也无法通过此检查。



### **5.2 VF LoadStoreElimination的限制**

VF LoadStoreElimination的目的是**在VF loop内部消除冗余的vload/vstore对**，这是VF深融合的前提——如果store-load对能被消除，数据就可以通过寄存器传递而非UB，实现真正的深融合。



#### **5.2.3 Store-Load消除的严格条件**

`isStoreLoadPair`（`HiIPUVFLoadStoreElim.cpp:288-315`）需要**三个条件同时满足**：

**条件1：MustAlias**

调用`checkOverlappingAddressAccess`，MayAlias/PartialAlias → 不构成pair。和Loop Fusion一样保守。

**条件2：Dist模式必须匹配**（仅三种组合）

| **组合**  | **Store Dist**        | **Load Dist**              | **典型场景**  |
| --------- | --------------------- | -------------------------- | ------------- |
| NORM      | NORMAL_B8/B16/B32     | NORM                       | 普通连续读写  |
| ONEPT+BRC | ONEPT_B8/B16/B32      | BRC_B8/B16/B32             | 广播写→广播读 |
| PK+UNPK   | PK_B32/PK_B64/PK4_B32 | UNPK_B16/UNPK_B32/UNPK4_B8 | 压缩写→解压读 |

**不在以上三种组合中的dist模式对，消除直接失败**：

 ● DINTLV模式（解交织store + 交织load）不在列表中 → **vstsx2和vldsx2之间的消除不支持**

 ● E2B模式不在列表中

 ● 任何非标准的dist组合都不支持

这意味着使用了`vldsx2`/`vstsx2`（dintlv模式）的OP之间，即使store和load地址完全相同，也无法消除store-load对。

**条件3：Preg必须匹配**（`isStoreLoadSamePreg`，line 263-286）

store的predicate寄存器必须与load后续计算使用的predicate寄存器一致。有一个例外：如果load有`pto.last_use` metadata，跳过Preg检查。

**pto.last_use是一个不安全的hack**：当PTO fusion标记了`MD_pto_last_use`时，意味着此load的数据在整个tile生命周期内是最后一次使用，不需要考虑Preg一致性。但PTO fusion只保证了tile级别的正确性，**不保证predicate级别的正确性**——如果store的Preg使部分元素无效，load后续计算可能使用到无效元素的值。

#### **5.2.4 中间指令的干扰——标量UB操作阻断消除**

`collectStorePairs`/`collectLoadPairs`在扫描时遇到中间指令的处理：

| **中间指令类型**                    | **处理方式**                     | **影响**                           |
| ----------------------------------- | -------------------------------- | ---------------------------------- |
| 非候选的向量store（非vstx1/vstsx1） | **立即终止扫描**（return false） | 任何非标准格式的向量store都会阻断  |
| MayAlias/PartialAlias的store        | **终止扫描**                     | 和Loop Fusion一样保守              |
| MustAlias的store                    | 加入InterST集合，需要先消除      | 可以后续处理，但增加了复杂度       |
| **标量UB store/load**               | **立即终止扫描**                 | **最严重**：任何标量UB操作都会阻断 |

**标量UB操作阻断是最常见的问题**：在PTO-ISA中，很多OP会在vector计算之间插入标量UB操作（如TCast设置全局寄存器、计算下一行的偏移量等），这些都会阻断LoadStoreElimination。这也是§3.2.3中"TCast全局寄存器设置阻碍融合"在编译器层面的具体体现。

#### **5.2.5 Store-Store和Load-Load消除的限制**

**Store-Store**（`isStorePair`，line 317-348）：

 ● MustAlias + 相同Preg + NORM模式 → 消除第一个store

 ● ONEPT + NORM + Preg来自PLT或相同 → 消除第一个store

 ● **只支持NORM-NORM和ONEPT-NORM两种组合**

**Load-Load**（`isLoadPair`，line 350-374）：

 ● MustAlias + 相同类型 + 都是NORM模式 → 消除第二个load

 ● **只支持NORM-NORM**

其他dist模式（DINTLV、BRC、UNPK等）的store-store和load-load都不支持消除。

#### 

### **5.3 Loop Fusion + LoadStoreElimination的协同问题**

#### **5.3.1 执行顺序导致的消除遗漏**

LoadStoreElimination在VF Fusion之前运行，但只能消除同一BB内的store-load对。融合前两个loop在不同BB，LoadStoreElim无法消除。融合后两个loop在同一BB，但LoadStoreElim不会再运行。

**实际的消除机制**：VF Fusion在融合时通过`FusionAdjacentCode`类处理相邻BB的指令移动（hoist/sink/blend），但**不做store-load消除**。融合后新暴露的store-load对可能永远无法被消除。

#### **5.3.2 三层保守叠加的恶性循环**

Loop Fusion和LoadStoreElim的保守性存在**恶性叠加**：

LoadStoreElim无法消除跨BB的store-load对
→ Fusion看到两个loop之间有store-load依赖
→ dependencesAllowFusion判断RAW MustAlias + 不同Preg
→ 拒绝融合
→ LoadStoreElim仍然无法消除（因为没融合，还是跨BB）

如果能先融合再消除，很多场景是安全的——融合后store-load在同一BB内，可以通过寄存器传递数据。但当前的Pass执行顺序不允许这种"先融合再消除"的模式。

#### **5.3.3 限制汇总**

| **限制类别**                      | **具体限制**                | **保守程度** | **对融合的影响**                     |
| --------------------------------- | --------------------------- | ------------ | ------------------------------------ |
| **Loop Fusion: MayAlias拒绝**     | W-W/W-R/R-W的MayAlias都拒绝 | 非常保守     | 错失大量合法融合机会                 |
| **Loop Fusion: RawDifferentPreg** | MustAlias + 不同Preg拒绝    | 合理但保守   | vlds无predicate的硬件限制            |
| **Loop Fusion: Access Pattern**   | 不同stride拒绝融合          | 物理约束     | 无法绕过（除非改变VAG使用方式）      |
| **Loop Fusion: 控制流依赖**       | FC1使用FC0的def即拒绝       | 过于保守     | loop-invariant值也被拒绝             |
| **LoadStoreElim: Dist模式限制**   | 只支持3种组合               | 严重         | DINTLV/E2B等模式的store-load无法消除 |
| **LoadStoreElim: 标量UB阻断**     | 标量UB操作终止扫描          | 严重         | TCast等OP的标量副作用阻断消除        |
| **LoadStoreElim: Preg匹配**       | store和load的Preg必须一致   | 合理但保守   | 不同Preg的场景无法消除               |

------

## **六、PyPTO2.0框架下VF融合与手写融合的差距**

### **5.1 根本矛盾**

**自动融合要满足通用性，而手写不需要**。这是不可调和的矛盾——手写融合可以在特定场景下突破通用性约束，获得极致性能，但这些优化无法泛化。

### **5.2 差距的具体体现**

#### **CodeGen层面**

| **自动融合**                                      | **手写融合**                   | **差距**                       |
| ------------------------------------------------- | ------------------------------ | ------------------------------ |
| CodeGen阶段无法感知VF融合，OP顺序不考虑融合友好性 | 手写直接按融合最优顺序编写     | 需要后端重排OP，但重排能力有限 |
| 无法对OP做等价变换以创造融合机会                  | 手写可以根据语义等价性自由变换 | 需要形式化验证语义等价         |

#### **PTO-ISA层面**

| **自动融合**                         | **手写融合**             | **差距**             |
| ------------------------------------ | ------------------------ | -------------------- |
| 无法对所有Shape提供特化实现          | 手写针对当前Shape特化    | 性能差距可达数倍     |
| 无法保证一定有适合当前场景融合的实现 | 手写直接写融合友好的实现 | 循环结构兼容性无保障 |

#### **编译器层面**

| **自动融合**                         | **手写融合**                 | **差距**               |
| ------------------------------------ | ---------------------------- | ---------------------- |
| 缺少CostModel，无法判断是否应该融合  | 手写基于经验直接决策         | 可能做出错误的融合决策 |
| 无法在融合后重新选择OP实现版本       | 手写直接选择最优实现         | 版本选择与融合脱节     |
| Alias Analysis精度不足               | 手写知道确切的内存布局       | 保守判断错失融合机会   |
| 精度建模缺失，只能保守选择高精度版本 | 手写知道当前场景的精度容忍度 | 牺牲融合换精度         |

### **5.3 即使解决所有问题，手写仍有的优势**

  1. **场景特权的深融合**：某些深融合在特定场景下是正确的，但从通用性来讲是不被允许的。自动融合无法做到"仅在此场景下允许此类融合"。

  2. **寄存器复用**：手写融合可以将需要复用的数据存放在寄存器里，使用时直接参与计算，减少UB-load。而自动融合基于PTO-ISA，只能按部就班从UB里面load数据。

  3. **全局优化视野**：手写可以整体考虑整个算子的数据流，做全局最优决策；自动融合是局部贪心的——按顺序逐个判断，无法回溯。