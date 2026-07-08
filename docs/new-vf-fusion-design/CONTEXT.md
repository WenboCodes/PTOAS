# New VF Fusion Design

Reconstructing the VF (vector-function) fusion pass on top of `pto.vmi`, a logical vector ISA layer between TileLang and the physical `pto.mi` ISA. Motivated by the structural limits of PyPTO2's post-hoc loop-fusion approach (see `PyPTO2-vf-fusion-analysis.md`).

## Language

**VF fusion**:
Merging multiple vector ops into one VF (vector-function) loop so data flows through registers instead of UB round-trips. The pass being reconstructed.
_Upstream_: TileOP templates (a `pto.vmi` op template library, NOT a dialect); each tileop expands to a self-contained `scf.for` over `N×VL` plus vmi ops. TileLang is only one possible producer of vmi IR (illustrated in `PTO-vmi-design.md` §12) and is not the design target of the fusion pass.
_Form_: post-hoc fusion of independent tileop loops into one shared loop — structurally the same shape as PyPTO2's `HiIPUVFLoopFusion`, but the eligibility judgment moves from the `pto.mi` physical layer (SCEV trip-count, MayAlias, loop-structure) up to the `pto.vmi` logical layer (tile-shape constraint, mask as first-class value, op Category).
_Partial fusion_: NOT all-or-nothing. A subset of compatible tileop loops may fuse while others stay independent; each independent group is lowered separately.
_Avoid_: loop fusion (too generic), vectorization, TileLang fusion (wrong upstream)

**VF deep fusion**:
Fusion where ops pass data through registers, eliminating UB load/store. Contrast with shallow fusion (only VF-launch amortization; data still via UB).
_Avoid_: (none)

**pto.vmi**:
A logical vector ISA layer between the TileLang programming model and the physical `pto.mi` ISA. Exposes only logical-contiguous semantics; physical SIMD register layout is owned by the compiler and invisible to the user.
_Avoid_: vmi (ambiguous), logical vector layer

**pto.mi**:
The physical micro-instruction ISA (CCE) below `pto.vmi`. Carries explicit layout (interleave / half / part / pack / dist).
_Avoid_: CCE ISA, MI

**pto.as**:
The compiler stage that owns layout assignment and lowering from `pto.vmi` to `pto.mi`.
_Avoid_: (none — proper name)

**TileOP**:
A **template library built on `pto.vmi` ops** (NOT a separate dialect). The user instantiates templates to author independent tileops; each tileop expands to a self-contained `scf.for` over an `N×VL` tile plus vmi ops. The VF-fusion pass takes this expanded vmi IR as input and recognizes independent tileop loops as IR patterns — it does not consume a dedicated TileOP op. Distinct from TileLang (a higher-level producer that may lower to vmi but is not the fusion pass's design target).
_Avoid_: tile op (ambiguous — could mean a single op), TileLang (different layer), TileOP dialect (there is none)

**shaped ptr**:
`!pto.ptr<T, ub, shape=[...], stride=[...]>` — a `pto.ptr` extended to carry the UB memory space's logical geometry (shape and stride, memref-style). `vload`/`vstore` take a multidim index expression (`%a[%i, %j]`) against a shaped ptr; the index references only induction variables, strides come from the ptr type. Distinct from physical layout (parity/half/part/pack), which is compiler-internal and owned by `pto.as`.
_Avoid_: memref (different dialect, same idea), layout (different axis)

**access pattern**:
The geometry of a `vload`/`vstore`'s UB access, expressed as induction variables + static strides (from the shaped ptr). VF-fusion's access-pattern-compatibility check compares two tileops' access patterns to decide fusion (replacing PyPTO2's LLVM-layer SCEV-AddRec stride comparison, §5.1.3). Operates on induction variables + strides ONLY; the dynamic `valid row` (Form B trip count) does NOT enter it — that governs trip-count only. The two are orthogonal. The same region-precision upgrades PyPTO2 §5.1.2's alias checks: W-W and WAR are fully resolved (distinct regions fuse even under ptr MayAlias); RAW is region-precise but still gated by A5's non-predicateable `vload` (consumer `[pmode]` must mask the stored padding).
_Avoid_: dist-mode (that is access *shape*: continuous/unpack/dintlv/brc; orthogonal to the index geometry), physical VAG stride

**mem2reg**:
A vmi pass running **after fusion** that promotes `vstore`→`vload` pairs at the same shaped-ptr index location to direct vreg SSA def-use, eliminating the UB round-trip (deep fusion). Replaces PyPTO2's `HiIPUVFLoadStoreElim` (§5.2) and fixes its ordering bug (LoadStoreElim ran pre-fusion, so post-fusion-exposed pairs were never eliminated). Resolves LoadStoreElim's three refusals: keys on index location not dist-mode (so DINTLV pairs eliminate), ignores scalar UB, and carries masks as first-class values (no Preg-match gate).
_Scope_: tileop-to-tileop UB store-load ONLY. Relationship to a reduce acc depends on orientation (ADR-0001): RowMax-style acc is in-loop `iter_args`, never UB, NOT promoted (orthogonal); ColMax-style acc crosses loops via whole-vreg store→load, IS promoted. The in-loop acc update (per-iteration `vcmax`/`vmax` into `iter_args` for RowMax) is register SSA, never touched.
_Capability boundary_: promotes **same-location same-shape** store→load, independent of reduce orientation. ColMax-style (acc direct in-loop) and RowMax-style (loop-1 `vstore %acc` → loop-2 outer `vload`, whole vreg both sides) both promote. How loop-2 internally consumes the promoted value (row-indexed extract etc.) is post-promotion SSA, not UB traffic.
_Avoid_: LoadStoreElim (the PyPTO2 equivalent it replaces), store-load elimination (generic)

**Category A / B / C**:
A lowering contract classifying a vmi op by its relationship to register layout: A = layout-passthrough; B = layout-rewritable along a matched axis; C = contiguous-required (forces layout materialization before the op).
_Avoid_: (none — canonical taxonomy)

**dist-mode**:
A `vload`/`vstore` attribute declaring the access shape (`continuous` | `unpack` | `dintlv` | `brc`). Orthogonal to layout inference (register-side placement), which `pto.as` decides separately.
_Avoid_: dist (overloaded — also the physical `pto.mi` token)

**VL**:
The physical 256B vreg capacity in elements, **dtype-dependent**: `VL = 256B / bitwidth(T)` (f32/i32 → 64, f16/bf16/i16 → 128, fp8/i8 → 256). NOT a free choice — determined by dtype. Bounds the surviving axis of a reduce to one physical register. (Current scope; a future extension using group reduce may allow `VL ∈ {64,128,256}` as a free choice, decoupling VL from dtype — see ADR-0001 Future Work.)
_Avoid_: dtype-independent logical element count (that was an earlier, rejected framing), physical vreg lane count (same thing, but "VL" is the contract term)

**tile-shape constraint**:
A tile's logical shape is `N × VL`. The **surviving axis of any reduce** (the axis the acc lives on) MUST equal VL, so the acc is exactly one `V<VL×T>` = one physical 256B vreg. The reduced axis may be either `N` (ColMax-style) or `VL` (RowMax-style, requires `N=VL`). "More elements" is carried by tiling into more blocks, never by stacking the surviving axis as `k·VL`. Note: surviving-axis=VL is necessary but NOT sufficient for same-loop fusion — see `acc 生命周期` (ColMax-style splits into two loops; RowMax-style fuses).
_Structural consequence_: the inner axis is VL = a single vreg (not a loop axis), so a tileop is a **single-loop** over `N` only; no 2-layer nested loop, no "合轴/非合轴" choice.
_Enforced upstream at tiling_: a tile's row or column must fit in one vreg, and tiling must not produce a compact tile that would require `{group=C}` lane-group reduce. The fusion pass never rejects a non-conforming tile because tiling cannot emit one.
_Avoid_: (none — canonical contract)

**acc 生命周期 (acc lifetime)**:
The second gate for same-loop reduce ⊕ element-wise fusion (after surviving-axis=VL). Same-loop fusion is possible only when the acc is **produced per-iteration and consumable in the same iteration** by element-wise. RowMax-style (per-row max, produced each iteration) → fuses in one loop. ColMax-style (VL column-maxes, each needs all `N` rows) → acc accumulates across all iterations, complete only after the loop → splits into two loops. This is NOT about traversal direction — both styles can traverse `N`, yet ColMax still can't fuse because its acc isn't complete mid-loop. (The earlier "traversal-direction consistency" framing was wrong and is superseded by this.)
_Avoid_: traversal-direction consistency (superseded — wrong framing), lane alignment (even earlier, also wrong)

**N (tile buffer size)**:
The compile-time-constant **buffer capacity** of a tile `N × VL` — the maximum number of `V<VL×T>` rows the tile can hold. `N` is an upper bound on `valid row`, NOT necessarily the trip count. In Form A the loop runs to `N` (static trip); in Form B it runs to `valid row` (dynamic trip, ≤ `N`). Buffer allocation is static because `N` is static.
_Avoid_: trip count (may be dynamic — see `valid row`), row count (ambiguous)

**valid data**:
The run-time-known count of elements that are actually meaningful in a tile, as opposed to the static buffer capacity `N`. Dynamicity splits into **two layers**:
- **row level**: how many rows are valid (`valid row`, ≤ `N`). Form A carries this via a mask that masks out overrun iterations; Form B makes the trip count dynamic (`to %valid_row`), so no overrun iterations run.
- **lane level**: the last row's tail (a partial VL). Carried by a `pto.vmi.mask` from `plt`, consumed by in-loop ops via `[pmode]`. Form A derives it from the `iter_args` counter `remaining`; Form B derives it from `valid row - %offset`.
This is why a fused loop can have a static N yet carry dynamic tail semantics, or a dynamic trip count with a dynamic lane tail.
_Avoid_: data length, element count (ambiguous)

**plt**:
`pto.vmi.plt %rem : i32 -> !pto.vmi.mask<L>, i32` (§10). General-form mask generator taking a run-time scalar `%rem`; yields a mask with the first `min(rem, VL)` lanes active. A **uniform expression**: degenerates to all-active when `rem ≥ VL`, to a short tail mask when `rem < VL`. No branching; tail-ness emerges from the monotonic decrease of the `iter_args` counter `remaining`. `pset "PAT_ALL"` and `pge "PAT_VLn"` are shorthand special cases absorbed by `plt`.
_Avoid_: tail mask (that is one degeneration of it), predicate generation, create_mask (not a real op name)

**valid row**:
The run-time-known count of valid rows in a tile (`≤ N`). In Form B this is the `scf.for` upper bound (dynamic trip count); in Form A it is materialized as an `iter_args` counter that the mask uses to mask out iterations beyond it. Distinct from `N` (the static buffer capacity) and from the lane-level tail.
_Avoid_: N, trip count (overloaded)

**iter_args counter**:
An `scf.for` `iter_args` value (e.g. `%remaining` in Form A) that carries the run-time valid-data count, initialized before the loop and decreased by `VL` each iteration (`%next = arith.subi %remaining, %cVL`). Used by Form A (which has no dynamic upper bound to derive the tail from). A reduce accumulator threads through `iter_args` the same way, regardless of form.
_Avoid_: loop counter (that is `%i` / `%offset`), induction variable

**寄存器直传 (register direct-pass, Class 2)**:
The property that lets reduce/broadcast share a vreg with element-wise ops **inside the same `N × VL` loop**, eliminating the UB round-trip. For reduce: the loop loads one `V<VL×T>` per iteration and the reduce produces a per-iteration acc **consumable in the same iteration** by element-wise. For broadcast: the broadcast source is one `V<VL×T>` vreg referenced directly as an operand by the element-wise inner work. The acc-vreg identity is static because the acc is an `scf.for` `iter_args` SSA value reused across iterations (independent of whether the trip count is static — see ADR-0002); the **valid-data** boundary is run-time and carried by a `mask` consumed via `[pmode]`.
_Two reduce orientations (acc lifetime decides fusion)_: RowMax-style (reduce along inner VL, survive on outer, requires `N=VL`) — acc = per-row max, **produced per-iteration, consumable in-loop** → **fuses in one loop**; reduce uses `vcmax` (whole-row reduce, **expensive**). ColMax-style (reduce along `N`, survive on inner=VL) — acc = "VL column-maxes", each needs all `N` rows → **acc accumulates across all iterations, complete only after the loop** → **splits into two loops**, acc crosses via UB (promoted by mem2reg); reduce uses `vmax` (element-wise, **cheap**). Both have acc = one `V<VL×T>` (surviving axis = VL); the difference is acc lifetime (see `acc 生命周期`), NOT traversal direction (the earlier "traversal-direction consistency" framing was wrong — both can traverse `N`).
_Cost-model trade-off_: ColMax (no fuse + cheap `vmax`) vs RowMax (fuse + expensive `vcmax`) is a cost-model question, not "fuse whenever possible" (open).
_Structural consequence_: because the inner axis is VL = a single vreg (not a loop axis), a tileop is a **single-loop** (`scf.for` over `N` only) — there is no 2-layer nested loop. "合轴/非合轴" (axis-merge vs not) does not exist in VMI; PyPTO2's 2D nested-loop form is gone.
_Scope boundary_: only reduces whose surviving axis is VL (acc = `V<VL×T>`) are in scope. The **compact lane-group reduce `{group=C}`** (`vcadd`/`vcmax`/`vcmin` with `{group=C}`, produces `V<C×T>`, C∈{1,2,4,8}) is **out of scope** for same-loop element-wise fusion — its acc shape `V<C×T>` does not align with an element-wise `V<VL×T>` operand. It still exists as a Category-B op (§2.2) and connects to element-wise only via `vbrc {group=C}`. Tiling is forbidden from producing a compact tile that would require it.
_NOTE on `vcmax` naming_: `vcmax` is overloaded. (a) `vcmax {group=C}` — compact lane-group reduce, out of scope (above). (b) `vcmax` whole-row reduce (no group attr) — reduces one `V<VL×T>` row to a scalar, used by **RowMax-style** (in scope, fuses in-loop, expensive). These are different ops sharing a name.
_Avoid_: register residency (the earlier name), deep fusion (that is the effect, not the property)
