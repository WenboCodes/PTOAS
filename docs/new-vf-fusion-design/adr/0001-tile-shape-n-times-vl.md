# Tile shape `N × VL`, with the surviving axis pinned to VL

## Context

PyPTO2's VF fusion fails to fuse reduce/broadcast tile ops with element-wise tile ops (see `../PyPTO2-vf-fusion-analysis.md` §3.3.1, §4.2). The root cause is register-residency: a reduce over an arbitrary axis produces a result that cannot sit in a single static register across the element-wise loop, so the compiler falls back to a UB round-trip and the two ops' loop structures become incompatible.

## Decision

Constrain every tile's logical shape to `N × VL`, where:

- **`VL` is the physical 256B vreg capacity in elements, dtype-dependent**: `VL = 256B / bitwidth(T)` (f32/i32 → 64, f16/bf16/i16 → 128, fp8/i8 → 256). VL is NOT a free choice — it is determined by dtype. (Current scope; see Future Work below.)
- **the surviving axis of any reduce must be `VL`** — i.e. the axis the reduce leaves behind (the acc axis) is `VL`, so the acc is exactly one `V<VL×T>` vreg = one physical 256B vreg. The reduced axis may be either `N` (outer) or `VL` (inner), as long as what survives is `VL`:
  - **ColMax-style** (reduce along `N`, surviving axis = inner = `VL`): tile `N × VL`, `vmax`/`vadd` accumulates into acc `V<VL×T>` across `N` iterations. acc = "VL column-maxes"; each column-max needs all `N` rows → **acc accumulates across all iterations, complete only after the loop** → **cannot fuse in-loop** (element-wise must wait for acc). Reduce uses `vmax` (element-wise, **cheap**). Splits into two loops; acc crosses via UB (promoted by mem2reg).
  - **RowMax-style** (reduce along inner `VL`, surviving axis = outer = `VL`): requires `N = VL`, so the tile is `VL × VL`; reduce along inner, acc `V<VL×T>` survives on the outer axis. acc = "per-row max"; each row's max is produced **within that iteration** and consumed by that row's element-wise `x - max` **in the same iteration** → **acc produced per-iteration, consumable in-loop** → **fuses in one loop**. Reduce uses `vcmax` (whole-row reduce, **expensive**).
- **same-loop fusion depends on acc lifetime, NOT traversal direction.** "Surviving axis = VL" makes the acc a single vreg (necessary), but is NOT sufficient: same-loop fusion needs the acc **produced per-iteration and consumable in the same iteration** (RowMax-style). ColMax-style's acc accumulates across all iterations and completes only after the loop, so it splits into two loops. (The earlier "traversal-direction consistency" framing was wrong — both styles can traverse `N`, yet ColMax still can't fuse because its acc isn't complete mid-loop.)
- **ColMax vs RowMax is a cost-model trade-off**, not "fuse whenever possible": ColMax doesn't fuse but uses cheap `vmax`; RowMax fuses but uses expensive `vcmax`. Choosing is a cost-model question (open).
- "More elements" is carried by tiling into more `N × VL` (or `VL × VL`) blocks, **never** by stacking the surviving axis as `k·VL`.
- reduce and element-wise run **inside the same `N × VL` loop** (RowMax-style only): each iteration loads one `V<VL×T>`, reduces out the row's max, and the element-wise op consumes it in the same iteration — no `N × VL` UB intermediate is materialized.
- this is enforced **upstream at tiling**: a tile's row or column MUST fit in one vreg, and tiling MUST NOT produce a compact tile that would require `{group=C}` lane-group reduce. The fusion pass never rejects a non-conforming tile because tiling cannot emit one.

## Why

With the surviving axis pinned to VL (= one physical 256B vreg), a reduce's acc collapses to a single vmi.vreg = one physical register. For RowMax-style (the fusing case), the acc is produced per-iteration and consumed by element-wise in the same iteration, so no `N × VL` UB intermediate is materialized and the fusion no longer depends on matching loop trip-counts / masks / strides across two incompatible loop structures. This is the only lever that makes Class 2 (register direct-pass) problems solvable; abstraction richness alone (Class 1) does not address them.

Pinning VL to physical vreg capacity (dtype-dependent) is what makes "surviving axis = VL" coincide with "acc = one physical register." If VL were dtype-independent (a free logical count), the surviving axis could be `VL` logically yet span multiple physical vregs, breaking register direct-pass.

**Acc lifetime is the second gate** (replacing the earlier "traversal-direction consistency" framing, which was wrong). Even with acc a single vreg, same-loop fusion needs the acc produced per-iteration and consumable in the same iteration (RowMax-style). ColMax-style's acc accumulates across all iterations and completes only after the loop, so it splits into two loops despite both styles traversing `N`. The ColMax (cheap `vmax`, no fuse) vs RowMax (expensive `vcmax`, fuses) trade-off is a cost-model question, left open.

**Structural consequence — tileop is a single loop.** Because the inner axis is VL = a single vreg (not a loop axis), a tileop is a single `scf.for` over `N` only — no 2-layer nested loop. PyPTO2's 2D nested-loop form (and its "axis-merge vs not" version choice) does not exist in VMI. This is a direct consequence of register direct-pass: the inner axis no longer being a loop dimension removes the need for nesting.

## Future Work

The current scope ties VL to dtype (f32→64, f16→128, fp8→256) — a tile's surviving axis has exactly one legal VL per dtype. A future extension relaxes this to allow `VL ∈ {64, 128, 256}` as a free choice per tile (e.g. an f32 tile with VL=128, spanning 2 physical vregs), by supporting **group reduce** (`{group=C}`, the reduce-B form currently out of scope) to keep the acc a single vreg across multiple physical registers. This is future work; the current pass does not support it.

## Consequences

- TileLang/TileOP tile shapes are no longer arbitrary: the surviving axis of any reduce must equal the dtype's VL. Authors cannot pick, e.g., a surviving axis of 100 for f32.
- A tile that "wants" a larger surviving axis must restructure as more `N × VL` blocks (or wait for Future Work), not stretch the surviving axis.
- RowMax-style reduce (along inner VL) requires `N = VL` — an additional tile-shape constraint on top of ColMax-style.
- **Tail has exactly one source.** Under "row/col fits one vreg," the only tail is the last row/column being short when the total element count is not a multiple of VL. This is simultaneously the outer-`N` tail (last loop iteration) and the last-row vreg-internal lane tail — same thing, two views. There are no other tail sources: partial-vreg intermediates do not arise, because tiling never produces a compact tile. The tail is absorbed by `[pmode]` masks on the in-loop element-wise/reduce ops (the in-loop consumers of the load), since `vload` itself is not predicateable on A5.
- The compact lane-group reduce `{group=C}` (`vcadd`/`vcmax`/`vcmin` with `{group=C}`, produces `V<C×T>`, C∈{1,2,4,8}) is **out of scope** for same-loop element-wise fusion (see glossary `寄存器直传` scope boundary). It still exists as a Category-B op (§2.2) and connects to element-wise only via `vbrc {group=C}`. Tiling is forbidden from producing a compact tile that would require it. (Note: `vcmax` without `{group=C}` is a different op — whole-row reduce used by RowMax-style, in scope.)
- Adding a new VL value is an ABI-level change: every reduce/broadcast register-residency derivation must be re-audited.
- The constraint is structural, not advisory — it is enforced at vmi type legality. §1.1 of `../PTO-vmi-design.md` already requires `L` to be a multiple of 64 for all dtypes; this ADR tightens it to "the surviving axis equals the dtype's VL (256B/bitwidth)".
