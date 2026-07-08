# Buffer capacity `N` static, `valid row` dynamic; dynamicity split across two layers

## Status

Accepted. Supersedes the original framing in this ADR's first draft, which incorrectly claimed the `scf.for` upper bound is always a static `N` and that dynamicity is carried solely by the mask. Corrected after the row-level vs lane-level dynamicity split was clarified.

## Context

PyPTO2's VF fusion rejects loops whose trip count is dynamic: `HiIPUVFLoopFusion` compares backedge-taken counts via SCEV, and `SCEVCouldNotCompute` (the case for loops whose valid-row boundary is run-time) means refusal (`../PyPTO2-vf-fusion-analysis.md` §5.1.1). Real kernels (softmax over run-time `seq_len`, dynamic-`M` matmul, tail blocks) routinely have exactly this shape.

## Decision

Separate three notions:

- **`N`** — the tile's **buffer capacity** (compile-time constant; how many `V<VL×T>` rows the tile can hold). Static, so buffer allocation is static.
- **`valid row`** — the run-time-known count of valid rows (`≤ N`).
- **lane-level tail** — the last row's partial-VL shortfall.

Dynamicity is split across **two layers**, carried differently depending on the loop form:

- **row level** (how many rows run): either
  - **Form A**: loop runs to static `N`; rows beyond `valid row` are masked out by the lane mask degenerating to 0-active; OR
  - **Form B**: loop runs to `%valid_row` (dynamic trip count); no overrun iterations run.
- **lane level** (last row's tail): always a `pto.vmi.mask` from `plt`, consumed by in-loop ops via `[pmode]`. Form A derives it from the `iter_args` counter `remaining`; Form B derives it from `valid row - %offset`.

Both forms are legal; Form B is preferred (no masked-empty iterations).

The reduce-acc vreg identity is static **not** because `N` is static, but because the acc is an `scf.for` `iter_args` SSA value reused across iterations (see ADR-0001). This holds in both forms and is independent of whether the trip count is static.

## Why

This is the mechanism by which VMI bypasses PyPTO2 §5.1.1 without giving up dynamic-shape support. PyPTO2 refused fusion because it fused *two independent loops* and needed their SCEV-computed backedge-taken counts to match — a dynamic boundary broke the match. VMI's fusion pass keeps a trip-count-consistency check (structurally similar to PyPTO2's), but it runs in the MLIR layer and passes when the two loops share the **same `%valid_row` SSA value** (trivially consistent, no SCEV needed) or when affine analysis can prove equivalence. The lane-level tail, being a first-class `pto.vmi.mask` operand, never enters the trip-count comparison.

## Consequences

- `N` static means static buffer allocation; it does NOT mean static trip count. A future reader seeing a static `N` plus a dynamic `scf.for` upper bound should not be surprised — they measure different things (capacity vs. rows-run).
- Trip-count-consistency is still a fusion eligibility check (Form A: static-N consistency; Form B: same-`%valid_row`-SSA-value or affine-provable equivalence). This is closer to PyPTO2's check than the original draft claimed — the gain is moving it from the LLVM layer (where SCEV fails on `pto.mi`'s POST_UPDATE/scatter patterns) to the MLIR layer (where structured tile analysis succeeds).
- Mask propagation must be sound and end-to-end: the lane-tail mask must reach every in-loop op's `[pmode]`, including the reduce's acc accumulation (inactive lanes per §2.3). This leans on §0.3's predicate-propagation rules.
- `vload` is not predicateable on A5 (§0.3-3): the tail load reads a full `VL` including invalid lanes; the in-loop ops' `[pmode]` masks ensure invalid lanes do not corrupt the acc.
- Form A's `iter_args` counter (`remaining`) is Form-A-specific — it exists because Form A has no dynamic upper bound to derive the tail from. Form B derives the tail from `valid row - %offset` and needs no such counter (though the reduce acc still threads through `iter_args` in both forms).
- RowMax-style reduce (ADR-0001) requires `N = VL`. Since VL is dtype-determined and static, RowMax-style's `N` is static and equals VL — but `valid row` is still dynamic (≤ `N`). So "N=VL" does NOT collapse the row-level dynamicity; it only fixes the buffer capacity to one vreg's worth of rows. Form A/B still apply on top.
