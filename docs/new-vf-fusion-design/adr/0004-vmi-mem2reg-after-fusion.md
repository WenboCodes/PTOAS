# vmi mem2reg pass, runs after fusion, eliminates tileop UB store-load

## Context

PyPTO2 splits this work into two passes (`../PyPTO2-vf-fusion-analysis.md` §5.3): `HiIPUVFLoopFusion` (merge loops, no store-load elimination) and `HiIPUVFLoadStoreElim` (eliminate store-load pairs). The fatal ordering: LoadStoreElim runs **before** Fusion, so it only sees same-BB pairs; pre-fusion the two loops are in different BBs (nothing to eliminate); post-fusion they're in one BB but LoadStoreElim doesn't re-run — newly-exposed pairs are never eliminated. Compounded by LoadStoreElim's own conservatism: it refuses DINTLV dist pairs (§5.2.3), is blocked by scalar UB ops (§5.2.4), and requires exact Preg match (§5.2.3).

## Decision

Replace both PyPTO2 passes with a single **vmi mem2reg** pass that runs **after** fusion:

- Fusion merges independent tileop loops into one shared loop (per ADR-0001/0002/0003 eligibility).
- mem2reg then runs on the fused loop. A `vstore %v, %a[%i,%j]` whose location (shaped-ptr index, ADR-0003) is dominated by a later `vload %a[%i,%j]` is promoted to SSA: `%v` flows directly to the load's users, the store-load pair is eliminated, data moves through vreg instead of UB (deep fusion).
- Cross-iteration store-load (loop-body `vstore` to a later iteration's `vload`) promotes to `scf.for` `iter_args`.

## Scope

mem2reg eliminates **tileop-to-tileop** UB store-load (the PyPTO2 LoadStoreElim target). Its relationship to a reduce's acc depends on the reduce orientation (ADR-0001):

- **RowMax-style** (fused in one loop): the acc is produced per-iteration and consumed in the same iteration as an `scf.for` `iter_args` SSA value, never passes through UB — mem2reg has nothing to promote. The acc and mem2reg are orthogonal here.
- **ColMax-style** (two loops): the acc DOES pass through UB — loop-1 `vstore %acc` → loop-2 outer `vload`. This whole-vreg store-load IS promoted by mem2reg (it is tileop-to-tileop UB traffic). So "acc is untouched by mem2reg" holds only for RowMax-style.

What mem2reg never touches is the **in-loop** acc update (the per-iteration `vmax`/`vadd` into the `iter_args` acc) — that is register SSA, not UB.

## Why

- **Fixes the ordering bug**: by running after fusion, every store-load pair is already in one BB/loop, so mem2reg sees them all. No "newly-exposed-but-never-eliminated" gap.
- **Resolves LoadStoreElim's three conservative refusals**:
  - *DINTLV dist unsupported* → mem2reg keys on the **shaped-ptr index location**, not on dist-mode. A DINTLV store-load pair at the same index is eliminated regardless of dist.
  - *Scalar UB blocks scan* → mem2reg promotes only vreg store-load; scalar UB ops don't terminate the scan.
  - *Preg mismatch* → vmi masks are first-class (`plt` / `[pmode]`) and travel with the promoted value; no separate Preg-match gate.
- **Replaces ad-hoc pair-matching with a standard SSA promotion**, which MLIR tooling already supports.

## Consequences

- Pipeline order: fusion → mem2reg. mem2reg depends on fusion having merged the loops; running it pre-fusion would reproduce PyPTO2's gap.
- mem2reg's precision depends on the shaped-ptr index (ADR-0003): if two tileops' index expressions cannot be proved same-location (affine-equivalent), the store-load is not eliminated — MayAlias-style conservatism returns, but at the index-expression level, far sharper than PyPTO2's UB-address interval.
- `vload` is not predicateable on A5 (ADR-0002): a load reads a full `VL` including padding. After mem2reg promotes a store-load pair, the consumer's `[pmode]` must still mask the padding lanes — promotion moves the value, not the mask obligation.
- **Capability boundary: mem2reg promotes same-location same-shape store→load.** It does NOT depend on reduce orientation. RowMax-style (reduce fused in-loop, acc passed directly, no UB) and ColMax-style (two loops: reduce, then element-wise) both have their acc traffic as whole-`V<VL×T>` store→whole-`V<VL×T>` load at the same location — both promote. In ColMax-style, the pair is loop-1's `vstore %acc` → loop-2's outer `vload %max_vec` (whole vector, once); how loop-2 internally consumes `max_vec` (column-aligned etc.) is post-promotion SSA, not UB traffic, and doesn't gate mem2reg. The true boundary is "same location + same shape" — a write-whole/read-slice asymmetry would NOT promote, but that pattern does not arise here because the inter-loop handoff is always whole-vreg.
- Reduce acc and mem2reg: orthogonal for RowMax-style (acc is in-loop `iter_args`, never UB); promoted for ColMax-style (acc crosses loops via whole-vreg store→load). What mem2reg never touches is the in-loop acc update (per-iteration `vcmax`/`vmax` into `iter_args` for RowMax), which is register SSA. Fusion must emit RowMax-style reduces with acc as in-loop `iter_args` (per ADR-0001); ColMax-style reduces with acc as a cross-loop UB value that mem2reg then promotes.
- mem2reg's location reasoning and fusion's alias check share the **same shaped-ptr index math** (both ask "do these two accesses hit the same UB location?"), but the two passes run separately (fusion asks conflict, mem2reg asks promotability). No coupling of internal data structures; analysis caching handles any reuse.
