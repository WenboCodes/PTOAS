# `vload`/`vstore` take multidim index expressions; `!pto.ptr` carries shape/stride

## Context

VMI's VF-fusion pass judges eligibility in the MLIR layer (ADR-0002, replacing PyPTO2's LLVM-layer SCEV/alias analysis). Two of its checks — UB alias analysis and access-pattern compatibility (`../PyPTO2-vf-fusion-analysis.md` §5.1.1, §5.1.3) — depend on knowing the structured shape of each tileop's UB access. PyPTO2 lost this structure because `pto.mi`'s `vlds`/`vsts` fold addressing into a scalar offset plus POST_UPDATE, leaving the LLVM layer to guess strides.

## Decision

- **`vload`/`vstore` address operand is a multidim index expression** (e.g. `vload %a[%i, %j]`), NOT a folded scalar offset. The IR does not collapse the index dimensions; the induction variables and dimension structure remain visible for analysis.
- **Shape/stride live on the pointer type.** `!pto.ptr<T, ub>` extends to carry shape and stride (e.g. `!pto.ptr<T, ub, shape=[M,K], stride=[s0,s1]>`), memref-style. The index expression references only induction variables (`%i`, `%j`); static strides come from the pointer type, dynamic strides from its symbolic operands.
- This is **distinct from physical layout**. Shape/stride describe the UB memory space's logical geometry; layout (parity/half/part/pack) is a separate, compiler-internal axis owned by `pto.as`. Putting shape/stride on the ptr type does not violate "layout invisible to the user" — the user already reasons about `a[i][j]` geometry.

## Why

Access-pattern analysis becomes "compare two `vload`s' ptr shape/stride + index affine maps for compatibility" — the mature memref/affine-dialect analysis path, instead of PyPTO2's LLVM-layer guessing at POST_UPDATE/scatter patterns. The multidim index must NOT be folded (S1b, not S1a-as-sugar), because folding discards the very structure the analysis needs. Shape/stride on the ptr type (option S1, over S2 index-encoded-stride or S3 full-affine-map) keeps the index expression as pure induction variables, making the analysis cleanest.

## Consequences

- `vload`/`vstore` op signatures change (§3 of `../PTO-vmi-design.md` currently shows `vload %ub[%offset]` with scalar offset — must be updated to multidim index). The `!pto.ptr` type gains shape/stride operands. Both are cross-layer sync items (ODS, C++ verifiers, lowering patterns, tests, docs).
- Access-pattern analysis operates on induction variables + static strides ONLY. The dynamic `valid_row` (ADR-0002, Form B trip count) does NOT enter access-pattern analysis — it governs trip-count only. The two are orthogonal: `%valid_row` decides how many iterations run; `%i`/`%j` + stride decide each iteration's access pattern.
- Lowers to `pto.mi`'s scalar-offset `vlds`/`vsts` at the `pto.as` stage, where the multidim index + stride are folded into the physical address — folding is deferred to lowering, not done in the vmi IR.
- **Alias-analysis fallout (replaces PyPTO2 §5.1.2's three checks):** the shaped ptr + multidim index let the MLIR-layer fusion pass compute the exact UB region each `vload`/`vstore` touches, upgrading PyPTO2's "MayAlias/MustAlias ⇒ refuse" to "regions don't overlap ⇒ fuse".
  - **W-W** (both loops write): fully resolved — distinct regions fuse even under ptr MayAlias.
  - **R-W / WAR** (loop0 reads, loop1 writes): fully resolved — exact regions replace PyPTO2's conservative `hasValidDependency` MustAlias chain.
  - **W-R / RAW** (loop0 writes, loop1 reads): region-precision gained, BUT the A5 constraint "`vload` is not predicateable" (ADR-0002) still bites — a load reads a full `VL` including padding, so RAW safety still depends on the consumer's `[pmode]` mask correctly excluding the stored padding. Not a pure win: mask propagation is load-bearing for RAW.
