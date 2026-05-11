# Grouping-aware optimization for SLOS / QuantumLayer

Date: 2026-04-05

## Problem statement
MerLin exposes a **grouping/binning** facility (e.g. `LexGrouping`, `ModGrouping`) that reduces the size of the feature vector returned by a `QuantumLayer` by **summing buckets of probabilities**.

Today, the forward pass computes the **full probability distribution** over output basis states (size exponential/combinatorial in modes/photons), and only **afterwards** applies grouping.

Because the strong simulator (SLOS) has memory scaling dominated by the size of the *last-layer* state vector, the goal is:

- keep the same user-facing semantics (grouped probabilities),
- but reduce peak memory / allocations by anticipating grouping during simulation, ideally without materializing the full final distribution.

## What exists today (code + doc)

### 1) Where grouping lives
- Grouping modules are in `merlin/utils/grouping.py`:
  - `LexGrouping(input_size, output_size)` groups consecutive indices into equal-width buckets (with zero padding).
  - `ModGrouping(input_size, output_size)` groups by index modulo `output_size`.
  - Both act on the **last dimension** of a tensor and are differentiable.

Docs:
- `docs/source/user_guide/grouping.rst` introduces these as **post-processing**.
- `docs/source/user_guide/measurement_strategy.rst` describes grouping as orthogonal to `MeasurementStrategy`.

### 2) Where grouping is applied in a `QuantumLayer.forward`
`QuantumLayer.forward` (in `merlin/algorithms/layer.py`) runs roughly:

1. Compute amplitudes via `ComputationProcess` (SLOS-based for strong simulation).
2. Convert amplitudes to probabilities and (when required) renormalize via `normalize_probabilities_and_amplitudes` (`merlin/utils/normalization.py`).
  - Note: this step is executed unconditionally in `QuantumLayer.forward` today.
  - For `PARTIAL`, the returned `distribution` is not used by `PartialMeasurementStrategy.process(...)`, but the returned (possibly renormalized) `amplitudes` are.
3. Apply photon loss and detector transforms (if configured):
  - for distribution-based strategies, transforms act on the **probability distribution**;
  - for partial measurement, detector transforms act on **complex amplitudes** before branching (via `DetectorTransform(partial_measurement=True)`).
    Photon loss is **not supported** in partial mode (non-identity photon loss triggers an error during layer initialization), so `apply_photon_loss(amplitudes)` is effectively a no-op.
4. Optional sampling (only for distribution-based strategies).
5. **Apply grouping** if `measurement_strategy.grouping` is provided:
  - for `PROBABILITIES`: grouping is applied to the (possibly transformed/sampled) **distribution**;
  - for `PARTIAL`: grouping is applied to branch probabilities inside `PartialMeasurement` construction.
6. Apply the output mapping `self.measurement_mapping(...)` (`merlin/measurement/mappers.py`) to obtain the final feature representation.
  - In particular, `MODE_EXPECTATIONS` reduction happens here (mask-based marginalization in `ModeExpectations`).

Concretely, the forward pass can be decomposed into the following stages (with the main code locations):

**(a) Compute probability amplitudes**
- Entry point: `QuantumLayer.forward` → `QuantumLayer._compute_amplitudes` (`merlin/algorithms/layer.py`).
- For strong simulation (SLOS): `ComputationProcess.compute(...)`, `compute_superposition_state(...)`, and `compute_ebs_simultaneously(...)` (`merlin/core/process.py`) call into:
  - `SLOSComputeGraph.compute(...)`
  - `SLOSComputeGraph.compute_batch(...)`
  - `SLOSComputeGraph.compute_pa_inc(...)`
  in `merlin/pcvl_pytorch/slos_torchscript.py`.

**(b) Convert amplitudes → probabilities (and renormalize when needed)**
- `QuantumLayer._renormalize_distribution_and_amplitudes` calls `normalize_probabilities_and_amplitudes(...)` (`merlin/utils/normalization.py`).
- The raw conversion itself is `probabilities_from_amplitudes(...)` (`merlin/utils/normalization.py`).

**(c) Apply photon loss and detector transforms (if configured)**
- For distribution-based strategies, this occurs in `DistributionStrategy.process(...)` (`merlin/measurement/strategies.py`) via:
  - `apply_photon_loss(distribution)` → `QuantumLayer._apply_photon_loss_transform` (bound method passed from `QuantumLayer.forward`)
  - `apply_detectors(distribution)` → `QuantumLayer._apply_detector_transform` (bound method passed from `QuantumLayer.forward`)
- For partial measurement, transforms are applied on **amplitudes** in `PartialMeasurementStrategy.process(...)` (`merlin/measurement/strategies.py`):
  - photon loss is required to be identity (partial mode raises if photon loss is non-trivial),
  - detector transform runs in `partial_measurement=True` mode and expects a complex amplitude tensor.

**(d) Optional sampling**
- Implemented by `SamplingProcess.pcvl_sampler(...)` (`merlin/measurement/process.py`), called from `DistributionStrategy.process(...)` through the `sample_fn` callback created in `QuantumLayer.forward`.

**(e) Apply grouping (only for PROBABILITIES and PARTIAL in the current QuantumLayer implementation)**
- For probabilities: `DistributionStrategy.process(...)` applies `grouping(distribution)` at the end (`merlin/measurement/strategies.py`).
- For partial measurement: grouping is forwarded to `partial_measurement(...)` (`merlin/measurement/process.py`), which delegates to `PartialMeasurement.from_detector_transform_output(..., grouping=...)`.

The grouping hook is effectively in `merlin/measurement/strategies.py`:
- `DistributionStrategy.process(...)` does:
  - photon loss → detectors → sampling → `grouping(distribution)`.

Net effect: grouping is a **late** step and currently receives a **distribution tensor**.

### Where “mode expectations” reduction is computed
For `MeasurementKind.MODE_EXPECTATIONS`, the per-mode reduction is not performed in `DistributionStrategy.process(...)`.
Instead, it happens in the *output mapping* stage applied after the measurement strategy returns a (possibly transformed/sampled) probability distribution:

- `QuantumLayer.forward` ends with `return self.measurement_mapping(results)`.
- `self.measurement_mapping` is created by `OutputMapper.create_mapping(...)` (`merlin/measurement/mappers.py`).
- When the kind is `MODE_EXPECTATIONS`, the mapper is `ModeExpectations` (`merlin/measurement/mappers.py`):
  - `ModeExpectations.forward(...)` converts amplitudes/probabilities to probabilities (via `Probabilities()`), then
  - computes per-mode expectations with `probability_distribution @ self.mask.T` in `ModeExpectations.marginalize_per_mode(...)`.

So, the “mode expectation reducing” happens in `merlin/measurement/mappers.py::ModeExpectations` (mask-based marginalization), not in `merlin/measurement/strategies.py`.

### 3) How SLOS is used by QuantumLayer
`ComputationProcess` (in `merlin/core/process.py`) builds:
- a `CircuitConverter` (unitary graph)
- a `SLOSComputeGraph` via `build_slos_distribution_computegraph` (`merlin/pcvl_pytorch/slos_torchscript.py`)

Then `compute(...)` calls `self.simulation_graph.compute(unitary, input_state)` which returns the **full final amplitude vector**.

Key SLOS implementation details (`merlin/pcvl_pytorch/slos_torchscript.py`):
- The graph is built layer-by-layer, one photon at a time.
- Each layer uses `layer_compute_vectorized(...)` which allocates a dense `result` of size `next_size` and fills it via `scatter_add_`.
- The *largest* allocation is typically the **last layer** (all output states).

SLOS already supports an `output_map_func` that can aggregate output states into fewer keys, but MerLin’s `ComputationProcess._setup_computation_graphs()` currently builds the graph without passing `output_map_func`.

## Important constraint: what grouping can and cannot save

### Grouping does not change the intrinsic strong-simulation cost
For generic grouping (summing probabilities of many orthogonal output states), SLOS still needs to compute amplitudes for each micro output state.

So grouping **does not reduce the asymptotic compute** of strong simulation in the generic case.

### But grouping can reduce peak memory and large transient allocations
The key opportunity is to avoid materializing:
- the full final amplitude vector (complex) **and/or**
- the full probability vector (real)

when the user only needs a small grouped output.

This is especially relevant because:
- `layer_compute_vectorized` always allocates the full `result` tensor for a layer,
- the last layer is the biggest, and
- the grouping output size can be orders of magnitude smaller.

## Compatibility constraints (measurement pipeline)
Grouping is currently applied **after**:
- photon loss transform
- detector transform
- sampling (if enabled)

Any grouping-aware optimization must preserve this ordering.

Why this ordering matters:
- Photon loss is a linear map on the **probability vector** expressed in the simulator basis, and it generally changes the effective output basis (states with fewer surviving photons appear after the transform).
- Grouping is also a linear map, but it is defined over the **current output indexing**. In practice, the grouping used by `QuantumLayer` is built for the post-transform output size, not the pre-loss simulator size.
- Therefore, loss and grouping do **not** commute in general: applying grouping first and loss afterwards would require a different reduced loss operator `L'` such that `G @ L == L' @ G` (up to orientation conventions), which is only true for specific, reduction-compatible groupings.
- For arbitrary groupings such as `LexGrouping` or `ModGrouping` on raw state indices, there is no reason to expect this commutation relation to hold.

Important nuance for `ComputationSpace.DUAL_RAIL`:
- `DUAL_RAIL` constrains the **ideal simulation basis** used by SLOS (one photon per rail pair in the no-loss model).
- A subsequent photon-loss transform is still meaningful: it maps that ideal dual-rail distribution to a larger post-loss basis that can contain states that are **not dual-rail anymore** (for example, vacuum on one qubit pair after a photon is lost).
- So, in the current design, `DUAL_RAIL` does **not** mean “all downstream probability vectors remain dual-rail”; it means “the exact interferometric simulation is restricted to the dual-rail sector before measurement noise/loss is applied.”

Physical caveat:
- If `DUAL_RAIL` is interpreted as a **post-selection rule** on physically valid outputs, then applying it only to the ideal pre-loss simulation is not fully coherent.
- In that stricter physical interpretation, loss should happen first on the full physical model, and the dual-rail condition should be enforced afterwards as a filtering/post-selection criterion on the measured outcomes.
- Therefore, the current `DUAL_RAIL + photon loss` behavior is better understood as an **algorithmic encoding restriction** (simulate the ideal circuit in the dual-rail logical sector, then apply a post-processing loss model), not as an exact model of “loss followed by dual-rail post-selection”.
- If exact post-selected physical semantics are required, the simulation should instead run in a larger physical space (typically `FOCK` or another non-post-selected basis), apply loss, then filter or renormalize on the subset of outcomes that still satisfy the dual-rail condition.

The same caveat applies, more mildly, to `UNBUNCHED`:
- `UNBUNCHED` in Merlin also restricts the **simulated ideal sector** in step (1).
- After a post-processing photon-loss transform, the resulting states remain collision-free (loss cannot create bunching), but the distribution is still not the exact same object as “full physical evolution with loss, then post-select onto the unbunched `n`-photon sector”.
- The reason is that, in the full physical model, some amplitudes that would bunch before loss could still contribute to valid post-loss outcomes after photons are removed. Those paths are absent when the ideal simulation was restricted to the unbunched sector from the start.

Practical consequence:
- If `UNBUNCHED` / `DUAL_RAIL` are meant as **algorithmic simulation restrictions**, allowing loss is consistent with the current Merlin design, but the semantics should be documented as such.
- If `UNBUNCHED` / `DUAL_RAIL` are meant as **strict physical post-selection rules**, then non-trivial photon loss should indeed be disabled (or ignored) in that mode, because the exact physically coherent alternative is to simulate in a larger space, apply loss, and only then post-select.
- In that strict post-selected interpretation, successful post-selected events are effectively conditioned on “no invalid event occurred”, so the retained normalized distribution is insensitive to loss except through the reduced success probability.

Therefore:
- **If photon loss / detectors / sampling are enabled**, we generally cannot push grouping inside SLOS unless we also fuse those transforms (or can represent them as linear maps applied before grouping).
- The biggest “safe” optimization target is the common case:
  - strong simulation
  - `shots == 0` (no sampling)
  - ideal detectors + no photon loss transform (identity transforms)
  - measurement kind `PROBABILITIES`
  - grouping present

### Could photon loss be applied “at the source” instead?
Conceptually, yes, but this is a different execution strategy with different tradeoffs.

Exact source-side loss would mean:
- replacing the single input Fock state by a **mixture of surviving input states**,
- then running the ideal interferometer on each surviving input sector,
- and summing the resulting output distributions with the appropriate binomial weights.

If the input state is `s = (s_1, ..., s_m)` and per-mode survival probabilities are `eta_i`, then source-side loss generates all survivor states `r <= s` with weights proportional to:

$$
\prod_i \binom{s_i}{r_i} \eta_i^{r_i} (1-\eta_i)^{s_i-r_i}.
$$

Potential advantage:
- each sub-simulation may involve fewer photons than the original `n`, so the SLOS state space per run can be much smaller.

Main drawback:
- exact evaluation now requires **multiple SLOS runs** (one per survivor input sector, or an equivalent multi-sector implementation), instead of one ideal SLOS run followed by a linear photon-loss transform.

So source-side loss is not automatically more efficient:
- it may help when loss is strong and the number of relevant survivor sectors is small,
- but it may be worse when many survivor sectors are possible, because the number of simulations grows combinatorially with the input occupation pattern.

For the current MerLin implementation, output-side photon loss is attractive because:
- SLOS still runs only once on the original ideal circuit,
- photon loss is then applied as a linear post-processing transform on probabilities,
- and this integrates naturally with the existing detector/sampling/grouping pipeline.

In short: source-side loss is a plausible alternative optimization strategy, but not an obvious universal win. It would need benchmarking against the current “one ideal run + post-loss transform” approach.

## Proposed approach (actionable)

### Proposal A (low risk, immediate win): fuse “probabilities + normalization + grouping” to avoid an extra full distribution allocation
**Idea:** today, `QuantumLayer.forward` always computes a full `distribution` tensor from amplitudes, then (optionally) groups it.

We can instead compute grouped probabilities directly from amplitudes:

- compute per-state probabilities on the fly as `p = amplitudes.real**2 + amplitudes.imag**2`
- aggregate into buckets via `scatter_add_` / `index_add_`
- normalize once at the end using the bucket sums (for `UNBUNCHED` / `DUAL_RAIL`)

This avoids allocating (and potentially keeping alive) a full probability tensor distinct from amplitudes.

What it saves:
- Memory: removes one large real tensor of size `#states`.
- Bandwidth: avoids writing the full probability vector when only grouped results are needed.

What it does *not* save:
- The final amplitude tensor still exists (because SLOS returns it).

Implementation sketch:
- Add an internal helper (e.g. `group_probabilities_from_amplitudes(amplitudes, group_indices, output_size, computation_space)`)
- Detect groupable groupings (`LexGrouping`, `ModGrouping`) in `QuantumLayer.forward` when strategy kind is `PROBABILITIES`.

### Proposal B (higher impact): grouping-aware last-layer evaluation in SLOS to avoid materializing the final amplitude vector
**Idea:** the biggest peak allocation is the last SLOS layer output (`next_size == #output_states`).

If the user only needs **grouped probabilities**, we can compute the last layer in **chunks** and immediately accumulate `|amp|^2` into buckets, never holding the full final amplitude vector.

Key observation:
- For a given output state, the amplitude is a sum of contributions from parent states (interference happens within that state), so we must compute the final **amplitude per destination** before squaring.
- But we do not need to compute all destinations at once: we can compute a chunk of destinations, square, bucket-accumulate, and discard.

Algorithm sketch (single unitary, probabilities only):
1. Run SLOS layers `0..(n_photons-2)` as today (keeps the penultimate amplitude vector).
2. For the final layer:
   - iterate over destination ranges `[d0, d1)`
   - filter operations whose `destinations` are in this range
   - compute a temporary `result_chunk` via `scatter_add_`
   - compute `p_chunk = |result_chunk|^2` and aggregate into `grouped` via `scatter_add_` on group ids
3. Apply per-batch normalization if required (`UNBUNCHED` / `DUAL_RAIL`).

What it saves:
- Peak memory: avoids allocating the last-layer amplitude tensor of size `#output_states`.
- Also avoids allocating the full probability vector.

What it costs:
- Extra overhead to filter/reindex ops per chunk.
- More kernel launches (chunks) vs one big scatter.

Where to expose it:
- Add `SLOSComputeGraph.compute_grouped_probs(...)` (and possibly a batched variant).
- Gate it from `ComputationProcess` / `QuantumLayer.forward` when:
  - measurement kind is `PROBABILITIES`
  - grouping is present and recognized (Lex/Mod at first)
  - `shots == 0`
  - detector/photon loss transforms are identity
  - caller does not request amplitudes or `ProbabilityDistribution` object (see note below)

### Proposal C (medium-term): formalize grouping as an index mapping (“GroupingSpec”) to let SLOS reuse existing output mapping machinery
SLOS already has an aggregation path based on `output_map_func` + `target_indices` scatter-add.

However:
- `output_map_func` is keyed on the *Fock state tuple*, not the *lex index*.
- `LexGrouping` / `ModGrouping` are index-based.

A pragmatic abstraction is to represent grouping as:
- `output_size: int`
- `group_index_for_state_index(i): int` for all `i in [0, input_size)`

This mapping can be materialized as a `torch.LongTensor` of shape `[input_size]` once, reused across forwards.

Action:
- Add a small adapter utility:
  - `lex_group_indices(input_size, output_size, device)`
  - `mod_group_indices(input_size, output_size, device)`
- Optionally extend `LexGrouping` / `ModGrouping` with a method (non-breaking) that returns this mapping.

This spec is also what Proposal A and B need.

## Edge cases / correctness considerations

1) **Detectors / photon loss / sampling**
- Current ordering is: transforms → sampling → grouping.
- Proposal A and B are only strictly correct if grouping is applied at the same point.
- Therefore, initial implementation should only activate when transforms are identity and `shots == 0`.

2) **Computation spaces requiring renormalization** (`UNBUNCHED`, `DUAL_RAIL`)
- Current code renormalizes using `sum_probs`.
- With grouped outputs, we can compute `sum_probs` as `grouped.sum(-1, keepdim=True)` since grouping is a partition + sum.

3) **`return_object=True` with grouping**
- `QuantumLayer.forward` can wrap outputs into `ProbabilityDistribution(...)` even if grouping changed the last dimension.
- `ProbabilityDistribution` only validates basis size in `from_tensor`; the bare constructor doesn’t validate against `(n_modes, n_photons, computation_space)`.
- Grouped features are not a physical Fock distribution; wrapping them into `ProbabilityDistribution` is likely misleading.

Recommendation:
- As part of the optimization work (or independently), decide and document behavior:
  - either forbid `return_object=True` when grouping is present for `PROBABILITIES`,
  - or return a plain `torch.Tensor` regardless,
  - or support a `ProbabilityDistribution` with a custom “grouped basis” (requires design work).

## Proposed implementation plan (concrete steps)

### Step 1 — Observability + constraints
- Add a small benchmark script or reuse `benchmarks/benchmark_slos_core.py` to measure:
  - peak memory
  - runtime
  - for a representative `(m, n_photons)` and output grouping size.

### Step 2 — Implement Proposal A (fused grouped probs from amplitudes)
- Add grouping-index helpers (probably under `merlin/utils/grouping.py` or a new small module).
- In `QuantumLayer.forward`, when kind is `PROBABILITIES` and grouping is `LexGrouping` or `ModGrouping` and transforms are identity and `shots==0`:
  - compute grouped probs directly from amplitudes,
  - normalize appropriately,
  - skip materializing full `distribution`.

Add tests:
- Verify `grouped == grouping(full_distribution)` for:
  - unbatched + batched
  - `ComputationSpace.FOCK` and `UNBUNCHED` normalization

### Step 3 — Implement Proposal B (chunked last-layer + grouped accumulation)
- Extend `SLOSComputeGraph` with a new method that:
  - computes layers up to `n_photons-1`
  - final layer computed in chunks
  - accumulates grouped probabilities

Then wire it through:
- `ComputationProcess` to call it when eligible
- `QuantumLayer.forward` fast-path

Add tests:
- Compare against baseline SLOS `compute(...)` → probs → grouping for small sizes.

### Step 4 — Expand eligibility and generalize
- Optionally support:
  - `ModGrouping` and `LexGrouping` for batched unitaries
  - `compute_batch` paths for amplitude-encoding workflows (only if measurement kind is probs)
- Explore if detector transform can be represented as a sparse linear map on distributions; if yes, fusing becomes possible (more complex, likely a separate project).

## Bottom line
- Grouping cannot generally reduce the intrinsic strong-simulation compute, but it **can** reduce peak memory and avoid large allocations.
- The best ROI is a two-stage approach:
  - quick win: fuse probabilities+normalization+grouping (Proposal A)
  - higher impact: compute the last SLOS layer in chunks and bucket-accumulate (Proposal B)

This aligns with the docs hint that `MeasurementStrategy` can carry information for SLOS optimizations, and it targets exactly the “last layer” memory cliff that currently limits scale.


---

## Summary + proposal: make SLOS reduction-aware without changing the API

Goal: let MerLin exploit an **output reduction** (grouping today, other reductions tomorrow) to lower SLOS **peak memory**, while keeping **exactly the same public API** (same signatures, same user-facing objects).

### Principle
Do not add any new parameters to `QuantumLayer.forward` nor to `MeasurementStrategy`. Instead:

1) **Compile an internal plan** from already-available objects:
- `measurement_strategy` (kind, computation_space)
- `measurement_strategy.grouping` (if present)
- effective presence of photon loss / detectors / sampling (shots)

2) **Route** to an optimized path only when it is *semantically safe*; otherwise fall back to the current implementation.

### Internal building block: `OutputReducer` (not exposed)
Introduce a small internal concept (private module or internal class) representing an output reduction, e.g.:

- `output_size: int`
- `group_index: torch.LongTensor` of shape `[#states]` (maps each output state index to a bucket)
- `stage: Literal["probabilities"]` (in a first iteration)
- `is_compatible(...) -> bool` (checks shots/transforms/kind/space)

Crucially, the reducer is **derived by introspection** of existing instances.
- If `grouping` is a `LexGrouping`/`ModGrouping`, we already have `input_size` and `output_size` and can build `group_index`.
- No public API change is required.

### Routing (no API change)
Inside `QuantumLayer.forward` (or a helper in `ComputationProcess`), build an internal “ForwardPlan” and decide:

- **Baseline path** if:
  - shots > 0, or
  - photon loss / detectors are non-trivial, or
  - kind ≠ PROBABILITIES, or
  - grouping is not recognized / no reducer available.

- **Optimized path** otherwise (first target):
  - strong simulation, `shots == 0`, identity transforms, kind=PROBABILITIES, grouping ∈ {Lex, Mod}.

This preserves the contract: same user code, same returned type (tensor), but MerLin “optimizes when it can”.

### Two reduction levels (API unchanged)

**Level 1 (quick, low risk):**
Fuse “probabilities + normalization + grouping” from the amplitudes returned by SLOS, avoiding allocation of a full `distribution` tensor.
- Benefit: removes one allocation of size `#states`.
- API unchanged.

**Level 2 (large memory win):**
Make SLOS able to evaluate the **last layer in chunks** and accumulate `|amp|^2` directly into buckets via `group_index`, without materializing the full final amplitude vector.
- Benefit: removes the most expensive allocation (final amplitudes) and removes the full distribution allocation.
- Can be implemented via an internal method (e.g. `SLOSComputeGraph._compute_reduced_probs(...)`) without exporting it in `merlin/__init__.py`.

### Generalization (future reduction-aware)
The same mechanism (`OutputReducer`) can represent other reductions already present **without touching the API**:
- `ModeExpectations` is a linear reduction `probs @ mask.T` → streaming accumulation is possible.
- any future feature that is a sum/aggregation over probabilities.

### Compatibility guardrails
Because the current order is: photon loss → detectors → sampling → grouping, the first iteration should only enable the optimization if:
- `shots == 0`,
- transforms are identity (no photon loss / no custom detectors),
otherwise fall back.

### Suggested rollout
1) Implement `OutputReducer` + routing + Level 1 (+ equality tests vs `grouping(full_distribution)`).
2) Add Level 2 (chunked last layer) behind strict eligibility (and optionally an internal feature gate).
3) Gradually expand eligibility (batched unitaries, additional reducers) once invariants are well understood.

## Validated conclusion

### What could be optimized?
When combining SLOS with grouping and, more generally, with downstream measurement reductions, the main inefficiency is that SLOS materializes a very large late-stage output tensor before Merlin applies the final downstream reductions (grouping, detector/loss transforms, output mappings, or post-selection-like restrictions).

This suggests two optimization levels:

1. **Direct last-layer reduction when the downstream reduction is simple and semantically compatible**
  - For plain grouping in the no-loss / no-sampling / identity-detector case, the last large SLOS stage can be made reduction-aware.
  - In that setting, grouped probabilities can be accumulated directly from that late SLOS stage instead of materializing the full probability vector.
  - This can save both memory peak and some memory bandwidth, and may also reduce runtime.

2. **Chunked last-layer evaluation when downstream processing must still happen afterwards**
  - For more complex flows (for example photon loss, detector transforms, or other post-processing that cannot be pushed into SLOS as-is), the last SLOS layer can still be evaluated in blocks/chunks.
  - Each chunk can be squared / transformed / accumulated incrementally, avoiding the large final allocation.
  - This mainly saves peak memory; it does not fundamentally reduce the amount of exact computation.

Important precision:
- For photon loss specifically, the loss map acts on **probabilities after squaring**, so it cannot in general be folded directly into the same final amplitude `scatter_add_` as if it were just another amplitude-space grouping.
- The correct optimization in that case is a **streaming / chunked** pipeline: compute final amplitudes per block, convert to probabilities, apply downstream transforms, then accumulate.

### Behaviors that should be fixed independently of the optimization

1. **Support photon loss in partial measurement**
  - If Merlin is intended to model the physically correct pipeline, partial measurement should support non-trivial photon loss.
  - This is not a small local fix: the current `PARTIAL` path is built around detector branching on amplitudes, while photon loss is implemented as a probability-space map.
  - Supporting both together requires redesigning the partial-measurement pipeline so that loss and partial detector branching compose consistently.

2. **Apply loss consistently with post-selection semantics**
  - If `UNBUNCHED` / `DUAL_RAIL` are intended as strict physical post-selection rules, then the current ordering is inconsistent.
  - The physically coherent behavior is: simulate in a larger physical space, apply loss, then post-select/filter valid outcomes.
  - So this is a semantic bug/oversight, not merely a missed optimization.

3. **Avoid computing probabilities that are then ignored**
  - In the current `PARTIAL` path, `QuantumLayer.forward` always computes `distribution = |amplitudes|^2`, even though `PartialMeasurementStrategy.process(...)` does not use that tensor.
  - This should be removed or bypassed in the partial path, keeping only the amplitude renormalization actually needed for the chosen computation space.

### Recommended framing
- Treat the SLOS/grouping work as an **optimization project**.
- Treat the `UNBUNCHED` / `DUAL_RAIL` + loss semantics, and `PARTIAL` + loss support, as a **correctness / model-semantics project**.
- Treat the unconditional probability computation in `PARTIAL` as a **small cleanup/performance fix**.
