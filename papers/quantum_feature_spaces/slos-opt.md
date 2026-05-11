# Review of SLOS / ComputationProcess optimization opportunities

Date: 2026-04-05

## Scope
This note summarizes a static review of the current `ComputationProcess` / `pcvl_pytorch` implementation in MerLin, with a focus on:

- compute hotspots,
- memory hotspots,
- current use of TorchScript / graph-based execution,
- concrete improvement options.

The goal is to identify where the current implementation spends most of its time and memory, and what changes would provide the best return on complexity.

## Executive summary
At the core of MerLin, `ComputationProcess` and `SLOSComputeGraph` are the dominant cost centers for strong simulation.

The main bottlenecks are:

1. repeated dense allocations in the SLOS layer-by-layer propagation,
2. large late-stage output tensors that are materialized before downstream reductions,
3. superposition batching logic in `ComputationProcess` that stores large intermediate amplitude tensors before combining them,
4. limited effective use of TorchScript in the true hot path,
5. an existing but currently unused output aggregation mechanism (`output_map_func`).

The best near-term optimization targets are:

1. reduce peak memory in `compute_ebs_simultaneously(...)` by streaming accumulation,
2. wire `output_map_func` from `ComputationProcess` into `SLOSComputeGraph`,
3. add a chunked / reducer-aware last-stage SLOS path,
4. fix a few concrete implementation issues around device transfer and output mapping.

## Current architecture

### `ComputationProcess`
`ComputationProcess` in `merlin/core/process.py` is responsible for:

- building the unitary tensor through `CircuitConverter`,
- building the simulation backend through `build_slos_distribution_computegraph(...)`,
- dispatching computation through:
  - `compute(...)`,
  - `compute_superposition_state(...)`,
  - `compute_ebs_simultaneously(...)`.

Today, `_setup_computation_graphs()` always builds the SLOS graph with:

- `keep_keys=True`,
- no forwarded `output_map_func`.

This means the simulation backend keeps full output basis metadata and computes the full output space even when a later reduction would make that unnecessary.

### `SLOSComputeGraph`
`SLOSComputeGraph` in `merlin/pcvl_pytorch/slos_torchscript.py`:

- pre-builds sparse transition metadata layer by layer,
- stores vectorized operations as `(sources, destinations, modes)`,
- executes each layer through `layer_compute_vectorized(...)` or `layer_compute_batch(...)`,
- optionally supports output aggregation through `output_map_func`.

The core numerical pattern is:

1. gather source amplitudes,
2. multiply by selected unitary coefficients,
3. `scatter_add_` into a new dense result tensor.

## Main findings

### 1) Repeated dense allocations are the dominant structural cost
At each SLOS layer, the code allocates a fresh dense output tensor sized by the number of reachable states in that layer.

This occurs in:

- `layer_compute_vectorized(...)`,
- `layer_compute_batch(...)`,
- the output mapping helper when `output_map_func` is used.

This pattern is simple and vectorized, but it implies:

- repeated allocations,
- repeated writes of large dense tensors,
- a peak memory footprint dominated by the last large layer.

This is the key reason why the “last large SLOS stage” dominates memory usage.

### 2) `compute_ebs_simultaneously(...)` amplifies the memory peak
The most important high-level memory issue is in `ComputationProcess.compute_ebs_simultaneously(...)`.

The current flow is:

1. find non-zero components of the input superposition,
2. propagate those components by batches through `compute_batch(...)`,
3. store all resulting amplitudes in a large tensor,
4. only at the end, contract with the input coefficients.

This is correct, but memory-heavy, because the intermediate tensor has shape roughly:

`[parameter_batch, num_non_zero_input_components, num_output_states]`

This means MerLin can pay a large memory cost twice:

- once in SLOS itself,
- once again in the superposition assembly layer above SLOS.

This is likely one of the highest-value optimization targets in the whole stack.

### 3) The output aggregation hook already exists but is not used from `ComputationProcess`
`SLOSComputeGraph` already supports an `output_map_func` path and builds mapped output indices.

However, `ComputationProcess._setup_computation_graphs()` does not pass `self.output_map_func` into `build_slos_distribution_computegraph(...)`.

As a result:

- the SLOS backend computes the unreduced output space,
- later code performs reductions on the full result,
- existing infrastructure for output aggregation is left unused.

This is both an architectural gap and a direct optimization opportunity.

### 4) TorchScript is present, but the hottest path is still largely Python-driven
The file is presented as TorchScript-optimized, but in practice the layer execution loop is still orchestrated from Python.

Notably:

- `_create_torchscript_modules()` builds Python lambdas for per-layer execution,
- only the output mapping helper is explicitly wrapped in `@jit.script`,
- the main loop over layers remains a Python loop over captured closures.

So the current design is better described as:

- pre-built tensor metadata,
- vectorized PyTorch kernels,
- partial TorchScript usage,

rather than a fully compiled graph execution path.

This is important because it limits how much “torch graph” optimization is actually happening in the hot path.

### 5) There is at least one concrete bug in the graph transfer path
`SLOSComputeGraph.to(...)` contains a problematic line for `target_indices` when `output_map_func` is enabled.

The method currently calls `self.target_indices.to(dtype=dtype, device=self.device)`, but:

- `dtype` is not defined in that scope,
- the result is not assigned back to `self.target_indices`.

This should be fixed independently of any optimization work.

### 6) Some batching assumptions are stronger than they look
`compute_batch(...)` validates only against `input_states[0]` for several checks and derives batch structure from it.

This is fine if all batched input states are guaranteed to belong to the same admissible family, but the implementation is more brittle than it appears and deserves cleanup if batching is expanded further.

## Improvement options

### Option A — Stream accumulation in `compute_ebs_simultaneously(...)`
This is the best immediate optimization.

Instead of storing all per-input-state output amplitudes in a large intermediate tensor and contracting at the end, process each chunk and accumulate directly into the final output tensor.

Current pattern:

1. compute chunk amplitudes,
2. store them in a large global tensor,
3. run a final contraction.

Proposed pattern:

1. compute chunk amplitudes,
2. multiply by the corresponding input coefficients immediately,
3. accumulate directly into the final amplitudes tensor.

Benefits:

- large reduction in peak memory,
- minimal semantic risk,
- no need to redesign SLOS itself.

Risk:

- low.

Recommendation:

- highest priority.

### Option B — Actually use `output_map_func` from `ComputationProcess`
Forward the existing `output_map_func` into `build_slos_distribution_computegraph(...)`.

This would allow the SLOS graph to perform output aggregation natively when the reduction is known upfront.

Benefits:

- reuses existing backend capability,
- likely reduces output dimensionality early,
- creates a clean integration point for grouping/reducer-aware execution.

Limits:

- current `output_map_func` is keyed on output states, not directly on tensor indices,
- some user-facing reducers are index-based and will require an adapter layer.

Risk:

- low to medium.

Recommendation:

- high priority, especially as an enabling step for later optimizations.

### Option C — Chunked last-stage SLOS execution
Add a dedicated execution path that computes only blocks of the final large SLOS stage, applies downstream operations incrementally, and accumulates the result.

This is the most important backend-level memory optimization for the strong simulator itself.

Benefits:

- removes the worst peak allocation,
- enables reducer-aware or transform-aware streaming,
- generalizes beyond simple grouping.

Limits:

- more implementation complexity,
- may not reduce total arithmetic cost,
- can increase launch overhead.

Risk:

- medium.

Recommendation:

- high value, but after Option A and B.

### Option D — Reducer-aware direct accumulation in the last SLOS stage
For simple compatible reducers, make the last large SLOS stage reduction-aware and accumulate directly into reduced buckets.

This is the most aggressive fast path.

Best fit:

- grouping-like reductions,
- identity detector / no sampling / no loss,
- simple exact probability-space reductions.

Benefits:

- lower memory,
- lower write bandwidth,
- possible runtime improvement.

Limits:

- only valid for semantically compatible downstream stages,
- not directly applicable to photon loss because loss acts after squaring in probability space.

Risk:

- medium.

Recommendation:

- implement only after a reducer abstraction is stabilized.

### Option E — Rationalize the “TorchScript” layer
The current design mixes vectorized eager-mode kernels with a small amount of TorchScript.

Possible directions:

1. keep the eager/vectorized design and optimize memory/layout,
2. move the actual hot path to a more modern compilation route (`torch.compile` if appropriate),
3. introduce a custom kernel if this part becomes critical enough.

This should not be the first optimization step, because the largest wins are still structural rather than compiler-related.

Risk:

- medium to high, depending on direction.

Recommendation:

- do later, after the structural bottlenecks are reduced.

## Suggested roadmap

### Phase 1 — Low-risk fixes and infrastructure
1. Fix `SLOSComputeGraph.to(...)`.
2. Forward `output_map_func` from `ComputationProcess`.
3. Clean up batching assumptions and small validation issues.

### Phase 2 — Immediate memory reduction
1. Rewrite `compute_ebs_simultaneously(...)` to accumulate outputs in streaming mode.
2. Avoid keeping unnecessary full intermediate tensors alive longer than needed.

### Phase 3 — Backend-aware reduction
1. Introduce a reducer abstraction for output aggregation.
2. Support reducer-aware chunked execution of the last large SLOS stage.
3. Add direct reduced-accumulation fast paths for simple reducers.

### Phase 4 — Compiler / kernel strategy
1. Reassess whether TorchScript still provides value.
2. Benchmark eager + vectorized vs compiled alternatives.
3. Only then consider deeper compilation work.

## Recommended framing
This work should be treated as three separate tracks:

1. **Performance fixes**
   - streaming accumulation,
   - reducer-aware execution,
   - output-map wiring,
   - memory peak reduction.

2. **Correctness / semantics fixes**
   - loss + post-selection semantics,
   - partial measurement + loss behavior.

3. **Cleanup / robustness fixes**
   - device-transfer bug,
   - batching validation assumptions,
   - unnecessary retained intermediates.

Keeping these tracks separate is important. Otherwise, a useful optimization effort risks getting blocked by broader semantic questions that belong to a different design decision.

## Bottom line
The current MerLin core is already built around a sensible reusable SLOS graph, but the main cost drivers are still:

- large dense layer outputs,
- late materialization of unreduced tensors,
- memory-heavy superposition assembly above the SLOS backend.

The highest-value next steps are:

1. stream accumulation in `compute_ebs_simultaneously(...)`,
2. wire and exploit `output_map_func`,
3. add chunked reducer-aware execution for the last large SLOS stage.

Those changes should deliver the biggest practical improvements before any deeper compiler rewrite is attempted.