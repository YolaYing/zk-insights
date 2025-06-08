# Libra GPU accelerations
By Yolanda Ying

This article presents a technical deep dive into the GPU acceleration strategies employed in **Libra**, a high-throughput proof system developed by **Polyhedra Network**. The content is based on Simon Lau’s presentation in the *Sumcheck Builders* series, where he detailed Libra’s design choices and the engineering trade-offs involved in achieving performant GPU-based proving.

This material is mainly based on Simon’s presentations, with some of my own interpretations added where the implementation details were not publicly available.

## Overview

This section summarizes key insights from the sharing session on Libra’s GPU acceleration strategy.

### GPU vs. CPU Performance

GPU acceleration becomes beneficial when the circuit size exceeds $2^{12}$. For smaller circuits, CPU execution remains more efficient due to lower overhead.

### Performance Factors

- For circuits larger than $2^{23}$, the system is clearly **memory-bound**, and high-bandwidth GPUs (e.g., A100, H100) are preferred.
- In the range $2^{13} \sim 2^{23}$, performance depends on the field:
  - **BN254** is **compute-bound** due to complex arithmetic.
  - $\mathbb{F}_2$ and M31 are **memory-bound**, dominated by data movement.

### GPU Acceleration Techniques

**Bookkeeping Table Bottleneck**: A key bottleneck in Libra arises from the bookkeeping table, where multiple gates may contribute to the same output index $x$. On GPU, this requires atomic additions to global memory, which are costly. To address this, we can localize all contributions to the same $x$ within a block, allowing accumulation to occur in fast shared memory and avoiding costly global atomics.

**SIMD-style Circuit Batching**: When proving the same circuit with different inputs, Libra applies SIMD-style batching to parallelize execution across instances, following a similar strategy used in BinaryGKR.


## Session Summary
### Execution Model: GPU vs CPU Division of Labor in Libra

| Step                                     | Component        | Executed On | Description                                                                 |
|------------------------------------------|------------------|-------------|-----------------------------------------------------------------------------|
| `Initialize_PhaseOne(f₁, f₃, A_{f₃}, g)` | Preprocessing    | GPU         | Precompute intermediate table $A_{hg}$ for Phase 1                            |
| `SumCheckProduct(...)` (Phase 1)         | Sumcheck rounds  | GPU         | Perform sumcheck over x for $∑_y f₁(g,x,y)·f₃(y)$                             |
| `Initialize_PhaseTwo(f₁, g, u)`          | Preprocessing    | GPU         | Precompute intermediate table $A_{f₁}$ for Phase 2                            |
| `SumCheckProduct(...)` (Phase 2)         | Sumcheck rounds  | GPU         | Perform sumcheck over y for $f₁(g,u,y)·f₂(u)·f₃(y)$                            |
| Fiat–Shamir hashing (SHA256)             | Challenge gen.   | CPU         | Compute challenges $u₁, ..., u_ℓ$ from transcript hash                        |
| 1-layer GKR verifier                     | Verification     | CPU         | Lightweight verification to ensure proof correctness for a single layer     |

- All arithmetic-heavy steps related to multilinear extensions and sumcheck execution are offloaded to the GPU.
- Only the randomness generation (via Fiat–Shamir transform) and a final sanity check are kept on the CPU.

### From Naive Execution to GPU Pipelining

Libra's prover architecture evolves from a naive layer-by-layer execution model toward a highly pipelined GPU implementation, with both sumcheck and GKR stages overlapping across proofs.

#### Naive Execution Model (Baseline)

In the naive version:
- Each proof processes GKR layers sequentially from top to bottom (Layer 0 → Layer 1 → Layer 2...).
- Within each layer, `Prepare Gx`, `Prepare Hy`, polynomial evaluation, and Fiat–Shamir rounds are executed in strict order.
- All GPU kernels are executed serially per layer, with idle time between layers and across proofs.


#### Optimized GPU Pipelining

Libra improves throughput through two kinds of pipelining:

1. **GKR-Level Pipelining**
   - Proofs are scheduled in staggered fashion:
     - While `Proof 0` is evaluating `Layer 1`, `Proof 1` can begin `Layer 0`, and so on.
   - This exploits GPU concurrency across multiple independent proofs.
   - Allows overlapping memory transfers, kernel execution, and challenge communication across layers.

2. **Sumcheck Stage Pipelining**
   - Within a single proof, each round of sumcheck (poly eval + challenge) overlaps with the next.
   - Instead of waiting for full completion of round `i`, round `i+1` is started as soon as partial results are ready.
   - Achieves up to **1300–1400 sumchecks/sec** throughput on BN254 at size $2^{22}$.

#### Summary

- Naive execution: strictly serial within and across layers.
- GPU pipelining: overlapping sumcheck rounds + interleaved proof layer execution.


### Key Optimization (1): Limit Fanout + Shared Mem
Key bottleneck in the GKR sumcheck implementation—accumulating values via atomic updates on the GPU during the lookup table construction phase.

#### Code Example

```cpp
for (long unsigned int i = 0; i < mul.sparse_evals_len; i++) {
    // g(x) += eq(rz, z) * v(y) * coef
    const Gate<F_primitive, 2>& gate = mul.sparse_evals[i];

    uint32_t x = gate.i_ids[0];  // first input wire
    uint32_t y = gate.i_ids[1];  // second input wire
    uint32_t z = gate.o_id;      // output wire

    hg_vals[x] += vals.evals[y] * (gate.coef * eq_evals_at_rz1[z]);  // Atomic accumulation
    gate_exists[x] = true;
}
```

#### Access Pattern Analysis

| Variable | Role             | Access Pattern                     | GPU-Friendliness         |
|----------|------------------|-------------------------------------|---------------------------|
| `x`      | Output index     | Incremental and contiguous, but atomic write (`+=`) | Poor (causes contention and warp serialization) |
| `y`      | Input index      | Some randomness, read-only          | Good (GPU handles read-only randomness well) |
| `z`      | Output index     | Some randomness, read-only          | Good (no conflicts, only lookup) |

- `x` is the accumulation target, leading to atomic read-modify-write per thread.
- Although `x` is incrementally accessed (e.g., 344, 346, 348...), the atomic `+=` operation becomes a performance bottleneck.
- `y` and `z` are mostly used for lookups, not mutated, and are acceptable for parallel access.

#### Lock-Based Accumulation (Public Implementation)

One approach to deal with the atomic write bottleneck on `x` is to use an explicit lock mechanism. Each thread acquires a lock on the target index `x` before updating the corresponding value in `hg_vals`. The general workflow is:

```cpp
// Acquire lock for index x
acquire_lock(d_lock_threadId, x, i);  // Atomic lock (slow but safe)

// Perform update
F old_hg_val = d_hg_vals[x];  // Some random read
F new_hg_val = old_hg_val + val_y * (coef * eq_z);
d_hg_vals[x] = new_hg_val;    // Write
d_gate_exists[x] = true;

// Release lock
release_lock(d_lock_threadId, x);  // Unlock after write
```
- `acquire_lock(...)` and `release_lock(...)` use atomic operations to ensure only one thread writes to `x` at a time.
- The read from `d_hg_vals[x]` involves some randomness, but since only one thread holds the lock, there are no race conditions.
- This lock-based approach avoids using `atomicAdd`, providing correctness guarantees even under concurrent access.
- In practice, it achieves around **3× speedup over the CPU version**.
- However, lock contention introduces significant overhead on GPUs, including warp divergence and thread stalls.
- Therefore, while effective, this is **not the most efficient solution** for large-scale parallel workloads.

#### Shared Memory Optimization with Layout Tuning

To overcome the performance limits of atomic or lock-based accumulation, they apply a **shared memory optimization** combined with **circuit layout tuning**.

Instead of allowing `x` values to be scattered across multiple CUDA blocks (which would require accumulation in global memory and thus locking), they **restructure the circuit** so that all gate contributions to a given `x` value are processed **within the same block**.

This enables:

- Accumulation of `hg_vals[x]` inside fast **CUDA shared memory** (on-chip SRAM), with no locking needed.
- Efficient intra-block reduction and final write-back to global memory.
- Much higher memory access efficiency compared to using atomic operations on global DRAM.

To achieve this, they require:
- Circuit reordering or padding to ensure that `x`-indices are block-local.
- Careful coordination between the **compiler, runtime scheduler, and circuit structure** — a form of software/hardware co-design.

This approach leads to approximately **2× speedup over the lock-based CUDA implementation**, and the core performance gain comes from the memory hierarchy: **SRAM (shared memory) vs DRAM (global memory)**.

### Key Optimization Idea (2): GPU's SIMD
This idea builds on the observation that when generating many proofs over the same circuit structure (but different inputs/witnesses), it is possible to fuse them together into a single SIMD-style execution on the GPU.

#### Circuit-Level SIMD Merging

- In many practical ZK proving systems, the prover repeatedly runs the same circuit with different inputs (e.g., proving multiple Merkle openings, ML inferences, or batch requests).
- Instead of treating each proof separately (as in the CUDA stream baseline), they align all proofs over a shared circuit and process them together in a **circuit-level SIMD** fashion.

#### Performance Impact

- Compared to the naive CUDA-streamed approach (which simply launches many parallel proofs), this SIMD-aware technique reduces kernel overhead, improves data locality, and eliminates redundant scheduling.
- Leads to approximately **30% improvement** over bare CUDA streams.
- Combined with earlier shared memory and lock-free optimizations, this yields:
  - 3× (baseline GPU over CPU)
  - 2× (shared memory over locking)
  - 2.4× (SIMD fusion over naive scheduling)
  - → Cumulative improvement of over **12×** vs CPU baseline


## Key Takeaways
### According to time segmentation of Libra
- In Libra, the precomputed table is the actual bottleneck, rather than the sumcheck itself.
- Within the sumcheck, polynomial evaluation and challenge substitution are the most time-consuming parts, while the Fiat–Shamir part is relatively cheap.
- The data movement between CPU and GPU is not significant under this construction, especially considering the kernel launch overhead.
### CPU vs GPU
- According to Expander’s experiments, GPU is not always better. The circuit size needs to be larger than $2^{12}$ to see real benefits. For BN256 (supported by Ingonyama's ICICI), the maximum speedup of GPU (NVIDIA H100 HBM3 80GB) over CPU (single-threaded 8460Y+) reaches 2829× at a circuit size of $2^{29}$.
### GPU Performance Factors: Experimental Takeaways

The performance of GPU-accelerated sumcheck protocols depends on a combination of factors, including circuit size, field complexity, GPU memory bandwidth, and frequency. Based on experimental results across different GPU architectures (RTX 3090, 4090, H100) and finite fields (M31, M31ext3, BN254), they summarize the following observations:

#### 1. Memory Bandwidth Dominates at Large Scales

- For sufficiently large circuits (typically beyond $2^{23}$), GPU performance becomes **clearly memory-bound**.
- Fields like **M31** and **M31ext3** are simple to compute, so they hit the memory bottleneck earlier due to fast arithmetic and insufficient compute saturation.
- In these cases, GPUs with **high memory bandwidth and capacity** (e.g., H100 with HBM3) provide significant performance advantages.

#### 2. Complex Fields Like BN254 Are More Compute-Bound

- For more complex fields such as **BN254**, the sumcheck prover is initially **compute-bound**, especially in the $2^{13}$ to $2^{23}$ circuit size range.
- These settings benefit more from **higher GPU frequency**, allowing BN254 to scale better on modern GPUs like the 4090 and H100 before hitting memory limits.

#### 3. Circuit Size Determines the Bottleneck Regime

| Circuit Size        | Bottleneck Type      | Hardware Recommendation                                      |
|---------------------|----------------------|---------------------------------------------------------------|
| $< 2^{13}$          | Kernel overhead / PCIe | Prefer CPU execution                                         |
| $2^{13} \sim 2^{23}$| Trade-off region       | Choose based on specific workload; frequency and memory both matter |
| $> 2^{23}$          | Memory-bound          | Use high-bandwidth GPUs (e.g., H100, A100)                    |

#### 5. Summary

> The primary factor affecting GPU performance is memory bandwidth, not frequency.
> However, in compute-intensive workloads like BN254, GPU frequency still plays a meaningful role before memory limits take over.

## Dive Deep
### Shared Memory Optimization

This section outlines a design-level guess on how shared memory optimization might be implemented in the context of Libra-style GKR. The following ideas are just my speculation, based on the observed structure and performance behavior, since the full implementation is not open source.

The reason this approach is possible is that **Libra transforms the GKR recurrence into a fused accumulation form**, where **multiple gates may contribute to the same output index `x`**. Unlike traditional fan-in 2 GKR circuits, Libra deliberately reuses `x` indices to reduce prover time. This leads to high fan-in accumulation patterns, which become a bottleneck on GPU due to atomic writes in global memory.

To optimize this, the key idea is:
> **Make all contributions to the same `x` block-local, so that they can be accumulated in fast shared memory without requiring global atomics.**

#### Design Guess: How Shared Memory Optimization Could Work

1. **Preprocess the gate list** to group gates by output index `x`.
   - For example, sort or bucket gates so that all updates to the same `x` are contiguous in memory.

2. **Map each group of gates (with the same `x`) to the same CUDA block**.
   - Ensure that `x` does not appear across multiple threadblocks, so that no inter-block accumulation is required.

3. **Within each block**, use `__shared__` memory to hold a temporary buffer for `hg_vals[x]`.
   - Each thread reads its gate inputs (`y`, `z`) and contributes a partial value to `hg_vals[x]`.

4. **Use a warp-level or block-level reduction** to accumulate partial values into the shared memory location for `x`.

5. **Once all threads in the block are done**, use a single thread to **write the final value of `x` from shared memory to global memory**.

6. **If some `x` still must appear across blocks**, fall back to a slower global `atomicAdd` path.

#### Why This Works in Libra

- Libra’s recurrence rewrites (e.g., transforming $\sum_{y,z} f(g,x,y,z)\cdot V(y)\cdot V(z)$ into multilinear sumcheck steps) naturally create many-to-one `x` usage.
- This pattern makes shared memory reduction both **safe** (no conflict across `x` in a block) and **worthwhile** (amortizes many updates into one write).
- The performance gain comes from avoiding repeated DRAM atomics and exploiting fast on-chip SRAM instead.

#### Further Considerations

- Requires compiler/runtime coordination to perform gate reordering.

### SIMD Fusion over Shared Circuit

This optimization idea is conceptually similar to the batched input model used in **BinaryGKR**. In both cases, the prover is tasked with generating multiple proofs over the **same circuit structure**, with each instance having a different input (witness). Instead of running them independently, the proofs are fused into a single sumcheck process via a new auxiliary variable.

Specifically:

- A new variable `$s$` is introduced to index which proof the current computation belongs to.
- The multilinear extension is lifted from $V(x)$ to $V(x, s)$, meaning we are now summing over both gate positions and proof indices.
- The sumcheck proceeds in variable order $x \rightarrow s \rightarrow y$, enabling:
  - Memory-coalesced access across gates (`x`)
  - SIMD-style fusion across different proofs (`s`)
  - Final composition of output wires (`y`)


## Questions for Further Exploration
- What are the trade-offs between **Triton-based** and **CUDA-based** implementations in different proving systems?

## Note by Yolanda Ying
Feel free to improve or extend the technical details. Contributions welcome!

## Source
Speaker: Simon Lau (Polyhedra Network)
Topic: New Result on GPU Accelerating GKR
Date: Friday, April 25, 1:00 PM ET
Link: https://sumcheck-builders.polyhedra.network/talks/gpu-gkr-acceleration

## Appendix
### Libra in 1 pages
* Core GKR Formula
    * For a fixed gate $\vec{g}$, Libra rewrites the verifier polynomial recurrence as:
    $$
    \tilde{V}_i(\vec{g}) = \sum_{\color{red}{\vec{u}}, \color{red}{\vec{v}}}
    \left[
      \widetilde{mult}(\vec{g}, \color{red}{\vec{u}}, \color{red}{\vec{v}})
      \cdot \tilde{V}_{i+1}(\color{red}{\vec{u}})
      \cdot \tilde{V}_{i+1}(\color{red}{\vec{v}})
    \right]
    $$
    
* Phase 1: Sum Over $\color{red}{\vec{u}}$
    * Group the expression by factoring out the sum over $\color{red}{\vec{v}}$:
    $$
    \sum_{\vec{u}, \vec{v}} \left[
      \widetilde{mult}(\vec{g}, 
      \color{red}{\vec{u}}, 
      \color{red}{\vec{v}})
      \cdot \tilde{V}_{i+1}(\color{red}{\vec{u}})
      \cdot \tilde{V}_{i+1}(\color{red}{\vec{v}})
    \right]
    =
    \sum_{\color{red}{\vec{u}}} 
    \tilde{V}_{i+1}(\color{red}{\vec{u}})
    \cdot h_{i+1}(\color{red}{\vec{u}})
    $$ where
    $$
    h_{i+1}(\color{red}{\vec{u}}) =
    \sum_{\color{red}{\vec{v}}}
    \widetilde{mult}(\vec{g}, \color{red}{\vec{u}}, \color{red}{\vec{v}})
    \cdot \tilde{V}_{i+1}(\color{red}{\vec{v}})
    $$

    **Steps:**
    - Initialize lookup table for $h_{i+1}(\color{red}{\vec{u}})$ 
        - GPU optimization keyword: **Prepare Gx**
    - Run Thaler'13 sumcheck over $\color{red}{\vec{u}}$

* Phase 2: Sum Over $\color{red}{\vec{v}}$ (with $\color{red}{\vec{u}} = \vec{r}_u$ fixed)
    * After sampling $\vec{r}_u$ from Phase 1, evaluate:
    $$
    \tilde{V}_i(\vec{g}) =
    \sum_{\color{red}{\vec{v}}}
    \left[
      \widetilde{mult}(\vec{g}, \vec{r}_u, \color{red}{\vec{v}})
      \cdot \tilde{V}_{i+1}(\vec{r}_u)
      \cdot \tilde{V}_{i+1}(\color{red}{\vec{v}})
    \right]
    $$

**Steps:**
- Initialize lookup table for $\widetilde{mult}(\vec{g}, \vec{r}_u, \color{red}{\vec{v}})$
    - GPU optimization keyword: **Prepare Hy**
- Run Thaler'13 sumcheck over $\color{red}{\vec{v}}$


Summary

| Phase    | Variable Summed Over | Lookup Table         | Sumcheck Target Function         |
|----------|----------------------|-----------------------|----------------------------------|
| Phase 1  | $\color{red}{\vec{u}}$               | $h_{i+1}(\color{red}{\vec{u}})$     | $\tilde{V}_{i+1}(\color{red}{\vec{u}}) \cdot h_{i+1}(\color{red}{\vec{u}})$ |
| Phase 2  | $\color{red}{\vec{v}}$               | $\widetilde{mult}(\vec{g}, \vec{r}_u, \color{red}{\vec{v}})$ | $\tilde{V}_{i+1}(\color{red}{\vec{v}})$                 |

Reference: Libra(https://eprint.iacr.org/2019/317)

### Supporting Diagrams
![Screenshot 2025-06-07 at 15.21.09_11zon](https://hackmd.io/_uploads/Bk_DAt-7eg.png)
![Screenshot 2025-06-07 at 15.44.37](https://hackmd.io/_uploads/HJq_CKZmeg.png)
![Screenshot 2025-06-07 at 16.35.42](https://hackmd.io/_uploads/SyORRYb7ex.png)
![Screenshot 2025-06-07 at 16.43.41](https://hackmd.io/_uploads/rJ30AFW7ex.png)
