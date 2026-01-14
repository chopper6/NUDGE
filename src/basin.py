# basin.py
"""
Transient-control basin simulator + pseudo-attractor basin aggregation.

Relies on:
- net.Net (G.n, G.nodeNums, G.prepare, G.Fmapd)
- other_methods.simulate.step_core, apply_update_rule, get_init_sample
- util.import_cp_or_np, util.get_node_dtype

Attractors are detected by first-repeat (mu, lambda). STG edges are optional.
"""

import itertools
import util
import numpy as np
from other_methods.simulate import apply_setting_to_x0
CUPY, cp = util.import_cp_or_np(try_cupy=1)

# ---- packing helpers (reuse logic from stg.py if you added it) ----
def _pack_rows_bool_to_bytes(x_bool):
    if CUPY:
        xb = cp.asnumpy(x_bool.astype(bool, copy=False))
    else:
        xb = np.asarray(x_bool, dtype=bool)  # <- NumPy fallback (no asnumpy)
    packed = np.packbits(xb, axis=1)
    return [packed[i].tobytes() for i in range(packed.shape[0])]

def _unpack_bytes_to_bool(b, n):
    import numpy as np
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(arr)[:n]
    return bits

def _canonicalize_cycle_bytes(cycle_bytes):
    """Make cycle invariant to rotation."""
    if not cycle_bytes:
        return tuple()
    L = len(cycle_bytes)
    rots = [tuple(cycle_bytes[r:]+cycle_bytes[:r]) for r in range(L)]
    return min(rots)

def _decode_state_bytes_to_dict(b, G):
    """Return {node_name: 0/1} for a packed byte state."""
    bits = _unpack_bytes_to_bool(b, G.n)
    return {name: int(bits[G.nodeNums[name]]) for name in G.nodeNames if not name.startswith(G.not_string)}

def _decode_cycle_to_dicts(cycle_bytes, G):
    """Return list of decoded dicts for a full attractor cycle."""
    return [_decode_state_bytes_to_dict(b, G) for b in cycle_bytes]

# ---- core: simulate until repeat (per-stg), record edges optionally ----

def _simulate_until_repeat(params, G, x0, max_steps=10000, record_edges=False):
    from other_methods.simulate import step_core, apply_update_rule

    num_samples, n = x0.shape
    node_dtype = util.get_node_dtype(params)
    x = cp.array(x0, dtype=node_dtype).copy()

    seen = [dict() for _ in range(num_samples)]
    traj = [[] for _ in range(num_samples)]
    done = cp.zeros(num_samples, dtype=bool)
    edges = [set() for _ in range(num_samples)] if record_edges else None

    # t=0 state
    packed = _pack_rows_bool_to_bytes(x)
    for i in range(num_samples):
        seen[i][packed[i]] = 0
        traj[i].append(packed[i])

    mu = cp.full(num_samples, -1, dtype=cp.int64)
    lam = cp.full(num_samples, -1, dtype=cp.int64)

    for t in range(1, max_steps + 1):
        xn = step_core(x, G, params)
        x, _ = apply_update_rule(params, G, x, xn)

        packed_next = _pack_rows_bool_to_bytes(x)

        if record_edges:
            for i in range(num_samples):
                if not done[i]:
                    edges[i].add((traj[i][-1], packed_next[i]))

        for i in range(num_samples):
            if done[i]:
                continue
            s = packed_next[i]
            if s in seen[i]:
                first = seen[i][s]
                mu[i] = first
                lam[i] = t - first
                traj[i].append(s)
                done[i] = True
            else:
                seen[i][s] = t
                traj[i].append(s)

        if bool(cp.all(done)):
            break

    # extract cycles
    cycles = []
    for i in range(num_samples):
        if mu[i] >= 0:
            m = int(mu[i].item()); L = int(lam[i].item())
            cycles.append(traj[i][m:m+L])
        else:
            cycles.append([])

    out = {
        "mu": (cp.asnumpy(mu) if CUPY else np.asarray(mu)),
        "lam": (cp.asnumpy(lam) if CUPY else np.asarray(lam)),
        "cycles": cycles,
    }
    if record_edges:
        union = set()
        for e in edges:
            union |= e
        out["edges_per_sample"] = edges
        out["edges_union"] = union
    return out

# ---- initial states builder given a transient control on subset ----

def build_initial_states(params, G, control_dict, mode="auto", max_samples=4096, seed=None):
    """
    control_dict: e.g., {"A":1, "B":0}. Applied at t=0 ONLY (transient).
    For the remaining n-k nodes, generate initial states by:
      - 'exhaust' : all combinations (2^(n-k)) if manageable,
      - 'random'  : random sampling,
      - 'auto'    : 'exhaust' if 2^(n-k) <= max_samples else 'random' with max_samples.
    Returns x0 of shape (S, n) boolean.
    """
    rng = cp.random.RandomState(seed) if CUPY else None
    base_nodes = [name for name in G.nodeNames if not name.startswith(G.not_string)]
    n = len(base_nodes)
    assert n == G.n

    free_nodes = [name for name in base_nodes if name not in control_dict]
    k = len(control_dict)

    total = 1 << (n - k)
    if mode == "auto":
        mode = "exhaust" if total <= max_samples else "random"

    idx_map = {name: G.nodeNums[name] for name in base_nodes}

    if mode == "exhaust":
        S = total
        x0 = cp.zeros((S, n), dtype=bool)
        for row, bits in enumerate(itertools.product([0,1], repeat=(n-k))):
            for j, name in enumerate(free_nodes):
                x0[row, idx_map[name]] = bits[j]
    else:
        S = max_samples
        p = 0.5
        if CUPY:
            x0 = cp.random.choice(a=[0,1], size=(S, n), p=[p, 1-p]).astype(bool)
        else:
            import numpy as _np
            x0 = cp.array(_np.random.choice(a=[0,1], size=(S, n), p=[p, 1-p])).astype(bool)

    # 1) apply init first
    merged_init = {**params.get("init", {}), **getattr(G, "params", {}).get("init", {})}
    if merged_init:
        params_tmp = dict(params)
        params_tmp["init"] = merged_init
        x0 = apply_setting_to_x0(params_tmp, G, x0)

    # 2) finally enforce transient control at t=0
    for name, val in control_dict.items():
        x0[:, idx_map[name]] = int(val)

    return x0

# ---- keys for basin aggregation ----

def make_full_attractor_key(cycle_bytes):
    """Canonical key for the *full* attractor."""
    return _canonicalize_cycle_bytes(cycle_bytes)

def make_pseudo_attractor_key(cycle_bytes, G, subset_names):
    """
    Reduce an attractor to the pattern(s) of a subset of nodes across its cycle.
    For a fixed point: single tuple of subset bits.
    For a cycle: tuple of subset-bit tuples, canonicalized by rotation.
    """
    if not cycle_bytes:
        return tuple()
    idx = [G.nodeNums[name] for name in subset_names]
    # project each state in the cycle to the subset bits
    proj = []
    for b in cycle_bytes:
        bits = _unpack_bytes_to_bool(b, G.n)
        proj.append(tuple(int(bits[j]) for j in idx))
    # canonicalize by rotation
    L = len(proj)
    rots = [tuple(proj[r:]+proj[:r]) for r in range(L)]
    return min(rots)

# ---- top-level: run basin under transient control ----

def basin_under_transient_control(
    params, G, control_dict,
    max_steps=10000,
    x0_mode="auto", max_samples=4096, seed=None,
    record_stg=False,
    pseudo_nodes=None
):
    """
    Returns:
      {
        'counts_full': {full_key: count},
        'probs_full':  {full_key: count/total},
        'counts_pseudo': {pseudo_key: count} (if pseudo_nodes),
        'probs_pseudo':  {pseudo_key: count/total} (if pseudo_nodes),
        'mu': array[S], 'lam': array[S],
        'edges_union': set((bytes,bytes))  (if record_stg)
      }
    """
    # Ensure transient-only semantics
    params = dict(params)
    #params['mutations'] = {}
    params.setdefault('update_rule', 'sync')

    # Build initial states set honoring the transient control at t=0
    x0 = build_initial_states(params, G, control_dict, mode=x0_mode, max_samples=max_samples, seed=seed)

    # Simulate until repeat
    out = _simulate_until_repeat(params, G, x0, max_steps=max_steps, record_edges=record_stg)

    # Aggregate basins
    from collections import Counter
    full_counts = Counter()
    pseudo_counts = Counter() if pseudo_nodes else None

    for cyc in out["cycles"]:
        k_full = make_full_attractor_key(cyc)
        full_counts[k_full] += 1
        if pseudo_nodes:
            k_pseudo = make_pseudo_attractor_key(cyc, G, pseudo_nodes)
            pseudo_counts[k_pseudo] += 1

    total = sum(full_counts.values())
    full_probs = {k: v/total for k, v in full_counts.items()}
    pseudo_probs = ({k: v/total for k, v in pseudo_counts.items()} if pseudo_nodes else None)

    result = {
        "counts_full": full_counts,
        "probs_full": full_probs,
        "counts_pseudo": pseudo_counts,
        "probs_pseudo": pseudo_probs,
        "mu": out["mu"],
        "lam": out["lam"],
    }
    if record_stg:
        result["edges_union"] = out["edges_union"]

    # Decode representative states for readability
    decoded_full = {}
    for cyc in out["cycles"]:
        k_full = make_full_attractor_key(cyc)
        if k_full and k_full not in decoded_full:
            decoded_full[k_full] = _decode_cycle_to_dicts(list(k_full), G)

    decoded_pseudo = {}
    if pseudo_nodes:
        for cyc in out["cycles"]:
            k_pseudo = make_pseudo_attractor_key(cyc, G, pseudo_nodes)
            if k_pseudo and k_pseudo not in decoded_pseudo:
                decoded_pseudo[k_pseudo] = [
                    {n: s for n, s in zip(pseudo_nodes, pattern)} for pattern in k_pseudo
                ]

    result["decoded_full"] = decoded_full
    result["decoded_pseudo"] = decoded_pseudo

    return result

def bias_under_transient_control(
    params, G, control_dict,
    time_steps=1000,
    window=100,
    x0_mode="auto",
    max_samples=4096,
    seed=None,
):
    """
    Estimate node-wise 'bias' under a transient control:
      - Build initial states with transient control at t=0 (using build_initial_states),
      - Run the dynamics for `time_steps`,
      - Compute, for each node, the average activity over the last `window`
        time steps and all samples.

    Returns:
      {node_name: bias_float}
    """
    from other_methods.simulate import step_core, apply_update_rule

    # Copy params and ensure an update rule
    params = dict(params)
    params.setdefault('update_rule', 'sync')

    # Build initial states with transient control at t=0
    x0 = build_initial_states(
        params, G, control_dict,
        mode=x0_mode,
        max_samples=max_samples,
        seed=seed
    )

    node_dtype = util.get_node_dtype(params)
    x = cp.array(x0, dtype=node_dtype).copy()
    S, n = x.shape
    assert n == G.n

    # Where to start counting (only last `window` steps)
    start_t = max(1, time_steps - window + 1)

    # sum over samples and time for each node
    sum_last = cp.zeros(n, dtype=cp.float32)

    for t in range(1, time_steps + 1):
        xn = step_core(x, G, params)
        x, _ = apply_update_rule(params, G, x, xn)

        if t >= start_t:
            # sum across samples for this time step
            sum_last += x.sum(axis=0)

    total = (time_steps - start_t + 1) * S
    bias_vec = sum_last / float(total)

    if CUPY:
        bias_vec = cp.asnumpy(bias_vec)

    # Map back to node names (excluding NOT nodes)
    node_bias = {
        name: float(bias_vec[G.nodeNums[name]])
        for name in G.nodeNames
        if not name.startswith(G.not_string)
    }
    return node_bias
