"""
Compare basins between control and non-control simulations.

Functions:
- build_attractor_table(res, G, pseudo_nodes=None)
- compare_basin_overlap_fixed(res_orig, res_ctrl, G, pseudo_nodes=None)
- check_pattern_occupancy_fixed(res_orig, res_ctrl, G, pattern_dict, pseudo_nodes=None)

Each attractor gets a stable integer 'code_name' (decimal from binary state order),
so comparisons stay consistent across runs.
"""

import numpy as np

# ---- basic unpackers ----

def _unpack_bytes_to_bool(b, n):
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(arr)[:n]
    return bits

def _node_order(G):
    names = [nm for nm in G.nodeNames if not nm.startswith(G.not_string)]
    names.sort(key=lambda nm: G.nodeNums[nm])
    return names

# ---- stable code_name encoding ----

def _state_bytes_to_code_int(b, G, names=None):
    names = names or _node_order(G)
    bits = _unpack_bytes_to_bool(b, G.n)
    code = 0
    for nm in names:
        idx = G.nodeNums[nm]
        code |= (int(bits[idx]) << idx)
    return code

def _cycle_bytes_to_code_tuple(cycle_bytes, G, names=None):
    names = names or _node_order(G)
    codes = tuple(_state_bytes_to_code_int(b, G, names) for b in cycle_bytes)
    if not codes:
        return tuple()
    L = len(codes)
    rots = [codes[r:]+codes[:r] for r in range(L)]
    return min(rots)

def _project_state(b, G, subset):
    bits = _unpack_bytes_to_bool(b, G.n)
    return {nm: int(bits[G.nodeNums[nm]]) for nm in subset}

def _full_attractor_decode(cycle_bytes, G):
    names = _node_order(G)
    out = []
    for b in cycle_bytes:
        bits = _unpack_bytes_to_bool(b, G.n)
        out.append({nm: int(bits[G.nodeNums[nm]]) for nm in names})
    return out

# ---- build attractor table ----

def build_attractor_table(res, G, pseudo_nodes=None):
    table = {}
    for k, count in res["counts_full"].items():
        lam = len(k)
        prob = res["probs_full"][k]
        codes = _cycle_bytes_to_code_tuple(list(k), G)
        states = _full_attractor_decode(list(k), G)
        item = {"key": k, "lambda": lam, "count": count, "prob": prob, "codes": codes, "states": states}
        if pseudo_nodes:
            pseudo = [_project_state(b, G, pseudo_nodes) for b in list(k)]
            L = len(pseudo)
            rots = [tuple(pseudo[r:]+pseudo[:r]) for r in range(L)]
            item["pseudo"] = min(rots)
        table[k] = item
    return table

# ---- comparisons ----

def compare_basin_overlap_fixed(res_orig, res_ctrl, G, pseudo_nodes=None):
    T0 = build_attractor_table(res_orig, G, pseudo_nodes)
    T1 = build_attractor_table(res_ctrl, G, pseudo_nodes)

    orig_keys = set(T0.keys())
    ctrl_keys = set(T1.keys())

    occupied = orig_keys & ctrl_keys
    unoccupied = orig_keys - ctrl_keys

    total_orig = sum(T0[k]["count"] for k in orig_keys)
    occ_size   = sum(T0[k]["count"] for k in occupied)
    unocc_size = sum(T0[k]["count"] for k in unoccupied)

    print("🏁 Basin Overlap (by FULL attractor):")
    print(f"  Original attractors: {len(orig_keys)}")
    print(f"  Occupied by control: {len(occupied)} (cover {occ_size}/{total_orig} = {occ_size/total_orig:.3f})")
    print(f"  Unoccupied:          {len(unoccupied)} (cover {unocc_size}/{total_orig} = {unocc_size/total_orig:.3f})")

    if unoccupied:
        print("\nUnoccupied original attractors:")
        for k in sorted(unoccupied, key=lambda kk: -T0[kk]["count"]):
            meta = T0[k]
            print(f"  λ={meta['lambda']}  basin={meta['count']}  prob={meta['prob']:.3f}  codes={meta['codes']}")
            if pseudo_nodes:
                print("   pseudo:", meta["pseudo"])
            for s in meta["states"]:
                print("   ", s)
            print()

def check_pattern_occupancy_fixed(res_orig, res_ctrl, G, pattern_dict, pseudo_nodes=None):
    subset = sorted(pattern_dict.keys(), key=lambda nm: G.nodeNums[nm])
    T0 = build_attractor_table(res_orig, G, pseudo_nodes or subset)
    T1 = build_attractor_table(res_ctrl, G, pseudo_nodes or subset)
    ctrl_keys = set(T1.keys())

    matched = []
    for k, meta in T0.items():
        hit = False
        for b in list(k):
            proj = _project_state(b, G, subset)
            if all(proj[nm] == pattern_dict[nm] for nm in subset):
                hit = True
                break
        if hit:
            matched.append(k)

    total_orig = sum(meta["count"] for meta in T0.values())
    total_match = sum(T0[k]["count"] for k in matched)

    occupied = [k for k in matched if k in ctrl_keys]
    unoccupied = [k for k in matched if k not in ctrl_keys]
    occ_size = sum(T0[k]["count"] for k in occupied)
    unocc_size = sum(T0[k]["count"] for k in unoccupied)

    print(f"\n🎯 Pattern {pattern_dict} (evaluated over FULL attractors):")
    print(f"  Matching original attractors: {len(matched)} (cover {total_match}/{total_orig} = {total_match/total_orig:.3f})")
    print(f"  Occupied under control: {len(occupied)} (cover {occ_size}/{total_orig} = {occ_size/total_orig:.3f})")
    print(f"  Not reached under control: {len(unoccupied)} (cover {unocc_size}/{total_orig} = {unocc_size/total_orig:.3f})")

    if occupied:
        print("\n  Reached (examples):")
        for k in sorted(occupied, key=lambda kk: -T0[kk]["count"]):
            meta = T0[k]
            print(f"   λ={meta['lambda']} basin={meta['count']} prob={meta['prob']:.3f} codes={meta['codes']}")
            if pseudo_nodes:
                print("    pseudo:", meta["pseudo"])
            for s in meta["states"]:
                print("    ", s)
            print()
    
    if unoccupied:
        print("\n  Unreached (examples):")
        for k in sorted(unoccupied, key=lambda kk: -T0[kk]["count"]):
            meta = T0[k]
            print(f"   λ={meta['lambda']} basin={meta['count']} prob={meta['prob']:.3f} codes={meta['codes']}")
            if pseudo_nodes:
                print("    pseudo:", meta["pseudo"])
            for s in meta["states"]:
                print("    ", s)
            print()
