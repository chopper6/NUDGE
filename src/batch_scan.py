#!/usr/bin/env python3
"""
Batch single-controller scan across many .bnet networks.

For each network:
  1) Choose an objective node (random, but deterministic with --seed), excluding '!X'.
  2) Run one baseline simulation (no mutations). Among samples where objective==1,
     infer "diversity markers" = genes that take BOTH 0 and 1 across those samples.
     (Exclude '!X' and exclude the objective itself.)
  3) Enumerate all single-node controllers (gene -> 0/1), excluding '!X' and excluding the objective.
     For each controller:
       - Run mutated simulation -> (avg, x_fin)
       - Keep only controllers where objective==1 for ALL samples in x_fin
       - Compute deleterious_count = number of diversity markers that become pinned (mean ~0 or ~1)
         Normalize by len(diversity_markers) to get deleterious_fraction
       - Run free simulation from x_fin -> (avg2, x_fin2)
       - Compute neo_fraction = fraction of samples with any gene changed (avg vs avg2) beyond eps_change
         (Exclude '!X' and exclude the objective from the comparison.)
  4) Store per-network summary:
       - mean neo_fraction across valid controllers (%)
       - mean normalized deleterious fraction across valid controllers (%)
       - number of valid single controllers

Finally:
  - Save results to a .pkl
  - Make a scatter plot:
      x = mean neo (%), y = mean deleterious (%),
      size/color by number of valid controllers: 1, 2-3, >=4
    Save as .png and .svg

Notes:
  - Compatible with numpy/cupy via util.import_cp_or_np(try_cupy=1)
  - Verbose printing via --verbose
"""

import sys
sys.path.append("src")

import os
import glob
import time
import argparse
import random
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import util
from other_methods import simulate, net

# Use cp (cupy or numpy)
CUPY, cp = util.import_cp_or_np(try_cupy=1)


# ============================================================
# Small helpers
# ============================================================
def is_real_gene(g: str) -> bool:
    return (g is not None) and (not g.startswith("!"))


def base_genes(G):
    return [g for g in G.nodeNames if is_real_gene(g)]


def map_idx(idx_full, ncols, assume_doubled=True):
    """Map G.nodeNums index -> column index of matrices with width ncols."""
    if idx_full is None or ncols <= 0:
        return None
    idx_full = int(idx_full)
    ncols = int(ncols)
    return (idx_full % ncols) if assume_doubled else (idx_full if idx_full < ncols else None)


def aligned_gene_indices(G_left, G_right, mat_left, mat_right, exclude_genes=(), assume_doubled=True):
    """
    Return (genes, idx_left, idx_right) where idx_* are in matrix column space.
    Excludes genes by NAME (exclude_genes).
    """
    exclude_genes = set(exclude_genes)

    nL = int(mat_left.shape[1])
    nR = int(mat_right.shape[1])

    genes = [g for g in base_genes(G_left) if (g in G_right.nodeNums) and (g not in exclude_genes)]

    idxL, idxR, common = [], [], []
    for g in genes:
        i = map_idx(G_left.nodeNums.get(g), nL, assume_doubled=assume_doubled)
        j = map_idx(G_right.nodeNums.get(g), nR, assume_doubled=assume_doubled)
        if i is None or j is None:
            continue
        common.append(g)
        idxL.append(i)
        idxR.append(j)

    return common, cp.asarray(idxL, dtype=cp.int32), cp.asarray(idxR, dtype=cp.int32)


def delta_aligned(mat_left, mat_right, idx_left, idx_right):
    """Compute |mat_left[:, idx_left] - mat_right[:, idx_right]| safely for numpy/cupy."""
    iL = cp.asnumpy(idx_left) if hasattr(cp, "asnumpy") else np.array(idx_left)
    iR = cp.asnumpy(idx_right) if hasattr(cp, "asnumpy") else np.array(idx_right)

    nL = int(mat_left.shape[1])
    nR = int(mat_right.shape[1])

    mask = (iL < nL) & (iR < nR)
    iL = iL[mask].astype(np.int64)
    iR = iR[mask].astype(np.int64)

    if iL.size == 0:
        raise RuntimeError("No aligned indices remain after filtering.")

    return cp.abs(mat_left[:, iL] - mat_right[:, iR])


def neo_sample_fraction_from_delta(delta, eps):
    """
    neo_fraction = fraction of samples that have >=1 changed gene (delta > eps).
    """
    changed = (delta > eps)
    return float(cp.mean(cp.any(changed, axis=1)))


def pinned_fraction_for_markers(x_fin, markers, G, eps=1e-3, assume_doubled=True):
    """
    Deleterious fraction defined as: among the given markers,
    count those whose mean is ~0 or ~1 in x_fin, then normalize by len(markers).
    """
    if markers is None:
        markers = []
    markers = [m for m in markers if is_real_gene(m)]
    denom = len(markers)
    if denom == 0:
        return 0.0  # define as 0 if no markers available

    ncols = int(x_fin.shape[1])
    pinned = 0
    for g in markers:
        idx = map_idx(G.nodeNums.get(g), ncols, assume_doubled=assume_doubled)
        if idx is None:
            continue
        m = float(cp.mean(x_fin[:, idx]))
        if (m < eps) or (m > 1.0 - eps):
            pinned += 1

    return float(pinned) / float(denom)


def infer_diversity_markers_from_baseline(
    params_global,
    network_file,
    obj_gene,
    assume_doubled=True,
    eps_pinned=1e-3,
    verbose=False,
):
    """
    Run ONE baseline simulation (no mutations).
    Filter samples with objective==1.
    Diversity markers = genes with both 0 and 1 across the filtered samples.
    """
    params_base = deepcopy(params_global)
    params_base["network_file"] = network_file
    params_base["mutations"] = {}
    params_base["use_mutations"] = False

    G = net.Net(params_base)
    G.prepare(params_base)

    avg, x_fin = simulate.measure(params_base, G)  # x_fin: (nsamples, ncols)

    obj_idx = map_idx(G.nodeNums.get(obj_gene), int(x_fin.shape[1]), assume_doubled=assume_doubled)
    if obj_idx is None:
        if verbose:
            print(f"  [baseline] objective index missing for {obj_gene}")
        return G, [], avg, x_fin

    # NEW: objective already always satisfied → skip network
    if bool(cp.all(x_fin[:, obj_idx] == 1)):
        if verbose:
            print("  [baseline] objective already satisfied in ALL samples; SKIP network")
        return None

    mask = (x_fin[:, obj_idx] == 1)
    n_ok = int(cp.sum(mask))

    # If n_ok == 0: DO NOT SKIP. Just no diversity markers can be inferred from objective==1 samples.
    if n_ok == 0:
        if verbose:
            print("  [baseline] no samples satisfy objective==1; diversity_markers=[] (continue)")
        return G, [], avg, x_fin

    x_ok = x_fin[mask, :]

    div = []
    ncols = int(x_ok.shape[1])
    for g in base_genes(G):
        if g == obj_gene:
            continue
        idx = map_idx(G.nodeNums.get(g), ncols, assume_doubled=assume_doubled)
        if idx is None:
            continue
        col = x_ok[:, idx]
        vmin = int(cp.min(col))
        vmax = int(cp.max(col))
        if vmin == 0 and vmax == 1:
            div.append(g)

    if verbose:
        print(f"  [baseline] objective==1 samples: {n_ok}; inferred diversity_markers: {len(div)}")

    return G, div, avg, x_fin


def choose_objective_gene(G0, rng: random.Random):
    genes = base_genes(G0)
    if len(genes) == 0:
        raise RuntimeError("No non-'!' genes found in network.")
    return rng.choice(genes)


# ============================================================
# Per-network scan (single-node controllers only)
# ============================================================
def scan_single_controllers_for_network(
    params_file,
    network_file,
    max_order=1,              # kept for safety; we will enforce max_order=1 in batch
    eps_change=1e-2,
    eps_pinned=1e-3,
    assume_doubled=True,
    seed=0,
    verbose=0,
):
    """
    Returns:
      summary dict + per-controller raw arrays
    """
    # ---- Load params ----
    params_base = util.load_yaml(params_file)
    params2 = deepcopy(params_base)
    params2["network_file"] = network_file
    params2["verbose_poke"] = False

    sim_params = simulate.get_sim_params(params2)
    params_global = {**sim_params, **params2}

    # ---- Build net to select objective and candidates ----
    G0 = net.Net(params_global)
    G0.prepare(params_global)

    rng = random.Random(seed)
    obj_gene = choose_objective_gene(G0, rng)

    if verbose:
        print(f"  objective_gene = {obj_gene}")

    # ---- Infer diversity markers from baseline ----
    baseline = infer_diversity_markers_from_baseline(
        params_global=params_global,
        network_file=network_file,
        obj_gene=obj_gene,
        assume_doubled=assume_doubled,
        eps_pinned=eps_pinned,
        verbose=(verbose >= 2),
    )

    if baseline is None:
        # Option B: skip network completely
        return {
            "network_file": network_file,
            "network_name": os.path.basename(network_file),
            "objective_gene": obj_gene,
            "skipped": True,
            "skip_reason": "trivial_objective",
        }

    G_base, diversity_markers, _avg0, _x0 = baseline
    # G_base, diversity_markers, _avg0, _x0 = infer_diversity_markers_from_baseline(
    #     params_global=params_global,
    #     network_file=network_file,
    #     obj_gene=obj_gene,
    #     assume_doubled=assume_doubled,
    #     eps_pinned=eps_pinned,
    #     verbose=(verbose >= 2),
    # )

    # ---- Candidates for control: all real genes except objective ----
    candidates = [g for g in base_genes(G_base) if g != obj_gene]
    n_nodes = len(candidates)
    total_candidates = 2 * n_nodes

    if verbose:
        print(f"  nodes={n_nodes}  single-node candidates={total_candidates}")

    # ---- Enumerate single controllers ----
    valid_controllers = []
    neo_fractions = []
    deleterious_fractions = []  # normalized pinned fraction on inferred diversity markers

    # Force single-controller only
    if max_order != 1 and verbose:
        print("  Note: batch scan enforces single-controller only (order=1).")

    total_tested = 0
    total_valid = 0
    t0 = time.time()

    for gene in candidates:
        for val in (0, 1):
            total_tested += 1
            controller = {gene: val}

            # 1) mutated run
            params_test = deepcopy(params_global)
            params_test["network_file"] = network_file
            params_test["mutations"] = controller
            params_test["use_mutations"] = True

            G_test = net.Net(params_test)
            G_test.prepare(params_test)

            avg, x_fin = simulate.measure(params_test, G_test)

            # validity: require objective == 1 for all samples
            obj_idx = map_idx(G_test.nodeNums.get(obj_gene), int(x_fin.shape[1]), assume_doubled=assume_doubled)
            if obj_idx is None:
                continue
            if not bool(cp.all(x_fin[:, obj_idx] == 1)):
                continue

            total_valid += 1

            # deleterious fraction on x_fin
            dfrac = pinned_fraction_for_markers(
                x_fin=x_fin,
                markers=diversity_markers,
                G=G_test,
                eps=eps_pinned,
                assume_doubled=assume_doubled,
            )

            # 2) free run from x_fin
            params_free = deepcopy(params_global)
            params_free["network_file"] = network_file
            params_free["mutations"] = {}
            params_free["use_mutations"] = False

            G_free = net.Net(params_free)
            G_free.prepare(params_free)

            avg2, _ = simulate.measure(params_free, G_free, x0=x_fin)

            # 3) neo fraction: compare avg vs avg2, exclude objective (and always exclude '!X')
            exclude_genes = ()
            genes_cmp, idx_test, idx_free = aligned_gene_indices(
                G_test, G_free, avg, avg2,
                exclude_genes=exclude_genes,
                assume_doubled=assume_doubled,
            )
            if len(genes_cmp) == 0:
                # no comparable genes after exclusion; define neo as 0
                neo_frac = 0.0
            else:
                delta = delta_aligned(avg, avg2, idx_test, idx_free)
                neo_frac = neo_sample_fraction_from_delta(delta, eps=eps_change)

            valid_controllers.append(controller)
            deleterious_fractions.append(float(dfrac))
            neo_fractions.append(float(neo_frac))

            if verbose >= 2 and (total_tested % max(1, total_candidates // 10) == 0):
                pct = 100.0 * total_tested / total_candidates
                dt = time.time() - t0
                print(
                    f"    tested={total_tested}/{total_candidates} ({pct:.1f}%) "
                    f"valid={total_valid} elapsed={dt:.1f}s"
                )


    # ---- Summaries ----
    if len(neo_fractions) == 0:
        mean_neo = float("nan")
        mean_del = float("nan")
    else:
        mean_neo = float(np.mean(np.asarray(neo_fractions, dtype=float)))
        mean_del = float(np.mean(np.asarray(deleterious_fractions, dtype=float)))

    out = {
        "network_file": network_file,
        "network_name": os.path.basename(network_file),
        "objective_gene": obj_gene,
        "n_nodes_base": len(base_genes(G_base)),
        "n_diversity_markers": int(len(diversity_markers)),
        "n_tested_single_controllers": int(total_tested),
        "n_valid_single_controllers": int(total_valid),
        "mean_neo_fraction": mean_neo,               # in [0,1]
        "mean_deleterious_fraction": mean_del,       # in [0,1]
        "skipped": False,
        "raw": {
            "valid_controllers": valid_controllers,
            "neo_fractions": neo_fractions,
            "deleterious_fractions": deleterious_fractions,
            "diversity_markers": diversity_markers,
        }
    }
    return out


# ============================================================
# Batch runner + plotting
# ============================================================
def size_bucket(n_valid: int) -> str:
    if n_valid <= 0:
        return None        # should not be plotted anyway
    if n_valid == 1:
        return "1"
    if n_valid == 2:
        return "2"
    if n_valid == 3:
        return "3"
    return "4+"


def make_scatter_plot(batch_summaries, out_folder, out_prefix):
    """
    x: mean_neo (%)   y: mean_deleterious (%)   size+color by n_valid bucket
    """
    xs, ys, buckets = [], [], []
    names = []

    for r in batch_summaries:
        if r.get("skipped", False):
            continue
        mn = r["mean_neo_fraction"]
        md = r["mean_deleterious_fraction"]
        if not np.isfinite(mn) or not np.isfinite(md):
            continue
        xs.append(100.0 * mn)
        ys.append(100.0 * md)
        buckets.append(size_bucket(int(r["n_valid_single_controllers"])))
        names.append(r["network_name"])

    # Bucket styles (size + color)
    # (You asked "colors different based on its size" -> fixed categorical colors.)
    bucket_order = ["1", "2", "3", "4+"]

    bucket_size = {
        "1": 40,
        "2": 70,
        "3": 110,
        "4+": 170,
    }

    bucket_color = {
        "1": "tab:blue",
        "2": "tab:orange",
        "3": "tab:green",
        "4+": "tab:red",
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    for b in bucket_order:
        xb = [x for x, bb in zip(xs, buckets) if bb == b]
        yb = [y for y, bb in zip(ys, buckets) if bb == b]
        if len(xb) == 0:
            continue
        ax.scatter(
            xb, yb,
            s=bucket_size[b],
            c=bucket_color[b],
            alpha=0.75,
            edgecolors="black",
            linewidths=0.7,
            label=f"valid controllers: {b}",
            zorder=2,
        )

    ax.set_xlabel("Mean neo fraction across valid controllers (%)", fontsize=12)
    ax.set_ylabel("Mean normalized deleterious fraction across valid controllers (%)", fontsize=12)
    ax.grid(True, linewidth=0.6, alpha=0.4, zorder=1)
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    fig.savefig(f"./{out_folder}/img/{out_prefix}_batch_scatter.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"./{out_folder}/img/{out_prefix}_batch_scatter.svg", bbox_inches="tight")
    plt.close(fig)


def save_pkl(obj, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def list_bnet_files(folder):
    pats = ["*.bnet", "*.BNet", "*.bnet.txt", "*.txt"]
    files = []
    for p in pats:
        files.extend(glob.glob(os.path.join(folder, p)))
    # If user really means "a folder with bnet files", most likely *.bnet
    files = sorted(set(files))
    # Keep only paths that look like bnet by extension OR contain commas in first non-empty line
    out = []
    for fp in files:
        if fp.lower().endswith(".bnet"):
            out.append(fp)
        else:
            # lightweight check
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "," in line:
                            out.append(fp)
                        break
            except Exception:
                pass
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Path to params.yaml")
    ap.add_argument("--bnet_folder", required=True, help="Folder containing .bnet files")
    ap.add_argument("--out_folder", default="output", help="Folder to store the outputs")
    ap.add_argument("--out_prefix", default="batch_warped", help="Prefix for output files")
    ap.add_argument("--eps_change", type=float, default=1e-2, help="Threshold for neo change in avg vs avg2")
    ap.add_argument("--eps_pinned", type=float, default=1e-3, help="Threshold for pinned (deleterious) detection")
    ap.add_argument("--assume_doubled", type=int, default=1, help="1 if nodeNums is 2x vs matrices due to '!X' nodes")
    ap.add_argument("--seed", type=int, default=0, help="Seed for objective node selection (per network)")
    ap.add_argument("--max_networks", type=int, default=0, help="If >0, only process first N networks")
    ap.add_argument("--verbose", type=int, default=1, help="0 silent, 1 per-network, 2 more details, 3 very verbose")
    args = ap.parse_args()

    bnet_files = list_bnet_files(args.bnet_folder)
    if args.max_networks and args.max_networks > 0:
        bnet_files = bnet_files[: args.max_networks]

    if len(bnet_files) == 0:
        raise RuntimeError(f"No .bnet-like files found in: {args.bnet_folder}")

    if args.verbose:
        print(f"Found {len(bnet_files)} networks in {args.bnet_folder}")
        print(f"Backend: {'CuPy' if CUPY else 'NumPy'}")

    batch_summaries = []
    t_all0 = time.time()

    for i, nf in enumerate(bnet_files):
        t0 = time.time()
        if args.verbose:
            print(f"\n[{i+1}/{len(bnet_files)}] network: {os.path.basename(nf)}")

        # IMPORTANT: vary seed per network, but deterministic overall
        seed_i = args.seed + i * 10007

        res = scan_single_controllers_for_network(
            params_file=args.params,
            network_file=nf,
            max_order=1,
            eps_change=args.eps_change,
            eps_pinned=args.eps_pinned,
            assume_doubled=bool(args.assume_doubled),
            seed=seed_i,
            verbose=args.verbose,
        )

        batch_summaries.append(res)


        if args.verbose:
            dt = time.time() - t0
            print(f"  objective={res.get('objective_gene', 'NA')}")

            if res.get("skipped", False):
                print(f"  SKIPPED: {res.get('skip_reason', 'unknown_reason')}   time={dt:.1f}s")
            else:
                mn = res["mean_neo_fraction"]
                md = res["mean_deleterious_fraction"]
                nvalid = res["n_valid_single_controllers"]
                ndm = res["n_diversity_markers"]

                print(f"  diversity_markers={ndm}  valid_single_controllers={nvalid}")

                if np.isfinite(mn) and np.isfinite(md):
                    print(f"  mean neo = {100.0*mn:.2f}%   mean deleterious = {100.0*md:.2f}%   time={dt:.1f}s")
                else:
                    print(f"  mean neo = NaN   mean deleterious = NaN   time={dt:.1f}s")

    # Save PKL (full raw)
    pkl_path = f"./{args.out_folder}/{args.out_prefix}_batch_results.pkl"
    save_pkl(batch_summaries, pkl_path)

    # Scatter plot (summary)
    make_scatter_plot(batch_summaries, args.out_folder, args.out_prefix)

    if args.verbose:
        dt_all = time.time() - t_all0
        print(f"\nSaved: {pkl_path}")
        print(f"Saved: {args.out_prefix}_batch_scatter.png")
        print(f"Saved: {args.out_prefix}_batch_scatter.svg")
        print(f"Total time: {dt_all:.1f}s")


if __name__ == "__main__":
    main()
