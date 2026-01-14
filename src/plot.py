
import csv, sys, ast, os, math
from collections import defaultdict
import util
import numpy as np
from matplotlib import rcParams, cm, pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import networkx as nx

# Styling 
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['axes.linewidth'] = 3.0  
boldness = 'semibold'
#plt.rcParams['font.weight'] = boldness
#rcParams['axes.labelweight'] = boldness
#rcParams['axes.titleweight'] = boldness
_top = cm.get_cmap('Set2', 8)
_bottom = cm.get_cmap('Dark2', 8)
#COLORS = np.vstack((_top(np.linspace(0, 1, 8)), _bottom(np.linspace(0, 1, 8))))
COLORS = ['#56B4E9','#009E73','#E69F00','#D55E00']
FONTSMALL, FONTMED, FONTLARGE = 18, 24, 32
CIRCLE_ORDER = ["C", "A", "B"]

FILETYPE= 'png'

BASE_DIR = './output/'
BASE_STR = 'large_12max_noTimeout_100maxRecEns8_MF100_longSimsx1000'
OUTPUT_DIR = os.path.join(BASE_DIR, 'img')

CSVS = {
    'IBMFA':       os.path.join(BASE_DIR, BASE_STR + '_MEAN-FIELD_RESULTS.csv'),
    'NUDGE':        os.path.join(BASE_DIR, BASE_STR + '_POKE_RESULTS.csv'),
    'LDOI':        os.path.join(BASE_DIR, BASE_STR + '_LDOI_RESULTS.csv'),
    'MC':          os.path.join(BASE_DIR, BASE_STR + '_MONTE_CARLO_RESULTS.csv'),
}

# Filters
USE_SUBSET = False
SUBSET_DIR = './models/metaAnalysis_10IO/nonOscil_z1000_01thresh/'
ERR_THRESH = 0
CONSTRAIN_BY_ERROR = True
ONLY_SOLO_CONTROLLERS = False

# ============ VIOLIN PLOTS (Error/Minutes) ============
def plot_error_results(csvs, output_dir, run_title):
    data = load_error_data(csvs)
    violins(data, 'Error', output_dir, run_title)
    violins(data, 'Minutes', output_dir, run_title)

def load_error_data(csvs):
    data = {}
    if USE_SUBSET:
        valid_networks = [f for f in os.listdir(SUBSET_DIR) if os.path.isfile(os.path.join(SUBSET_DIR, f))]
    for k, path in csvs.items():
        if path is None:
            continue
        data[k] = {'Error': [], 'Minutes': []}
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                network_name = row['Network'].split('/')[-1]
                if (not USE_SUBSET) or (network_name in valid_networks):
                    if row['Errors'] != 'NA':
                        errs = np.array(ast.literal_eval(row['Errors']), float)
                        data[k]['Error'].append(float(np.mean(errs)))
                    data[k]['Minutes'].append(float(row['Minutes']))
    return data

def violins(data, key, output_dir, run_title, swap_keys=False):
    if swap_keys:
        methods = list(data[key].keys())
        features = [data[key][k] for k in methods]
    else:
        methods = list(data.keys())
        features = [data[k][key] for k in methods]
    labels = [m if m != 'NUDGE' else 'NUDGE ' for m in methods] # renaming our method over time
    logy=False
    if key == 'Error':
        ylabel = 'Error'
        plt.ylim([0, 1])
    elif key == 'Minutes':
        ylabel = 'Time (minutes)'
        logy=True
    elif key in 'fraction_exact_found':
        ylabel = 'Single Node Controllers Found'
    elif key in 'fraction_found':
        ylabel = 'Single Node Controllers Found (SUBSETS)'
    elif key == 'avg_size':
        ylabel = 'Solo Controller Size'
    else:
        ylabel = key
    print('\n\n',"plotting key",key)
    std_violin(features, labels, ylabel, alpha=.8, gridalpha=.4, colors=COLORS, minorgrid=True, logy=logy)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "{}_{}_{}.{}".format(run_title, util.timestamp(), key, FILETYPE))
    plt.savefig(outpath)
    plt.clf()
    plt.close()

def std_violin(features, labels, ylabel, minorgrid=False, alpha=1, gridalpha=.4, colors=('#4287f5',), logy=False):
    # features = list of lists [[]], each list is one violin
    # labels = xtick labels
    plt.figure(figsize=(9, 8)) #2 * len(features)
    plt.grid(alpha=gridalpha, which='major')
    if minorgrid:
        plt.grid(alpha=gridalpha/2, which='minor')
        plt.minorticks_on()

    for z, f in enumerate(features):
        parts = plt.violinplot(f, positions=[z+1], widths=0.9, showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[z % len(colors)])
            pc.set_edgecolor('black')
            pc.set_linewidths(2) 
            pc.set_alpha(1)
    stats_fn = lin_stats
    means, lows, q1s, meds, q3s, highs = [], [], [], [], [], []
    for f in features:
        m, (lo, q1, med, q3, hi) = stats_fn(f)
        means.append(m); lows.append(lo); q1s.append(q1); meds.append(med); q3s.append(q3); highs.append(hi)
    inds = np.arange(1, len(features) + 1)
    print('\t',labels,'have means',means)
    plt.scatter(inds, means, marker='o', color='red', s=400, zorder=3)
    plt.vlines(inds, q1s, q3s, color='k', lw=16)
    plt.vlines(inds, lows, highs, color='k', lw=4)
    plt.xticks(inds, labels, fontsize=FONTMED)
    plt.yticks(fontsize=FONTMED)
    plt.ylabel(ylabel, fontsize=FONTLARGE, labelpad=15)
    plt.tick_params(axis='both', width=2.5, length=6, direction='out')

    if logy:
        plt.yscale('log')
    elif len(features) >= 3:
        p1 = ztest_pvalue(features[0], features[1]) # IBMFA vs NUDGE
        p2 = ztest_pvalue(features[2], features[1]) # LDOI vs NUDGE
        print("\tp1=",p1,'\n\tp2=',p2)
        add_pval(p1, (0,1))
        add_pval(p2, (2,1), second=True)
    plt.tight_layout()

def log_stats(x):
    x = np.asarray(x, float)
    x = x[x > 0]
    if x.size == 0:
        return np.nan, [np.nan] * 5
    lx = np.log10(x)
    return 10 ** lx.mean(), [10 ** np.percentile(lx, p) for p in (10, 25, 50, 75, 90)]

def lin_stats(x):
    x = np.asarray(x, float)
    return (float(np.mean(x)) if len(x) else np.nan), [np.percentile(x, p) for p in (10, 25, 50, 75, 90)]

def ztest_pvalue(arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    mean1 = arr1.mean()
    mean2 = arr2.mean()
    var1 = arr1.var(ddof=1)
    var2 = arr2.var(ddof=1)

    # std err
    se = np.sqrt(var1/len(arr1) + var2/len(arr2))

    z = (mean1 - mean2) / se

    # normal CDF
    p = 2 * (1 - 0.5 * (1 + math.erf(np.abs(z) / np.sqrt(2))))

    return p

def add_pval(pval, xindices, y_offset=0.1, second=False):
    ax = plt.gca()

    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yticklabels = ["0", "0.2", "0.4", "0.6", "0.8", "1.0"]

    x1, x2 = xindices
    if x1 > x2:
        x1, x2 = x2, x1

    if pval < 1e-3:
        s = "***"
    elif pval < 1e-2:
        s = "**"
    elif pval < 0.05:
        s = "*"
    else:
        s = "ns"

    x1 += 1
    x2 += 1

    ymin, ymax = ax.get_ylim()
    if second:
        down = .15 
    else:
        down = .07
    y = ymax + (ymax - ymin)*y_offset - down
    yline = ymax + (ymax - ymin)*(y_offset/1.2)  - down
    new_ymax = ymax + (ymax - ymin)*(y_offset*2) - down

    ax.plot([x1, x1, x2, x2],
            [yline, y, y, yline],
            color='black', linewidth=2)

    ax.text((x1 + x2)/2, y, s,
            ha='center', va='bottom', fontsize=24)

    ax.set_ylim(ymin, new_ymax)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

import numpy as np


# ============ VENN (NUDGE vs union of LDOI ∪ IBMFA) + PRINT UNIQUES ============
def proportional_venn(A, B, AB, output_png_prefix, labels=("A", "B"), color_indices=None):
    """Draw a proportional two-set Venn diagram.
    A=unique to labels[0], B=unique to labels[1], AB=overlap.
    """
    A = int(round(A*100))
    B = int(round(B*100))
    AB = int(round(AB*100))
    assert(99 <= A + B + AB <= 101) # check that essentially sum to 100, with tolerance for rounding
    AB = 100-A-B

    sizeA = A + AB
    sizeB = B + AB
    rA = (np.sqrt(sizeA / np.pi) if sizeA > 0 else 0.0)
    rB = (np.sqrt(sizeB / np.pi) if sizeB > 0 else 0.0)

    def circle_intersection_area(d, rA, rB):
        if d >= rA + rB:
            return 0.0
        if d <= abs(rA - rB):
            return np.pi * min(rA, rB) ** 2
        part1 = rA**2 * np.arccos((d**2 + rA**2 - rB**2) / (2.0 * d * rA))
        part2 = rB**2 * np.arccos((d**2 + rB**2 - rA**2) / (2.0 * d * rB))
        part3 = 0.5 * np.sqrt((-d+rA+rB)*(d+rA-rB)*(d-rA+rB)*(d+rA+rB))
        return part1 + part2 - part3

    if sizeA == 0 and sizeB == 0:
        d = 0.0
    else:
        target_overlap = AB
        if rA and rB:
            d_values = np.linspace(abs(rA-rB)+1e-6, rA+rB-1e-6, 500)
            areas = [circle_intersection_area(d, rA, rB) for d in d_values]
            d = d_values[int(np.argmin(np.abs(np.array(areas) - target_overlap)))]
        else:
            d = 0.0

    fig, ax = plt.subplots()
    if color_indices is None:
        colors = ('red', 'blue')
    else:
        colors = (COLORS[color_indices[0]], COLORS[color_indices[1]])

    circleB = plt.Circle((d, 0), rB, alpha=0.5, color=colors[1])
    circleA = plt.Circle((0, 0), rA, alpha=0.5, color=colors[0])
    ax.add_artist(circleB)
    ax.add_artist(circleA)

    maxR = max(rA, rB, 1.0)
    ax.text(-3, maxR + 0.15*maxR, labels[0], ha="center", va="bottom", fontsize=20)
    ax.text(d+3, maxR + 0.15*maxR, labels[1], ha="center", va="bottom", fontsize=20)

    ax.text(-rA*0.5-1, 0, str(A)+'%', ha="right", va="center", fontsize=FONTSMALL)
    ax.text(d+4, 0, str(B)+'%', ha="center", va="center", fontsize=FONTSMALL)
    ax.text(d/2-1, 0, str(AB)+'%', ha="center", va="center", fontsize=FONTSMALL)

    ax.set_xlim(-maxR, d+maxR)
    ax.set_ylim(-maxR, maxR*1.6)
    ax.set_aspect('equal', 'box')
    ax.axis("off")
    if not os.path.isdir(os.path.dirname(output_png_prefix)):
        os.makedirs(os.path.dirname(output_png_prefix), exist_ok=True)
    outpath = "{}_{}_coverage_{}_vs_{}.{}".format(output_png_prefix, util.timestamp(), labels[0], labels[1], FILETYPE)
    plt.savefig(outpath)
    plt.clf()
    plt.close()

def found_it(a, B):
    """Return True if list/set a is a subset of any element in B."""
    for b in B:
        if b == 'NA':
            continue
        try:
            if set(a).issubset(b) or b == ['1'] or b == {'1': 1}:
                return True
        except TypeError:
            pass
    return False

def load_controllers_and_errors(csvs):
    controllers, errors = {}, {}
    for k, path in csvs.items():
        if path is None:
            continue
        method_controllers, method_errors = defaultdict(dict), defaultdict(dict)
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                network = row['Network']
                target = row['Target']
                controllers_value = ast.literal_eval(row['Controllers'])
                method_controllers[network][target] = controllers_value
                if row['Errors'] == 'NA':
                    method_errors[network][target] = 'NA'
                else:
                    method_errors[network][target] = ast.literal_eval(row['Errors'])
        controllers[k], errors[k] = dict(method_controllers), dict(method_errors)
    return controllers, errors

def network_and_target_in_all(controllers, *methods):
    for method in methods:
        if method not in controllers:
            return False
    common_pairs = None
    for m in methods:
        pairs = {(n, t) for n in controllers[m] for t in controllers[m][n]}
        common_pairs = pairs if common_pairs is None else (common_pairs & pairs)
    return common_pairs

def trim_by_error(controllers, errors, method, network, target):
    if (controllers[method][network][target] == [['0']]):
        return [['0']]
    elif (errors[method][network][target] == 'NA'):
        return []
    if not CONSTRAIN_BY_ERROR:
        return controllers[method][network][target]
    return [a for a, b in zip(controllers[method][network][target], errors[method][network][target]) if b <= ERR_THRESH]

def min_size_sets(S, force_min_size=None):
    """
    Return the set of minimal-size controllers as frozensets.

    Edge cases:
      - Exclude ['0'] entirely (no controllers suggested)
      - Treat ['1'] as size 0 (no control needed)
    """
    cleaned = [s for s in S if s != ['0']]

    if not cleaned:
        return set()

    def set_size(s):
        if s == [] or s == ['1']:
            return 0
        if isinstance(s, dict):
            if s == {} or s == {'1': 1}:
                return 0
            return len(s)
        try:
            return len(s)
        except TypeError:
            return 1

    if force_min_size is None:
        target = min(set_size(s) for s in cleaned)
    else:
        target = force_min_size

    def to_fset(s):
        if isinstance(s, dict):
            return frozenset(s.keys())
        return frozenset(s)

    return {to_fset(s) for s in cleaned if set_size(s) == target}


def venn_method_vs_poke(controllers, errors, title, output_dir, method, color_indices=(0, 1), only_solo_controllers=ONLY_SOLO_CONTROLLERS):

    counts_A_only = 0  # unique to `method`
    counts_B_only = 0  # unique to NUDGE
    counts_both   = 0  # overlap

    common_pairs = network_and_target_in_all(controllers, method, 'NUDGE')
    if not common_pairs:
        print("No common (network,target) across {} and NUDGE.".format(method))
        return

    def set_size(s):
        if s == [] or s == ['1']:
            return 0
        if isinstance(s, dict):
            if s == {} or s == {'1': 1}:
                return 0
            return len(s)
        try:
            return len(s)
        except TypeError:
            return 1

    num_counted = 0
    for (network, target) in sorted(common_pairs):
        A_sets = trim_by_error(controllers, errors, method, network, target)
        B_sets = trim_by_error(controllers, errors, 'NUDGE', network, target)

        if only_solo_controllers:
            A_sets = return_only_solo_controllers(A_sets)

            #print("NUDGE only:",onlyA,"\non network", network,'with target',target)
            B_sets = return_only_solo_controllers(B_sets)


        A_sizes = [set_size(s) for s in A_sets if s != ['0']]
        B_sizes = [set_size(s) for s in B_sets if s != ['0']]
        if not A_sizes and not B_sizes:
            continue
        if not A_sizes:
            global_min = min(B_sizes)
        elif not B_sizes:
            global_min = min(A_sizes)
        else:
            global_min = min(min(A_sizes), min(B_sizes))

        A = min_size_sets(A_sets, force_min_size=global_min)
        B = min_size_sets(B_sets, force_min_size=global_min)

        alll  = A | B
        both  = A & B
        onlyA = A - B
        onlyB = B - A

        if len(alll)>0 and global_min>0: # find some controllers and best solution requires at least 1 controller
            counts_A_only += len(onlyA)/len(alll)
            counts_B_only += len(onlyB)/len(alll)
            counts_both   += len(both)/len(alll)

            num_counted += 1

    counts_A_only /= num_counted
    counts_B_only /= num_counted 
    counts_both /= num_counted

    out_prefix = os.path.join(output_dir, title)
    proportional_venn(counts_A_only, counts_B_only, counts_both,
                      output_png_prefix=out_prefix,
                      labels=(method, 'NUDGE'),
                      color_indices=color_indices)



def return_only_solo_controllers(controllers):
    solos = [lst for lst in controllers if len(lst) < 2]
    if len(solos)==1 and (solos[0] ==['0']): # err on 0 not measured
         return []
    else:
        return solos

def print_poke_uniques(controllers, errors):
    """
    Print MINIMAL controllers that NUDGE finds which BOTH LDOI and IBMFA do not find.
    Ignores Monte-Carlo entirely.
    """
    common_pairs = network_and_target_in_all(controllers, 'NUDGE', 'LDOI', 'IBMFA')
    if not common_pairs:
        print("No common (network,target) across NUDGE, LDOI, IBMFA.")
        return

    def set_size(s):
        empty_size=0
        if s == [] or s == ['1']:
            return empty_size
        if isinstance(s, dict):
            if s == {} or s == {'1': 1}:
                return empty_size
            return len(s)
        try:
            return len(s)
        except TypeError:
            return 1

    total_uniques = 0

    for (network, target) in sorted(common_pairs):
        poke_all = trim_by_error(controllers, errors, 'NUDGE',       network, target)
        ldoi_all = trim_by_error(controllers, errors, 'LDOI',       network, target)
        mean_all = trim_by_error(controllers, errors, 'IBMFA', network, target)

        all_sets = [s for s in (poke_all + ldoi_all + mean_all) if s != ['0']]
        if not all_sets:
            continue
        global_min = min(set_size(s) for s in all_sets)

        poke_min   = min_size_sets(poke_all,           force_min_size=global_min)
        others_min = min_size_sets(ldoi_all + mean_all, force_min_size=global_min)

        uniques = sorted(
            list(poke_min - others_min),
            key=lambda s: (len(s), sorted(s))
        )

        if uniques:
            total_uniques += len(uniques)
            print("[NUDGE unique vs BOTH LDOI and IBMFA] Network={} Target={}:".format(network, target))
            for u in uniques:
                print("   ", sorted(list(u)))

    if total_uniques == 0:
        print("No NUDGE-unique controllers against both LDOI and IBMFA.")

def mechanism_graph(params, G):
    cfg = {
        "node_size": params.get("node_size", 1000),
        "arrow": params.get("arrow_head_size", 20),
        "rad": params.get("rad", 0.22),
        "higher": params.get("higher_order_edge_color", "blue"),
        "ct_edge": params.get("controller_target_edge_color", "red"),
        "self_fill": params.get("self_driven_fill_color", "pink"),
        "font": params.get("font_size", 10),
        "bbox": params.get("label_box", dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)),
    }
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G)

    # (unchanged) edge coloring logic if you still draw edges elsewhere
    for u, v, d in G.edges(data=True):
        G[u][v]["color"] = cfg["higher"] if d.get("value", 0) > 1 else "black"

    draw_nodes(G, pos, cfg)

    oneway, bidir, selfloops = partition_edges(G)
    margin = cfg["node_size"] ** 0.5
    draw_edges(G, pos, oneway, 0.0, cfg["arrow"], margin+1, margin+1)
    draw_bidir(G, pos, bidir, cfg["rad"], cfg["arrow"], margin+1, margin+1)
    draw_edges(G, shift_pos(pos), selfloops, 0.75, cfg["arrow"], margin, margin)

    nx.draw_networkx_labels(G, pos, font_color="black", font_size=cfg["font"], font_family="serif")
    draw_edge_detail_labels(G, pos, bidir, cfg["rad"], cfg["font"], cfg["bbox"])
    plt.axis("off"); plt.tight_layout(); plt.show()


def draw_nodes(G, pos, cfg):
    # First draw fills for self-driven nodes (so outline renders above)
    self_driven_nodes = [n for n, d in G.nodes(data=True) if d.get("self_driven", False)]
    if self_driven_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=self_driven_nodes,
            node_size=4000,
            node_color=cfg["self_fill"],
            edgecolors="none",
            linewidths=2
        )

    # Then draw outlines for all nodes based on controller/target flags
    controller_or_target = []
    regular = []
    for n, d in G.nodes(data=True):
        is_ct = d.get("controller", False) or d.get("target", False)
        if is_ct:
            controller_or_target.append(n)
        else:
            regular.append(n)

    if controller_or_target:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=controller_or_target,
            node_size=4000,
            node_color="none",
            edgecolors=cfg["ct_edge"],
            linewidths=4
        )

    if regular:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=regular,
            node_size=4000,
            node_color="none",
            edgecolors="black",
            linewidths=2
        )

def partition_edges(G):
    all_other = [(u,v) for u,v in G.edges() if u != v]
    selfloops = [(u,v) for u,v in G.edges() if u == v]
    bidir = {tuple(sorted((u,v))) for u,v in all_other if G.has_edge(v,u)}
    oneway = [(u,v) for u,v in all_other if tuple(sorted((u,v))) not in bidir]
    return oneway, list(bidir), selfloops


def draw_edges(G, pos, edgelist, rad, arrow, ms, mt):
    if not edgelist:
        return
    colors = [G[u][v].get("color",(0,0,0,1)) for u,v in edgelist]
    nx.draw_networkx_edges(
        G, pos, edgelist=edgelist, edge_color=colors, width=4, arrows=True,
        arrowstyle="-|>", arrowsize=arrow, connectionstyle=f"arc3,rad={rad}",
        min_source_margin=ms, min_target_margin=mt
    )


def draw_bidir(G, pos, bidir_pairs, rad, arrow, ms, mt):
    for a, b in bidir_pairs:
        if G.has_edge(a,b):
            draw_edges(G, pos, [(a,b)], rad, arrow, ms, mt)
        if G.has_edge(b,a):
            draw_edges(G, pos, [(b,a)], rad, arrow, ms, mt)


def shift_pos(pos):
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    dy = 0.03 * max((max(xs)-min(xs)), (max(ys)-min(ys))) or 1.0
    return {n:(x, y+dy) for n,(x,y) in pos.items()}


def draw_edge_detail_labels(G, pos, bidir_pairs, rad, font, bbox):
    ax = plt.gca()
    has_text = lambda t: (t is not None) and (str(t).strip() != "")

    straight = {(u, v): d.get("detail")
                for u, v, d in G.edges(data=True)
                if u != v and tuple(sorted((u, v))) not in bidir_pairs and has_text(d.get("detail"))}
    if straight:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=straight, label_pos=0.5,
                                     rotate=False, font_size=font, bbox=bbox)

    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    scale = max((max(xs) - min(xs)), (max(ys) - min(ys))) or 1.0
    dy = 0.03 * scale

    loops = {(u, v): d.get("detail")
             for u, v, d in G.edges(data=True)
             if u == v and has_text(d.get("detail"))}
    if loops:
        pos_loops_labels = {n: (x, y + dy * 2.4) for n, (x, y) in pos.items()}
        nx.draw_networkx_edge_labels(G, pos_loops_labels, edge_labels=loops, label_pos=0.5,
                                     rotate=False, font_size=font, bbox=bbox)

    k = 0.32
    side = -1 if rad >= 0 else 1   # arc3: positive rad bends to the "left" of u->v
    rabs = abs(rad)

    for a, b in bidir_pairs:
        if not (G.has_edge(a, b) or G.has_edge(b, a)):
            continue
        xa, ya = pos[a]; xb, yb = pos[b]
        mx, my = (xa + xb) / 2.0, (ya + yb) / 2.0
        dx, dy2 = xb - xa, yb - ya
        L = (dx*dx + dy2*dy2) ** 0.5 or 1.0
        nxp, nyp = -dy2 / L, dx / L
        off = k * max(rabs, 1e-3) * L

        if G.has_edge(a, b):
            t_ab = G[a][b].get("detail")
            if has_text(t_ab):
                ax.text(mx + side*nxp*off, my + side*nyp*off, str(t_ab),
                        fontsize=font, bbox=bbox, ha="center", va="center")

        if G.has_edge(b, a):
            t_ba = G[b][a].get("detail")
            if has_text(t_ba):
                # Reverse direction: left normal flips sign automatically
                ax.text(mx - side*nxp*off, my - side*nyp*off, str(t_ba),
                        fontsize=font, bbox=bbox, ha="center", va="center")


def nx_plot(G): 

    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    in_degree_sequence = sorted((d for n, d in G.in_degree()), reverse=True)
    out_degree_sequence = sorted((d for n, d in G.out_degree()), reverse=True)
    

    xboth, deg_count = np.unique(degree_sequence, return_counts=True)
    xin, in_deg_count = np.unique(in_degree_sequence, return_counts=True)
    xout, out_deg_count = np.unique(out_degree_sequence, return_counts=True) 

    fig = plt.figure(figsize=(8, 6))
    plt.minorticks_on()
    plt.grid(alpha=.4, which='both')
    width = .3
    size=100

    alpha=.8
    plt.scatter(xin-width, in_deg_count,s=size, alpha=alpha,label='in-degree',color=COLORS[0], linewidth=1, edgecolor='black')
    plt.scatter(xout, out_deg_count,s=size, alpha=alpha,label='out-degree',color=COLORS[1], linewidth=1, edgecolor='black')
    plt.scatter(xboth+width, deg_count,s=size, alpha=alpha,label='both',color=COLORS[2], linewidth=1, edgecolor='black')
    
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("# of Nodes")
    plt.legend()

    fig.tight_layout()
    plt.show()
    plt.clf()
    plt.close()



########################################################################################

def plot_MCcoverageTest(data, methods, title):
    for k in data.keys():
        if k not in ['uncontrollable_found','controllable_found','trivially_controllable_found']:
            violins(data, k, OUTPUT_DIR, title +'_'+k, swap_keys=True) 

def MCcoverageTest():
    controllers, errors = load_controllers_and_errors(CSVS)

    methods = list(CSVS.keys())
    methods.remove('MC')
    fraction_found, avg_size, fraction_exact_found, controllable_found, uncontrollable_found, trivially_controllable_found, controllability = [{k:[] for k in methods} for _ in range(7)]
    for k in methods:
        for network in controllers['MC'].keys():
            this_network_found = 0
            this_network_size = 0
            this_network_exact_found = 0
            num_found_checked, num_size_checked = 0, 0

            this_cf, this_uf, this_tf = 0,0,0
            num_c, num_u, num_t = 0,0,0
            this_cb, num_cb = 0,0

            for target in controllers['MC'][network].keys():
                if network in controllers[k].keys() and target in controllers[k][network].keys():
                    mc_cnts = trim_by_error(controllers, errors, 'MC', network, target)
                    other_cnts = trim_by_error(controllers, errors, k, network, target)
                    #if ('23935937' in network) and (target == '!NOTCH_SIGNAL') and (k=='NUDGE'):
                    #    print("\tMc cnts=",mc_cnts,'\n\t',k,'=',other_cnts)
                    if (len(mc_cnts) > 0) and (mc_cnts[0] not in [['0'],['1']]): 
                        num_found_checked += 1
                        found_for_target = 0
                        avg_size_for_target = 0
                        exact_found_for_target = 0
                        for cnt in mc_cnts:

                            if has_controller(other_cnts, cnt):
                                found_for_target += 1 
                                this_avg_size = calc_avg_size(other_cnts, cnt)
                                avg_size_for_target += this_avg_size
                                if this_avg_size == 1:
                                    exact_found_for_target += 1
                        this_network_found += found_for_target / len(mc_cnts)
                        this_network_exact_found += exact_found_for_target/ len(mc_cnts)
                        if found_for_target > 0:
                            this_network_size += avg_size_for_target/found_for_target
                            num_size_checked += 1

                    # whether or not matches MC is the error in controllability case
                    mc_cnts = controllers['MC'][network][target]
                    other_cnts = controllers[k][network][target]
                    if (len(mc_cnts) > 0):
                        num_cb += 1
                        if mc_cnts[0] == ['0']: # uncontrollable
                            num_u += 1
                            if (len(other_cnts) > 0) and  (other_cnts[0] == ['0']):
                                this_uf += 1
                                this_cb += 1
                        else: # controllable
                            num_c += 1
                            if (len(other_cnts) > 0) and  (other_cnts[0] != ['0']):
                                this_cf += 1
                                this_cb += 1

                        if mc_cnts[0] == ['1']: # trivially controllable
                            num_t += 1
                            if (len(other_cnts) > 0) and  (other_cnts[0] == ['1']):
                                this_tf += 1
            
            if num_cb > 0:
                controllability[k] += [this_cb / num_cb]
            if num_u > 0:
                uncontrollable_found[k] += [this_uf / num_u]
            if num_c > 0:
                controllable_found[k] += [this_cf / num_c]
            if num_t > 0:
                trivially_controllable_found[k] += [this_tf / num_t]

            if num_found_checked > 0:
                fraction_found[k] += [this_network_found/num_found_checked]
                fraction_exact_found[k] += [this_network_exact_found/num_found_checked]
            if num_size_checked > 0:
                avg_size[k] += [this_network_size/num_size_checked]

    data = {
        'fraction_found': fraction_found,
        'avg_size': avg_size,
        'fraction_exact_found': fraction_exact_found,
        'uncontrollable_found': uncontrollable_found,
        'controllable_found': controllable_found,
        'trivially_controllable_found': trivially_controllable_found,
        'controllability':controllability,
    }
    return data
                   
def has_controller(controllers, controller):
    return any(set(controller).issubset(cnt) for cnt in controllers)

def calc_avg_size(controllers, controller):
    matches = [cnt for cnt in controllers if set(controller).issubset(cnt)]
    avg_size = sum(len(cnt) for cnt in matches) / len(matches) 
    return avg_size

###############################################################################################################################

def load_controllers_and_errors(csvs):
    controllers, errors = {}, {}
    for k, path in csvs.items():
        if path is None:
            continue
        method_controllers, method_errors = defaultdict(dict), defaultdict(dict)
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                network = row['Network']
                target = row['Target']
                controllers_value = ast.literal_eval(row['Controllers'])
                method_controllers[network][target] = controllers_value
                if row['Errors'] == 'NA':
                    method_errors[network][target] = 'NA'
                else:
                    method_errors[network][target] = ast.literal_eval(row['Errors'])
        controllers[k], errors[k] = dict(method_controllers), dict(method_errors)
    return controllers, errors




def proportional_venn3(region_avgs, output_png_prefix,
                       labels=("A","B","C"),
                       color_indices=(0,1,2),
                       tiny_threshold=0.005,   # hide labels < 0.5% of total
                       grid_res=400,
                       figsize=(10,10)):
    """
    Draw a 3-set area-proportional Venn (three circles) from region fractions.

    Parameters
    ----------
    region_avgs : dict
        Keys: "A","B","C","AB","AC","BC","ABC" (fractions, sum ~= 1.0).
    output_png_prefix : str
        Path prefix for the output .png (timestamp + labels appended).
    labels : (str,str,str)
        Set labels for A,B,C (shown in legend).
    color_indices : (int,int,int)
        Indices into your COLORS palette for A,B,C.
    tiny_threshold : float
        Fractions below this (of total) are not labeled on the diagram.
    grid_res : int
        Grid resolution for centroid placement of region labels.
    figsize : (float,float)
        Matplotlib figure size in inches.
    """

    # --- 1) Normalize and unpack ---
    keys3 = ["A","B","C","AB","AC","BC","ABC"]
    reg = {k: float(region_avgs.get(k, 0.0)) for k in keys3}
    total = sum(reg.values()) or 1.0
    for k in keys3:
        reg[k] /= total

    # Per-set totals (include all regions that contain the letter)
    def set_total(name):
        return sum(v for k, v in reg.items() if name in k)

    SA = set_total("A")
    SB = set_total("B")
    SC = set_total("C")

    # Pairwise totals (include the triple region)
    def pair_total(x, y):
        return sum(v for k, v in reg.items() if x in k and y in k)

    TAB = pair_total("A","B")
    TAC = pair_total("A","C")
    TBC = pair_total("B","C")

    # Scale to a convenient drawing area
    AREA_SCALE = 100.0
    SA *= AREA_SCALE; SB *= AREA_SCALE; SC *= AREA_SCALE
    TAB *= AREA_SCALE; TAC *= AREA_SCALE; TBC *= AREA_SCALE

    def radius_from_area(a):
        return math.sqrt(max(a, 0.0) / math.pi)

    rA, rB, rC = radius_from_area(SA), radius_from_area(SB), radius_from_area(SC)

    # --- 2) Circle-circle intersection area + inverse (solve distance from overlap) ---
    def circ_intersection_area(d, r1, r2):
        if d >= r1 + r2: return 0.0
        if d <= abs(r1 - r2): return math.pi * min(r1, r2)**2
        a1 = r1*r1 * math.acos((d*d + r1*r1 - r2*r2) / (2*d*r1))
        a2 = r2*r2 * math.acos((d*d + r2*r2 - r1*r1) / (2*d*r2))
        a3 = 0.5 * math.sqrt(max(0.0, (-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2)))
        return a1 + a2 - a3

    def solve_distance_for_area(target, r1, r2, tol=1e-6):
        # Clamp to achievable range
        minA = 0.0
        maxA = math.pi * min(r1, r2)**2
        t = max(min(target, maxA), minA)

        lo = max(1e-6, abs(r1 - r2) + 1e-6 if t > 0 else r1 + r2)  # if t==0, lo==hi==r1+r2
        hi = r1 + r2 - 1e-6
        if t == 0.0:
            return r1 + r2  # just-touching or apart

        # Bisection on d in (|r1-r2|, r1+r2)
        for _ in range(60):
            mid = 0.5*(lo + hi)
            area = circ_intersection_area(mid, r1, r2)
            if area > t:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < tol:
                break
        return 0.5*(lo + hi)

    # Distances that achieve the requested pairwise overlaps
    dAB = solve_distance_for_area(TAB, rA, rB)
    dAC = solve_distance_for_area(TAC, rA, rC)
    dBC = solve_distance_for_area(TBC, rB, rC)

    # --- 3) Trilateration: place circles in 2D ---
    # A at (0,0), B at (dAB,0). Solve for C s.t. |C-A|=dAC and |C-B|=dBC
    Ax, Ay = 0.0, 0.0
    Bx, By = dAB, 0.0

    # xC derived from two-circle intersection formula along AB
    xC = (dAC**2 - dBC**2 + dAB**2) / (2*dAB) if dAB > 0 else 0.0
    # Pick the "upper" solution for readability; if negative under sqrt, clamp to 0
    y_sq = max(dAC**2 - xC**2, 0.0)
    yC = math.sqrt(y_sq)

    centers = {
        "A": np.array([Ax, Ay]),
        "B": np.array([Bx, By]),
        "C": np.array([xC, yC]),
    }
    radii = {"A": rA, "B": rB, "C": rC}

    # --- 4) Build region masks on a grid (for centroids + final labels) ---
    base = max(rA, rB, rC, 1.0)
    padding = base * 0.8 + max(dAB, dAC, dBC, 1.0)
    xmin = min(Ax, Bx, xC) - padding
    xmax = max(Ax, Bx, xC) + padding
    ymin = min(Ay, By, yC) - padding
    ymax = max(Ay, By, yC) + padding

    X = np.linspace(xmin, xmax, grid_res)
    Y = np.linspace(ymin, ymax, grid_res)
    xx, yy = np.meshgrid(X, Y)

    def inside(name):
        c = centers[name]
        r = radii[name]
        return ((xx - c[0])**2 + (yy - c[1])**2) <= (r*r + 1e-9)

    M = {s: inside(s) for s in ["A","B","C"]}

    def region_mask(code):
        need = set(code)
        mask = np.ones_like(xx, dtype=bool)
        for s in ["A","B","C"]:
            if s in need:
                mask &= M[s]
            else:
                mask &= ~M[s]
        return mask

    region_keys = ["A","B","C","AB","AC","BC","ABC"]
    masks = {k: region_mask(k) for k in region_keys}

    # Measure realized fractions (for sanity; based on pixel counts)
    pixel_area = (xmax - xmin)*(ymax - ymin) / (grid_res*grid_res)
    realized = {k: masks[k].sum() * pixel_area for k in region_keys}
    total_area = sum(realized.values()) or 1.0
    for k in realized:
        realized[k] /= total_area  # now in [0,1], sums ~1

    # Centroids for labels
    def centroid(mask):
        idx = np.argwhere(mask)
        if idx.size == 0:
            return None
        ys = Y[idx[:, 0]]
        xs = X[idx[:, 1]]
        return np.array([xs.mean(), ys.mean()])

    centroids = {k: centroid(masks[k]) for k in region_keys}

    # --- 5) Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    colors = tuple(COLORS[i] for i in color_indices)

    circ = {
        "A": plt.Circle(centers["A"], rA, alpha=0.7, color=colors[0]),
        "B": plt.Circle(centers["B"], rB, alpha=0.7, color=colors[1]),
        "C": plt.Circle(centers["C"], rC, alpha=0.7, color=colors[2]),
    }

    for name in CIRCLE_ORDER:
        ax.add_artist(circ[name])

    # Legend for set labels
    handles = [
        Patch(facecolor=colors[0], alpha=0.9, label=labels[0]),
        Patch(facecolor=colors[1], alpha=0.9, label=labels[1]),
        Patch(facecolor=colors[2], alpha=0.9, label=labels[2]),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=FONTSMALL)

    # Region % labels: show only non-tiny and non-zero targets (use your target reg, not realized)
    for k in region_keys:
        v = reg[k]  # fraction of total desired
        if v <= 0.0 or v < tiny_threshold:
            continue
        ct = centroids.get(k)
        if ct is None:
            continue
        ax.text(ct[0], ct[1], "{:.0f}%".format(v*100.0), ha="center", va="center", fontsize=FONTSMALL)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', 'box')
    ax.axis("off")

    # Ensure output dir exists and save
    outdir = os.path.dirname(output_png_prefix)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    outpath = os.path.join(OUTPUT_DIR, "{}_{}.{}".format(output_png_prefix, util.timestamp(),FILETYPE))
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.clf()
    plt.close()

    return outpath


def venn3_methods_compare(controllers, errors, title, output_dir, methods,
                          color_indices=(0,1,2), only_solo_controllers=ONLY_SOLO_CONTROLLERS):
    if len(methods) != 3:
        raise ValueError("methods must be a list/tuple of exactly 3 method names")

    # Regions for a 3-set Venn
    region_names = ["A","B","C","AB","AC","BC","ABC"]

    # Running totals (we'll average across network-target pairs)
    region_totals = {name: 0.0 for name in region_names}
    num_pairs_counted = 0

    common_pairs = network_and_target_in_all(controllers, methods[0], methods[1], methods[2])
    if not common_pairs:
        print("No common (network,target) across {}.".format(", ".join(methods)))
        return {name: 0.0 for name in region_names}

    for (network, target) in sorted(common_pairs):
        method_sets = []
        for m in methods:
            s = trim_by_error(controllers, errors, m, network, target)
            if only_solo_controllers:
                s = return_only_solo_controllers(s)
            method_sets.append(s)

        # Compute global minimal size across all three method sets
        def set_size(s):
            if s == [] or s == ['1']:
                return 0
            if isinstance(s, dict):
                if s == {} or s == {'1': 1}:
                    return 0
                return len(s)
            try:
                return len(s)
            except TypeError:
                return 1

        size_lists = [[set_size(x) for x in s if x != ['0']] for s in method_sets]
        if not any(size_lists):
            continue

        candidates = [min(sl) for sl in size_lists if sl]
        if not candidates:
            continue
        global_min = min(candidates)

        # Reduce each method to its min-size controllers under the shared global_min
        min_sets = [min_size_sets(s, force_min_size=global_min) for s in method_sets]

        A, B, C = min_sets
        union_all = A | B | C
        if len(union_all) == 0:
            continue

        # Count each element into its precise Venn region
        counts = defaultdict(int)
        for elem in union_all:
            inA = elem in A
            inB = elem in B
            inC = elem in C

            if inA and not inB and not inC: counts["A"] += 1
            elif inB and not inA and not inC: counts["B"] += 1
            elif inC and not inA and not inB: counts["C"] += 1
            elif inA and inB and not inC: counts["AB"] += 1
            elif inA and inC and not inB: counts["AC"] += 1
            elif inB and inC and not inA: counts["BC"] += 1
            elif inA and inB and inC: counts["ABC"] += 1

        # Normalize by |A ∪ B ∪ C| and add to running totals
        total = float(len(union_all))
        for name in region_names:
            region_totals[name] += counts[name] / total

        num_pairs_counted += 1

    # Average across all counted (network, target) pairs
    if num_pairs_counted == 0:
        print("No comparable min-size controllers across the three methods.")
        return {name: 0.0 for name in region_names}

    region_avgs = {name: region_totals[name] / num_pairs_counted for name in region_names}
    return region_avgs


###############################################################################################

def proportional_venn4(region_avgs, output_png_prefix, labels=("A","B","C","D"),
                       color_indices=(0,1,2,3), tiny_threshold=0.005,  # hide <0.5%
                       grid_res=400):
    """
    Draw an approximate 4-set Euler diagram from region fractions (region_avgs)
    collected as in venn4_methods_compare(). Regions are:
      A,B,C,D, AB,AC,AD, BC,BD,CD, ABC,ABD,ACD,BCD, ABCD
    Values should sum to ~1.0; we rescale internally to sum to 1.0.
    Only non-zero regions >= tiny_threshold are labeled.
    """

    # --- 1) Normalize and unpack regions (ensure all 15 keys exist) ---
    all_regions = ["A","B","C","D",
                   "AB","AC","AD","BC","BD","CD",
                   "ABC","ABD","ACD","BCD","ABCD"]
    reg = {k: float(region_avgs.get(k, 0.0)) for k in all_regions}
    total = sum(reg.values()) or 1.0
    for k in all_regions:
        reg[k] /= total

    # --- 2) Target set areas and pairwise overlaps (include higher-order overlaps) ---
    def set_total(name):
        # Sum all regions that include the set letter
        return sum(v for k, v in reg.items() if name in k)

    SA, SB, SC, SD = (set_total("A"), set_total("B"), set_total("C"), set_total("D"))

    def pair_total(x, y):
        return sum(v for k, v in reg.items() if x in k and y in k)

    TAB = pair_total("A","B")
    TAC = pair_total("A","C")
    TAD = pair_total("A","D")
    TBC = pair_total("B","C")
    TBD = pair_total("B","D")
    TCD = pair_total("C","D")

    # Scale areas to a convenient canvas area (e.g., 100.0)
    AREA_SCALE = 100.0
    SA *= AREA_SCALE; SB *= AREA_SCALE; SC *= AREA_SCALE; SD *= AREA_SCALE
    TAB *= AREA_SCALE; TAC *= AREA_SCALE; TAD *= AREA_SCALE
    TBC *= AREA_SCALE; TBD *= AREA_SCALE; TCD *= AREA_SCALE

    # Radii from set areas
    def radius_from_area(a):
        return math.sqrt(max(a, 0.0) / math.pi)

    rA, rB, rC, rD = (radius_from_area(SA), radius_from_area(SB),
                      radius_from_area(SC), radius_from_area(SD))

    # --- 3) Circle-circle intersection area helper ---
    def circ_intersection_area(d, r1, r2):
        d = float(d)
        if d >= r1 + r2: return 0.0
        if d <= abs(r1 - r2): return math.pi * min(r1, r2)**2
        # Lens area
        a1 = r1*r1 * math.acos((d*d + r1*r1 - r2*r2) / (2*d*r1))
        a2 = r2*r2 * math.acos((d*d + r2*r2 - r1*r1) / (2*d*r2))
        a3 = 0.5 * math.sqrt(max(0.0, (-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2)))
        return a1 + a2 - a3

    # --- 4) Place circles: initial layout and tiny grid search to match pair overlaps ---
    # Layout: A(-dx,0), B(+dx,0), C(0,+dy), D(0,-dy); also allow small per-circle tweaks.
    # We'll grid-search dx,dy scale to best match pair targets.
    def pair_error(dx, dy):
        # centers
        Ax, Ay = -dx, 0.0
        Bx, By = +dx, 0.0
        Cx, Cy = 0.0, +dy
        Dx, Dy = 0.0, -dy
        # distances
        dAB = math.hypot(Ax-Bx, Ay-By)
        dAC = math.hypot(Ax-Cx, Ay-Cy)
        dAD = math.hypot(Ax-Dx, Ay-Dy)
        dBC = math.hypot(Bx-Cx, By-Cy)
        dBD = math.hypot(Bx-Dx, By-Dy)
        dCD = math.hypot(Cx-Dx, Cy-Dy)
        # areas predicted
        pAB = circ_intersection_area(dAB, rA, rB)
        pAC = circ_intersection_area(dAC, rA, rC)
        pAD = circ_intersection_area(dAD, rA, rD)
        pBC = circ_intersection_area(dBC, rB, rC)
        pBD = circ_intersection_area(dBD, rB, rD)
        pCD = circ_intersection_area(dCD, rC, rD)
        # MSE across pairs (avoid overweighting zero targets)
        err  = (pAB - TAB)**2 + (pAC - TAC)**2 + (pAD - TAD)**2
        err += (pBC - TBC)**2 + (pBD - TBD)**2 + (pCD - TCD)**2
        return err

    # Choose reasonable search ranges based on radii
    base = max(rA, rB, rC, rD, 1.0)
    dx_vals = np.linspace(0.4*base, 2.5*base, 36)
    dy_vals = np.linspace(0.4*base, 2.5*base, 36)

    best = (None, float("inf"))
    for dx in dx_vals:
        for dy in dy_vals:
            e = pair_error(dx, dy)
            if e < best[1]:
                best = ((dx, dy), e)

    (dx, dy), _ = best

    # Final centers
    centers = {
        "A": np.array([-dx, 0.0]),
        "B": np.array([+dx, 0.0]),
        "C": np.array([0.0, +dy]),
        "D": np.array([0.0, -dy]),
    }
    radii = {"A": rA, "B": rB, "C": rC, "D": rD}

    # --- 5) Grid-sample to find centroids for each region for labeling ---
    # Bounding box
    RMAX = max(radii.values()) + max(dx, dy) + 1.0
    xmin, xmax = -RMAX, +RMAX
    ymin, ymax = -RMAX, +RMAX

    X = np.linspace(xmin, xmax, grid_res)
    Y = np.linspace(ymin, ymax, grid_res)
    xx, yy = np.meshgrid(X, Y)
    points = np.stack([xx, yy], axis=-1)  # (H,W,2)

    # Masks for each set
    def inside(name):
        c = centers[name]
        r = radii[name]
        return ((xx - c[0])**2 + (yy - c[1])**2) <= (r*r + 1e-9)

    M = {s: inside(s) for s in ["A","B","C","D"]}

    # Build masks for each region name
    def region_mask(code):
        need = set(code)
        # included sets must be True; excluded must be False
        mask = np.ones_like(xx, dtype=bool)
        for s in ["A","B","C","D"]:
            if s in need:
                mask &= M[s]
            else:
                mask &= ~M[s]
        return mask

    region_masks = {k: region_mask(k) for k in all_regions}

    # For centroid, use the sampler points of the region; compute mean of coordinates
    def centroid_of_mask(mask):
        idx = np.argwhere(mask)
        if idx.size == 0:
            return None
        # idx gives [row, col] -> map to X,Y
        ys = Y[idx[:, 0]]
        xs = X[idx[:, 1]]
        return np.array([xs.mean(), ys.mean()])

    centroids = {k: centroid_of_mask(m) for k, m in region_masks.items()}

    # --- 6) Plot ---
    fig, ax = plt.subplots(figsize=(16, 10))
    colors = tuple(COLORS[i] for i in color_indices)

    # Draw circles
    circ = {
        "A": plt.Circle(centers["A"], rA, alpha=0.45, color=colors[0]),
        "B": plt.Circle(centers["B"], rB, alpha=0.45, color=colors[1]),
        "C": plt.Circle(centers["C"], rC, alpha=0.45, color=colors[2]),
        "D": plt.Circle(centers["D"], rD, alpha=0.45, color=colors[3]),
    }
    for c in circ.values():
        ax.add_artist(c)

    # Set labels for sets near their circle tops
    def label_circle_top(name, text):
        cx, cy = centers[name]
        r = radii[name]
        ax.text(cx, cy + r + 0.12*RMAX, text, ha="center", va="bottom", fontsize=20)

    handles = [
        Patch(facecolor=colors[0], alpha=0.45, label=labels[0]),
        Patch(facecolor=colors[1], alpha=0.45, label=labels[1]),
        Patch(facecolor=colors[2], alpha=0.45, label=labels[2]),
        Patch(facecolor=colors[3], alpha=0.45, label=labels[3]),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=FONTSMALL)

    # Label non-zero, non-tiny regions with their percentage
    for k in all_regions:
        v = reg[k]  # fraction of total
        if v <= tiny_threshold or v <= 0.0:
            continue
        ct = centroids.get(k, None)
        if ct is None:
            continue
        ax.text(ct[0], ct[1], "{:.0f}%".format(v*100.0), ha="center", va="center", fontsize=FONTSMALL)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax*1.05)
    ax.set_aspect('equal', 'box')
    ax.axis("off")

    # Ensure directory exists
    outdir = os.path.dirname(output_png_prefix)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    outpath = os.path.join(OUTPUT_DIR, "{}_{}.{}".format(output_png_prefix, util.timestamp(),FILETYPE))
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.clf()
    plt.close()

###############################################################################

def main(choice):
    if choice.lower() == 'batch':
        title=BASE_STR

        #plot_error_results(CSVS, OUTPUT_DIR, title)

        controllers, errors = load_controllers_and_errors(CSVS)
        color_indices = (0,1,2)
        methods = list(CSVS.keys())
        methods.remove('MC')
        data = MCcoverageTest()
        plot_MCcoverageTest(data, methods, title)    

        controllers.pop('MC', None)
        errors.pop('MC', None)
        CSVS.pop('MC',None)
        if len(CSVS)==3:
            region_avgs = venn3_methods_compare(controllers, errors, title, OUTPUT_DIR, methods, color_indices=color_indices, only_solo_controllers=ONLY_SOLO_CONTROLLERS)
            proportional_venn3(region_avgs, title, labels=methods, color_indices=color_indices, tiny_threshold=0.01, grid_res=400)

        plot_error_results(CSVS, OUTPUT_DIR, title)
        #print_poke_uniques(controllers, errors)

    elif choice.lower() == 'mechanism':
        assert(0) # in progress
        plot_mechanism()

    else:
        sys.exit("Unrecognized last argument " + str(choice) + ", should be 'batch' or 'mechanism'")

####################################################################

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Usage: python NUDGE_plots.py [batch | mechanism]")
    main(sys.argv[1])



