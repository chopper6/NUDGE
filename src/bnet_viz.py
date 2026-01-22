
import re, json, math, shutil, subprocess, random
from math import ceil
from collections import defaultdict
from pathlib import Path
import networkx as nx

DEFAULT_COLORS = {
    "input":   "#B7E1F7",
    "internal":"#CBE8C0",
    "output":  "#FFF6B0",
    "Y":       "#FFD580",
}
DEFAULT_EDGE_COLOR = {"+": "#2E75B6", "-": "#C00000", "both": "#7F7F7F"}

def parse_bnet(text):
    rules = {}
    nodes = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            continue
        node, expr = parts
        nodes.append(node)
        rules[node] = expr
    return nodes, rules

def extract_literals(expr):
    toks = re.findall(r'!?\b[a-zA-Z_][a-zA-Z0-9_]*\b', expr)
    lits = []
    for t in toks:
        if t.startswith("!"):
            lits.append((t[1:], "-"))
        else:
            lits.append((t, "+"))
    return lits

def darker(hex_color, factor=0.7):
    c = [int(hex_color[i:i+2],16) for i in (1,3,5)]
    c = [max(0, min(255, int(v*factor))) for v in c]
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

def _chunk(seq, k):
    if not seq:
        return []
    if not k or k <= 0:  # None or 0 -> single row
        return [list(seq)]
    return [seq[i:i+k] for i in range(0, len(seq), k)]

def render_bnet_graph(
    bnet_text=None,
    bnet_path=None,
    max_per_row=8,
    max_per_row_input=None,
    max_per_row_output=None,
    order_mode="hierarchical_order",
    osci_top_pct=0.5,
    osci_seed=42,
    x_gap=2.0,
    y_gap=2.2,
    y_merge_threshold=1,
    scale=48.0,

    # Node appearance
    node_shape="circle",        # "circle", "ellipse", "box" (square/rect), etc.
    node_width=0.9,
    node_height=0.9,
    fontsize=11,
    fontname="DejaVu Sans",
    bold_labels=True,
    node_style="filled",        # NEW: e.g. "rounded,filled" for rounded boxes
    node_border_width=2.2,      # NEW: control perimeter thickness
    node_border_color=None,     # NEW: override border color (default darker(fill))
    use_rounded_box=False,      # NEW: convenience; if True -> shape=box & style includes rounded

    # Edge appearance
    edge_penwidth=2,            # NEW: edge thickness
    arrowsize=0.7,              # keep as before

    # IO layout control
    outputs_layout="separate",  # NEW: "separate" (default) or "internal"

    palette=None,
    edge_palette=None,
    output_dir=".",
    base_name="bnet_render",
    write_positions_json=True,
    return_artifacts=True,

    # Focus subgraph options
    focus_targets=None,
    focus_max_hops=None,
):
    if not bnet_text and bnet_path:
        with open(bnet_path) as f:
            bnet_text = f.read()
    if not bnet_text:
        raise ValueError("Provide either bnet_text or bnet_path")

    COLORS = dict(DEFAULT_COLORS)
    if palette:
        COLORS.update(palette)
    EDGE_COLOR = dict(DEFAULT_EDGE_COLOR)
    if edge_palette:
        EDGE_COLOR.update(edge_palette)

    nodes, rules = parse_bnet(bnet_text)

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)

    edge_signs = defaultdict(set)
    for dst, expr in rules.items():
        for src, sgn in extract_literals(expr):
            if src not in G:
                G.add_node(src)
            edge_signs[(src, dst)].add(sgn)
    for (src, dst), sset in edge_signs.items():
        sign = "both" if len(sset) > 1 else list(sset)[0]
        G.add_edge(src, dst, sign=sign)

    def is_pure_self(node, expr):
        lits = extract_literals(expr)
        vars_only = set(v for v, s in lits)
        return len(vars_only) == 1 and node in vars_only

    # --- Focus subgraph (unchanged logic) ---
    H = G
    if focus_targets:
        if isinstance(focus_targets, str):
            targets = [focus_targets]
        else:
            targets = list(focus_targets)
        targets = [t for t in targets if t in G]
        if not targets:
            raise ValueError("None of the focus_targets exist in the network.")

        sub_nodes = set()
        if focus_max_hops is None:
            for t in targets:
                sub_nodes.update(nx.ancestors(G, t))
            sub_nodes.update(targets)
        else:
            Gr = G.reverse(copy=False)
            for t in targets:
                lengths = nx.single_source_shortest_path_length(Gr, t, cutoff=focus_max_hops)
                sub_nodes.update(lengths.keys())
            sub_nodes.update(targets)
        H = G.subgraph(sub_nodes).copy()
        nodes = [n for n in nodes if n in H]
    else:
        targets = None  # for artifacts

    # --- Classify nodes based on H ---
    y_node = "Y" if "Y" in H.nodes else None
    inputs = sorted([n for n in nodes if n != y_node and is_pure_self(n, rules.get(n, "")) and n in H])
    # Compute raw outputs first (sinks in H)
    outputs_raw = sorted([n for n in nodes if n != y_node and H.out_degree(n) == 0 and n not in inputs and n in H])

    # Decide whether outputs are separate or considered internal
    if outputs_layout == "internal":
        outputs = []  # no separate outputs row
        internal = sorted([n for n in nodes if n in H and n not in set(inputs + ([y_node] if y_node else []))])
    elif outputs_layout == "separate":
        outputs = outputs_raw
        internal = sorted([n for n in nodes if n in H and n not in set(inputs + outputs + ([y_node] if y_node else []))])
    else:
        raise ValueError('outputs_layout must be "separate" or "internal"')

    deg_out = {n: H.out_degree(n) for n in internal}
    deg_in  = {n: H.in_degree(n)  for n in internal}

    if order_mode == "hierarchical_order":
        internal_sorted = sorted(internal, key=lambda n: (-deg_out[n], deg_in[n], n))
    elif order_mode == "random_order":
        rng = random.Random(osci_seed) if osci_seed is not None else random
        internal_sorted = internal.copy()
        rng.shuffle(internal_sorted)
    elif order_mode == "osci_order":
        hier = sorted(internal, key=lambda n: (-deg_out[n], deg_in[n], n))
        k = math.ceil(len(hier) * osci_top_pct)
        top_pool = hier[:k]
        bottom_pool = hier[k:]
        rng = random.Random(osci_seed) if osci_seed is not None else random
        internal_sorted = []
        take_top = True
        while top_pool or bottom_pool:
            if take_top and top_pool:
                i = rng.randrange(len(top_pool))
                internal_sorted.append(top_pool.pop(i))
            elif (not take_top) and bottom_pool:
                i = rng.randrange(len(bottom_pool))
                internal_sorted.append(bottom_pool.pop(i))
            else:
                pool = top_pool if top_pool else bottom_pool
                while pool:
                    i = rng.randrange(len(pool))
                    internal_sorted.append(pool.pop(i))
            take_top = not take_top
    else:
        raise ValueError(f"Unknown order_mode: {order_mode}")

    # --- Rows ---
    rows = []
    for i, r in enumerate(_chunk(inputs, max_per_row_input)):
        rows.append((f"inputs_{i}", r))

    internal_rows = [internal_sorted[i:i+max_per_row]
                     for i in range(0, len(internal_sorted), max_per_row)]
    for i, r in enumerate(internal_rows):
        rows.append((f"internal_{i}", r))

    if outputs_layout == "separate":
        output_rows = _chunk(outputs, max_per_row_output)
        if output_rows:
            for i, r in enumerate(output_rows):
                rows.append((f"outputs_{i}", r))
            if y_node:
                last_label, last_out = rows[-1]
                if last_label.startswith("outputs_") and len(last_out) <= y_merge_threshold:
                    last_out.append(y_node)
                else:
                    rows.append(("Y", [y_node]))
        elif y_node:
            rows.append(("Y", [y_node]))
    else:
        # No separate outputs row; Y stays alone if present
        if y_node:
            rows.append(("Y", [y_node]))

    GLOBAL_COLS = max(
        max_per_row or 1,
        (max_per_row_input if max_per_row_input else (len(inputs) if inputs else 1)),
        (max_per_row_output if max_per_row_output else ((len(outputs) if outputs_layout=="separate" else 1))),
    )

    def place_row(row_nodes, row_idx, pos_map, total_rows):
        n = len(row_nodes)
        total_width = (GLOBAL_COLS - 1) * x_gap
        start_x = -total_width / 2.0
        left_pad = (GLOBAL_COLS - n) / 2.0
        for i, node in enumerate(row_nodes):
            x = start_x + (left_pad + i) * x_gap
            y = (total_rows - 1 - row_idx) * y_gap
            pos_map[node] = (x, y)

    total_rows = len(rows)
    pos_grid = {}
    for row_idx, (_label, row_nodes) in enumerate(rows):
        if row_nodes:
            place_row(row_nodes, row_idx, pos_grid, total_rows)

    def node_kind(n):
        if n == y_node: return "Y"
        if n in inputs: return "input"
        # If outputs are treated as internal, keep them visually "internal"
        if outputs_layout == "separate" and n in (outputs if 'outputs' in locals() else []):
            return "output"
        return "internal"

    def gv_pos(n):
        x, y = pos_grid[n]
        return f"{x*scale},{y*scale}!"

    # --- Node/edge defaults ---
    effective_shape = node_shape
    effective_style = node_style
    if use_rounded_box:
        effective_shape = "box"
        if "rounded" not in effective_style:
            effective_style = "rounded," + effective_style if effective_style else "rounded"

    dot_lines = []
    dot_lines.append('digraph G {')
    dot_lines.append('  graph [splines=ortho, overlap=false, outputorder=edgesfirst, bgcolor="white"];')
    dot_lines.append(
        f'  node  [shape={effective_shape}, style="{effective_style}", fontname="{fontname}", '
        f'fontsize={fontsize}, fixedsize=true, width={node_width}, height={node_height}];')
    dot_lines.append(f'  edge  [penwidth={edge_penwidth}, arrowsize={arrowsize}];')

    def html_escape(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    for n in nodes:
        if n not in H:
            continue
        fill = COLORS[node_kind(n)]
        border = node_border_color if node_border_color else darker(fill, 0.7)
        if bold_labels:
            label_attr = f'<<B>{html_escape(n)}</B>>'
        else:
            label_attr = f'"{n}"'
        dot_lines.append(
            f'  "{n}" [label={label_attr}, fillcolor="{fill}", color="{border}", '
            f'penwidth={node_border_width}, pos="{gv_pos(n)}", pin=true];'
        )

    for u, v, data in H.edges(data=True):
        color = EDGE_COLOR.get(data.get("sign", "+"), "#444444")
        dot_lines.append(f'  "{u}" -> "{v}" [color="{color}"];')
    dot_lines.append('}')

    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = out_dir / f"{base_name}.dot"
    with open(dot_path, "w") as f:
        f.write("\n".join(dot_lines))

    png_out = out_dir / f"{base_name}.png"
    svg_out = out_dir / f"{base_name}.svg"
    neato_bin = shutil.which("neato")
    render = {"png": False, "svg": False, "neato_found": bool(neato_bin)}
    if neato_bin:
        try:
            subprocess.run([neato_bin, "-n2", "-Tpng", str(dot_path), "-o", str(png_out)], check=True)
            render["png"] = True
        except Exception as e:
            render["png"] = f"Failed: {e}"
        try:
            subprocess.run([neato_bin, "-n2", "-Tsvg", str(dot_path), "-o", str(svg_out)], check=True)
            render["svg"] = True
        except Exception as e:
            render["svg"] = f"Failed: {e}"

    pos_json = out_dir / f"{base_name}_positions.json"
    with open(pos_json, "w") as f:
        json.dump({k: v for k, v in pos_grid.items() if k in H}, f, indent=2)

    artifacts = {
        "dot": str(dot_path),
        "png": str(png_out) if render.get("png") is True else render.get("png"),
        "svg": str(svg_out) if render.get("svg") is True else render.get("svg"),
        "positions_json": str(pos_json),
        "neato_found": render["neato_found"],
        "order_mode": order_mode,
        "max_per_row_internal": max_per_row,
        "max_per_row_input": max_per_row_input,
        "max_per_row_output": max_per_row_output,
        "focus_targets": targets if focus_targets else None,
        "focus_max_hops": focus_max_hops,
        "outputs_layout": outputs_layout,
        "num_nodes_drawn": H.number_of_nodes(),
        "num_edges_drawn": H.number_of_edges(),
    }
    return artifacts
