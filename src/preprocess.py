'''
Preprocess networks before a batch run by sorting networks by the number of nodes or generating targets
'''

import sys, os, shutil, re, random, csv
import networkx as nx
#import numpy as np

NETWORK_OUTPUT_DIR = './input/networks/temp/'
NETWORK_INPUT_DIR = './input/networks/all/'
MAX_NODES = 20

MAX_SCC = 10
SORT_BY = 'scc'

TARGET_INPUT_DIR = './input/networks/large11SCC/'
TARGET_OUTPUT_CSV = './input/targets_large11SCC_4targets2.csv'
TARGETS_PER_NET = 4
INCLUDE_NON_OUTPUTS = True

#############################################################################################

def sort_networks():
	if not os.path.exists(NETWORK_OUTPUT_DIR):
		os.makedirs(NETWORK_OUTPUT_DIR)

	sizes = []
	for fname in os.listdir(NETWORK_INPUT_DIR):
		if fname.lower().endswith(".bnet"):

			source_file_path = os.path.join(NETWORK_INPUT_DIR, fname)
			graph = build_graph_from_bnet(source_file_path)
			size = compute_size(graph)

			if SORT_BY.lower() == 'nodes':
				max_size = MAX_NODES
			elif SORT_BY.lower() == 'scc':
				max_size = MAX_SCC
			if size > max_size:
				sizes += [size]
				shutil.copy(source_file_path, os.path.join(NETWORK_OUTPUT_DIR, fname))

	#print("max size =",np.max(sizes),'\nmean size =',np.mean(sizes),'\nmedian size =',np.median(sizes),'\nmin size =',np.min(sizes))
	#print("num nets =",len(sizes))

def build_graph_from_bnet(bnet_file_path):
	graph = nx.DiGraph()
	identifier_pattern = re.compile(r'!?[A-Za-z_][A-Za-z0-9_]*')
	keywords = {"and", "or", "not", "AND", "OR", "NOT"}

	with open(bnet_file_path, "r", encoding="utf-8", errors="ignore") as handle:
		for raw_line in handle:
			line = raw_line.strip()
			if not line:
				continue
			if line.startswith("#") or line.startswith("//"):
				continue

			if "," in line:
				lhs, rhs = line.split(",", 1)
			elif "\t" in line:
				lhs, rhs = line.split("\t", 1)
			else:
				continue

			target_raw = lhs.strip()
			target_node = target_raw.lstrip("!").strip()

			if target_node not in graph:
				graph.add_node(target_node)

			inputs_found = identifier_pattern.findall(rhs)
			normalized_inputs = set()

			for token in inputs_found:
				normalized = token.lstrip("!")
				if normalized in keywords:
					continue
				normalized_inputs.add(normalized)

			for input_node in normalized_inputs:
				if input_node not in graph:
					graph.add_node(input_node)
				graph.add_edge(input_node, target_node)

	return graph


def compute_size(graph):
	if SORT_BY.lower() == 'nodes':
		return graph.number_of_nodes()
	if graph.number_of_nodes() == 0:
		return 0
	sizes = [len(component) for component in nx.strongly_connected_components(graph)]
	return max(sizes) if sizes else 0


################################################################################

def select_targets():
	data = {'Network':[],'Target':[]}
	for filename in os.listdir(TARGET_INPUT_DIR):
		if not (filename.endswith('.bnet') or filename.endswith('.txt')):
			continue
		print("Processing file",filename)
		file_path = os.path.join(TARGET_INPUT_DIR, filename)
		
		G, expressions = load_network_topology(file_path)
		inputs = get_inputs(G)
		outputs = get_outputs(G, expressions)

		num_targets = min(TARGETS_PER_NET, len(G.nodes)-len(inputs))

		targets = []
		num_target_outputs = min(len(outputs),num_targets)
		for i in range(num_target_outputs):
			cont = True
			while cont:
				target=random.choice(outputs)
				if target not in targets:
					targets.append(target)
					cont=False
		if INCLUDE_NON_OUTPUTS:
			for i in range(num_targets-num_target_outputs):
				cont = True 
				while cont:
					target=random.choice(list(G.nodes))
					if (target not in targets) and (target not in inputs):
						targets.append(target)
						cont= False
		
		for target in targets:
			data['Network'] += [filename]
			data['Network'] += [filename]
			data['Target'] += [target]
			data['Target'] += ['!' + target]

	write_targets_to_csv(data)

def load_network_topology(file_path):
	G = nx.DiGraph()
	expressions = {}
	with open(file_path, 'r') as f:
		lines = f.readlines()
	if file_path.endswith('.txt') and lines and lines[0].lower().startswith("target"):
		lines = lines[1:]

	for line in lines:
		line = line.strip()
		if not line or line.startswith('#'):
			continue
		if '=' in line:
			target, expr = map(str.strip, line.split('=', 1))
		elif ',' in line:
			target, expr = map(str.strip, line.split(',', 1))
		else:
			continue
		expressions[target] = expr
		factors = parse_logic_expression(expr)
		G.add_node(target)
		for f in factors:
			G.add_node(f)
			G.add_edge(f, target)
	return G, expressions

def get_inputs(G):
	inputs = []
	for node in G.nodes:
		in_edges = list(G.in_edges(node))
		if not in_edges:
			inputs.append(node)
			continue

		all_self_loops = all(src == node for src, _ in in_edges)
		if all_self_loops:
			inputs.append(node)

	return inputs

def get_outputs(G, expressions):
	outputs = []
	for node in G.nodes:
		if G.out_degree(node) != 0:
			continue  
		incoming_from_others = [
			src for src, tgt in G.in_edges(node)
			if src != node
		]
		if incoming_from_others:
			outputs.append(node)
	return outputs

def parse_logic_expression(expr):
	tokens = re.findall(r'\b\w+\b', expr)
	return {t for t in tokens if t not in {'0', '1'}}

def write_targets_to_csv(data):
	with open(TARGET_OUTPUT_CSV, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(data.keys())
		n_rows = max(len(v) for v in data.values())
		for i in range(n_rows):
			row = [data[k][i] if i < len(data[k]) else "" for k in data]
			writer.writerow(row)


#################################################################################

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python preprocess [sort | targets]")
	if sys.argv[1] == 'sort':
		sort_networks()
	elif sys.argv[1] == 'targets':
		select_targets()
	else:
		sys.exit("Unrecognized arg, should be 'sort' or 'targets'")