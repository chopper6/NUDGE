'''
contructs Net objects that include graph structure and the associated logic functions                        

'''

import util
import os, sys, itertools

CUPY, cp = util.import_cp_or_np(try_cupy=1)

class Net:
    def __init__(self, params, network_file=None):
        self.params = params
        self.F, self.Fmapd = {}, None
        self.n, self.n_neg = 0, 0
        self.nodes, self.parityNodes = [], []
        self.nodeNames, self.nodeNums = [], {}
        self.not_string = '!'
        self.debug = params.get('debug', False)
        ml = network_file or params['network_file']
        self.read_from_file(ml)

    def __str__(self):
        return "Net(n=%d)" % self.n

    def prepare(self, params=None):
        if params is not None:
            self.params = params
        self.add_self_loops()
        self.check_mutations()
        self.apply_mutations()
        self.build_Fmapd_and_A()

    def read_from_file(self, net_file):
        if not os.path.isfile(net_file):
            sys.exit("Can't find network file: " + str(net_file))
        with open(net_file, 'r', encoding='utf-8') as f:
            format_name = get_encoding(f, net_file)
            node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = get_file_format(format_name)
            self.not_string = not_str
            loop = 0
            while True:
                line = read_line(f, loop)
                if (line is None) or (line.isspace()):
                    break
                parts = line.strip().split(node_fn_split)
                node_name = read_node_name(parts, strip_from_node, self.debug, self.nodeNames)
                self.add_node(node_name)
                if len(parts) == 1 or (len(parts[1].strip()) == 0):
                    self.F[node_name] = [[node_name]]
                else:
                    self.F[node_name] = read_in_function(parts[1], format_name)
                loop += 1
        self.add_hidden_inputs()
        self.build_negative_nodes()

    def add_node(self, nodeName, isNegative=False):
        newNode = Node(self, nodeName, self.n_neg, isNegative=isNegative)
        self.nodeNames.append(nodeName)
        self.nodeNums[nodeName] = self.n_neg
        self.parityNodes.append(newNode)
        self.n_neg += 1
        if not isNegative:
            self.F[nodeName] = self.F.get(nodeName, [])
            self.nodes.append(newNode)
            self.n += 1

    def build_negative_nodes(self):
        for i in range(self.n):
            self.add_node(self.not_string + self.nodeNames[i], isNegative=True)

    def add_self_loops(self):
        for node in self.nodes:
            if self.F[node.name] in [[['0']],[['1']]]:
                self.params.setdefault('init', {})[node.name] = int(self.F[node.name][0][0])
                self.F[node.name] = [[node.name]]
            if node.name in self.params.get('inputs', []) and util.istrue(self.params, 'pin_inputs'):
                self.F[node.name] = [[node.name]]

    def check_mutations(self):
        for k in self.params.get('mutations', {}).keys():
            if k not in self.nodeNames:
                sys.exit("Mutation on %s not in network" % k)

    def apply_mutations(self):
        if len(self.params.get('mutations', {})) == 0:
            return
        for node in self.nodeNames:
            if node in self.params['mutations']:
                lng = len(self.F[node])
                self.F[node] = [[node] for _ in range(max(1,lng))]
                self.params.setdefault('init', {})[node] = int(self.params['mutations'][node])

    def build_Fmapd_and_A(self):
        # Build tensors used by simulate.step_core
        n = self.n
        nodes_to_clauses = []
        clauses_to_threads = []
        threads_to_nodes = []
        nodes_clause = {i: [] for i in range(n)}
        self.A = cp.zeros((n, n), dtype=bool)
        self.num_clauses = 0
        self.max_literals = 0
        self.max_clauses = 0
        for i in range(n):
            numbered = [clause_num_and_add_to_A(self, i, c) for c in self.nodes[i].F()]
            self.max_clauses = max(self.max_clauses, len(numbered))
            self.max_literals = max(self.max_literals, *(len(c) for c in numbered)) if numbered else self.max_literals
            nodes_to_clauses += numbered
            nodes_clause[i] = list(range(self.num_clauses, self.num_clauses + len(numbered)))
            self.num_clauses += len(numbered)

        make_square_clauses(self, nodes_to_clauses)
        nodes_to_clauses = cp.array(nodes_to_clauses, dtype=get_index_dtype(self.n))
        if self.num_clauses > 0:
            assert nodes_to_clauses.shape == (self.num_clauses, self.max_literals)

        m = min(self.params['clause_bin_size'], max(1, self.max_clauses))
        i = 0
        while sum(len(v) for v in nodes_clause.values()) > 0:
            this_set = [[] for _ in range(n)]
            threads_to_nodes.append(cp.zeros((n, n), dtype=bool))
            sorted_keys = sorted(nodes_clause, key=lambda k: len(nodes_clause[k]), reverse=True)
            nodes_clause = {k: nodes_clause[k] for k in sorted_keys}
            node_indx = 0
            prev_take = sorted_keys[node_indx] if sorted_keys else 0

            for j in range(n):
                if sum(len(v) for v in nodes_clause.values()) > 0:
                    take = sorted_keys[node_indx]
                    threads_to_nodes[i][j, take] = 1
                    if len(nodes_clause[take]) >= m:
                        this_set[j] = nodes_clause[take][:m]
                        if len(nodes_clause[take]) == m:
                            node_indx = min(node_indx + 1, len(sorted_keys) - 1)
                        del nodes_clause[take][:m]
                    else:
                        top = len(nodes_clause[take])
                        this_set[j] = nodes_clause[take][:top]
                        rem = m - top
                        this_set[j] += [this_set[j][-1]] * rem if this_set[j] else [0] * rem
                        del nodes_clause[take][:top]
                        node_indx = min(node_indx + 1, len(sorted_keys) - 1)
                else:
                    threads_to_nodes[i][j, prev_take] = 1
                    this_set[j] = this_set[j - 1] if j>0 else [0]*m
                prev_take = take if sum(len(v) for v in nodes_clause.values()) > 0 else prev_take

            clauses_to_threads.append(this_set)
            i += 1
            if i > 1000000:
                sys.exit("infinite loop in build_Fmapd_and_A")

        thread_dtype = util.get_uint_dtype(max(self.params['num_samples'],self.num_clauses))
        clauses_to_threads = cp.array(clauses_to_threads, dtype=thread_dtype)
        threads_to_nodes = cp.array(threads_to_nodes, dtype=bool)
        self.Fmapd = {'nodes_to_clauses': nodes_to_clauses, 'clauses_to_threads': clauses_to_threads, 'threads_to_nodes': threads_to_nodes}

    def add_hidden_inputs(self):
        for node in list(self.nodeNames):
            for clause in self.F[node]:
                for lit in clause:
                    base = lit.replace('!','')
                    if base not in self.nodeNames:
                        self.add_node(base); self.F[base] = [[base]]

class Node:
    def __init__(self, G, name, num, isNegative=False):
        self.name = name
        self.num = num
        self.isNegative = isNegative
        self.G = G
    def F(self):
        return self.G.F[self.name]
    def setF(self, new_fn):
        self.G.F[self.name] = new_fn

######################################################################################

def get_file_format(format_name):
    if format_name == 'DNFsymbolic': return '\t', ' ', '&', '-', [], []
    if format_name == 'DNFwords':    return '*= ', ' or ', ' and ', 'not ', ['(', ')'], ['(', ')']
    if format_name == 'bnet':        return ',', '|', '&', '!', ['(', ')'], []
    return '=', ' AND ', ' OR ', 'NOT ', ['(', ')'], []

def read_node_name(parts, strip_from_node, debug, existing):
    node_name = parts[0].strip().replace(' ', '').replace('\t', '')
    for s in strip_from_node: node_name = node_name.replace(s, '')
    if debug: assert node_name not in existing
    return node_name

def read_in_function(line_part, format_name):
    fn = []
    node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = get_file_format(format_name)
    for clause in line_part.split(clause_split):
        clean = clause
        for s in strip_from_clause: clean = clean.replace(s, '')
        lits = []
        for lit in clean.split(literal_split):
            name = lit
            for s in strip_from_node: name = name.replace(s, '')
            lits.append(name.replace(' ', '').replace('\t', ''))
        fn.append(lits)
    return fn

def clause_num_and_add_to_A(self_net, source_node_num, clause):
    clause_fn = []
    self_net.max_literals = max(self_net.max_literals, len(clause))
    for name in clause:
        literal_node = self_net.nodeNums[name]
        clause_fn.append(literal_node)
        base_idx = literal_node - self_net.n if self_net.parityNodes[literal_node].isNegative else literal_node
        self_net.A[source_node_num, base_idx] = 1
    return clause_fn

def make_square_clauses(self_net, nodes_to_clauses):
    for clause in nodes_to_clauses:
        while len(clause) < self_net.max_literals:
            clause.append(clause[-1] if clause else 0)

def get_index_dtype(max_n):
    if max_n < 256: return cp.uint8
    if max_n < 65536: return cp.uint16
    return cp.uint32

def read_line(file, loop):
    line = file.readline()
    if loop > 1000000: sys.exit("Hit an infinite loop reading net file")
    return None if (not line or len(line) == 0) else line

def get_encoding(file, net_file):
    ext = net_file.split('.')
    return 'bnet' if ext[-1] == 'bnet' else file.readline().replace('\n', '')
