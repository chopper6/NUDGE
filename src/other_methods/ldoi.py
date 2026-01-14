'''
Logical domain of influence (LDOI), following:
Yang, Gang, Jorge Gómez Tejeda Zañudo, and Réka Albert. "Target control in logical models using the domain of influence of nodes." Frontiers in physiology 9 (2018): 454.

The implementation optionally parallelizes many runs of LDOI with the Cupy library on GPU.
'''

import util, RSC, logic
CUPY, cp = util.import_cp_or_np(try_cupy=1)

def find_controllers(params, Forig):
	if not params['trim_ldoi']:
		approx = params['approx']; params['approx'] = False
		F = RSC.load_F(params)[0]
		params['approx'] = approx
	else:
		F = Forig
	params['mutations'] = {}

	ldoi_dims = 1 if len(F) > params['ldoi_pair_cap'] else 2
	#self_drivers, nodeNames = run_ldoi_parallel(F, params, 1, max_steps=1) # ldoi specifically for 1-step
	self_drivers, nodeNames = run_ldoi(F, params, ldoi_dims) # regular ldoi
	self_drivers = raise_dim(self_drivers)
	#self_drivers, nodeNames = run_ldoi(F, params, ldoi_dims, max_steps=1) 
	self_driven = calc_ldoi_self_driven(self_drivers)
	drives, nodeNames = run_ldoi(F, params, ldoi_dims)
	target = nodeNames.index(params['target'])
	return extract_controllers(self_driven, drives, target, nodeNames)


def run_ldoi(F, params, ldoi_dims, max_steps=None):
	if ldoi_dims==1:
		return run_ldoi_mixed(F, params, ldoi_dims, max_steps=max_steps)
	else:
		return run_ldoi_parallel(F, params, ldoi_dims, max_steps=max_steps)

########################################################################################

def extract_controllers(self_driven, drives, target, nodeNames):
	n2 = len(drives)
	assert(n2 == len(nodeNames))
	c1_all = control_condition_1(self_driven, drives, target)
	c2_all = control_condition_2(self_driven, drives, target, build_exclude_cols(n2, 'singles'))
	singles_mask = cp.diag(c1_all) | cp.diag(c2_all)
	if bool(singles_mask.any()):
		idxs = (cp.where(singles_mask)[0].get().tolist() if CUPY else cp.where(singles_mask)[0].tolist())
		return [[nodeNames[int(i)]] for i in idxs]

	I = cp.eye(n2, dtype=cp.bool_); tri = cp.triu(cp.ones((n2, n2), dtype=cp.bool_), k=1)
	d1 = (c1_all & ~I) & tri
	d2 = (control_condition_2(self_driven, drives, target, build_exclude_cols(n2, 'doubles')) & ~I) & tri
	doubles_mask = d1 | d2
	if bool(doubles_mask.any()):
		pairs = (cp.argwhere(doubles_mask).get().tolist() if CUPY else cp.argwhere(doubles_mask).tolist())
		return [[nodeNames[int(i)], nodeNames[int(j)]] for i, j in pairs if i < j]
	return [[]]

def raise_dim(L):
	if L.ndim == 2:
		n2 = L.shape[0]
		L3 = cp.zeros((n2, n2, n2), dtype=L.dtype)
		ii = cp.arange(n2); L3[ii, ii, :] = L
		return L3
	return L

def calc_ldoi_self_driven(self_drivers):
	n2 = len(self_drivers)
	idx = cp.arange(n2)
	part1 = self_drivers[idx[:, None], idx[None, :], idx[:, None]]
	part2 = self_drivers[idx[:, None], idx[None, :], idx[None, :]]
	return part1 & part2

def control_condition_1(self_driven, L, target):
	return (self_driven & L[:, :, target])

def control_condition_2(self_driven, L, target, exclude_cols):
	n2 = L.shape[0]
	S = (self_driven & L[:, :, target]).astype(cp.uint8)
	L2 = L.reshape(-1, n2).astype(cp.uint8)
	V = L2  #* exclude_cols if (exclude_cols is not None) else L2
	U = V @ S
	T = (U * V).sum(axis=1) > 0
	return T.reshape(n2, n2)

def build_exclude_cols(n2, mode):
	M = cp.ones((n2*n2, n2), dtype=cp.uint8)
	i_idx = cp.repeat(cp.arange(n2), n2)
	j_idx = cp.tile(cp.arange(n2), n2)
	if mode == 'singles':
		d = (i_idx == j_idx); M[d, i_idx[d]] = 0
	elif mode == 'doubles':
		d = (i_idx != j_idx); M[d, i_idx[d]] = 0; M[d, j_idx[d]] = 0
	else:
		raise ValueError("mode must be 'singles' or 'doubles'")
	return M

def run_ldoi_mixed(F, params, dims, inputs=None, verbose=False, max_steps=None):
	n2 = len(F)
	names = list(F.keys())
	L = cp.zeros((n2, n2, n2), dtype=bool)
	original_mutations = params['mutations']
	for i in range(n2):
		if '!' in names[i]:
			name = names[i].replace('!','')
			val = 0
		else:
			name = names[i]
			val = 1
		params['mutations'] = {**original_mutations, **{name:val}}
		Li, nodeNames = run_ldoi_parallel(F, params, dims, inputs=inputs, verbose=verbose, max_steps=max_steps)
		L[i] = Li
	params['mutations'] = original_mutations
	return L, nodeNames

def run_ldoi_parallel(F, params, dims, inputs=None, verbose=False, max_steps=None):
	loop, drivers, cont, pinned_nodes = init_ldoi(F, params, dims, inputs=inputs)
	i = 0
	while cont:
		cont = step_ndim(drivers, verbose=verbose)
		loop = util.loop_debug(len(F), loop, expo=4)
		i += 1
		if (max_steps is not None) and (i == max_steps):
			break
	drivers.convert_solutions()
	return drivers.L, drivers.nodeNames

def init_ldoi(F, params, dims, inputs=None):
	if inputs is not None and len(inputs) == 0: inputs = None
	pinned_nodes = generate_base_pinned_nodes(F, params, inputs=inputs)
	loop = 0
	drivers = Drivers(F, params, pinned_nodes, dims, inputs=inputs)
	cont = True
	return loop, drivers, cont, pinned_nodes

def generate_base_pinned_nodes(F, params, inputs=None):
	base_pinned_nodes = []
	if 'mutations' in params.keys():
		for name in params['mutations']:
			base_pinned_nodes += [a_base_pinned_node(F, params, name, params['mutations'][name])]
	if inputs is not None:
		for i in range(len(inputs)):
			base_pinned_nodes += [a_base_pinned_node(F, params, params['inputs'][i], inputs[i])]
	return base_pinned_nodes

def a_base_pinned_node(F, params, name, val):
	if val == 0:
		return list(F.keys()).index('!' + name)
	else:
		return list(F.keys()).index(name)

def step_ndim(drivers, verbose=False):
	drivers.L_next = drivers.L.copy()
	drivers.L_next = propagate_matrix(drivers, drivers.L_next)
	cont = cp.any(drivers.L != drivers.L_next)
	drivers.L = drivers.L_next.copy()
	return cont

def propagate_matrix(drivers, M):
	P = ((M | drivers.pins) & (~drivers.pins_compls)).astype(drivers.intdtype)
	clauses_on = (P @ drivers.nodes2clauses) >= drivers.composite_counts
	M = M | ((clauses_on @ drivers.clauses2nodes) & (~drivers.pins_compls))
	return M

class Drivers:
	def __init__(self, F, params, pinned_nodes, dims, inputs=None):
		self.n2 = len(F); self.n = int(len(F)/2); self.dims = dims
		self.params = params; self.inputs = inputs; self.F = F
		self.build_node_names()
		self.build_clause_mapping()
		self.shape = tuple([self.n2 for _ in range(dims+1)])
		self.pins = cp.zeros(self.shape, dtype=bool)
		self.diag = cp.arange(self.n2)
		for i in range(dims):
			ind = [slice(None) for _ in range(dims+1)]
			ind[i] = ind[-1] = self.diag
			self.pins[tuple(ind)] = 1
		self.base_pins = cp.zeros(self.n2, dtype=bool)
		for nodeNum in pinned_nodes:
			self.pins[..., nodeNum] = 1
			self.pins[..., (nodeNum + self.n) % self.n2] = 0
			self.base_pins[nodeNum] = 1
		self.pins_compls = cp.roll(self.pins, self.n, axis=-1)
		self.L = cp.zeros(self.shape, dtype=bool)
		self.L_next = cp.zeros(self.shape, dtype=bool)
		self.self_contra = cp.any(self.pins & self.pins_compls, axis=-1)
		self.pins[self.self_contra] = 0
		self.pins_compls[self.self_contra] = 0

	def convert_solutions(self):
		F_keys = list(self.F.keys())
		hits = cp.argwhere(self.L); hits = hits.get() if CUPY else hits
		soln = {}
		for *driver_idxs, driven_idx in hits:
			key = tuple(dict.fromkeys(self.nodeNames[int(i)] for i in driver_idxs))
			val = F_keys[int(driven_idx)]
			soln.setdefault(key, []).append(val)
		for k, v in soln.items():
			seen, uniq = set(), []
			for x in v:
				if x not in seen:
					seen.add(x); uniq.append(x)
			soln[k] = uniq
		self.solutions = soln

	def build_clause_mapping(self):
		all_clauses, Flist, maxlng = [], {}, 0
		for k in self.nodeNames:
			Flist[k] = logic.str_to_F(self.F[k])
			for clause in Flist[k]:
				if clause not in all_clauses:
					all_clauses += [clause]
					maxlng = max(maxlng, len(clause))
		self.N = len(all_clauses); self.Flist = Flist
		self.intdtype = util.get_int_dtype(maxlng)
		self.clauses2nodes = cp.zeros((self.N, self.n2), dtype=bool)
		self.nodes2clauses = cp.zeros((self.n2, self.N), dtype=self.intdtype)
		for i in range(self.N):
			for j in range(self.n2):
				if all_clauses[i] in Flist[self.nodeNames[j]]: self.clauses2nodes[i, j] = 1
				if self.nodeNames[j] in all_clauses[i]: self.nodes2clauses[j, i] = 1
		self.composite_counts = cp.sum(self.nodes2clauses, axis=0)

	def build_node_names(self):
		self.nodeNames = []
		for k in self.F.keys():
			if '!' not in k: self.nodeNames += [k]
		for k in self.F.keys():
			if '!' in k: self.nodeNames += [k]

#################################################################################

if __name__ == "__main__":
	# for debugging
	# load G and some default params? does LDOI take any params now?
	dims = 2
	F = {
		'X':'(X)',
		'Y':'(X)',
		'!X':'(!X)',
		'!Y':'(!X)',
	}
	F2 = {
		'A':'A',
		'B':'A | B',
		'C':'!A & B',
		'!A':'!A',
		'!B':'!A & !B',
		'!C':'A | !B',
	}
	params = {
		'mutations':{},
	}

	drivers, node_names = run_ldoi(F2, params, dims)
	print(node_names,'\n\n',drivers.astype(int))