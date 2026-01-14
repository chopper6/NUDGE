'''
Find the mechanism responsible for the effect of each controller

'''
import RSC, logic, plot, util
import networkx as nx
import sys, itertools
import numpy as np

def find_mechanism(params, F, all_controllers, controller, cnt_fns, drives_self_driven, self_driven, plot_it=False):

	if (all_controllers[0] == ['0']):
		print("No mechanism as desired attractor is not possible (terminal logic conjunction = False)")
		return 'NA' # no mechanism desired attractor is not possible

	data, mech_graph, self_driven_by_controller, original_target = init_mechanism(params, F, all_controllers, controller, cnt_fns, drives_self_driven, self_driven)

	controller_to_self_driven(params, F, self_driven_by_controller, controller, data, mech_graph)
	self_driven_to_self_driven(params, F,self_driven_by_controller, data, mech_graph)
	self_driven_to_target(params, F,self_driven_by_controller, original_target, data, mech_graph)

	data['graph'] = mech_graph
	if plot_it:
		#print_edge_details(mech_graph)
		plot.mechanism_graph(params, mech_graph) 
	return data

def print_edge_details(G):
	print('\nedges of mechanism graph:')
	for u, v in G.edges():
		d = G.get_edge_data(u, v) or {}
		t = d.get("detail", "")
		if t is not None:
			print("\t{}\t{}\t'{}'".format(u, v, t))
		else:
			print("\t{}\t{}".format(u, v))

#####################################################################################################

def init_mechanism(params, F, all_controllers, controller, cnt_fns, drives_self_driven, self_driven):

	assert(controller in all_controllers)
	mech_graph = nx.DiGraph()
	controller.sort()
	
	data = {'graph':mech_graph, 'controller -> self-driven':[],'self-driven -> self-driven':[],'self-driven -> target':[]}
	name, label = get_name_and_label([controller])
	mech_graph.add_node(name,label=label,target=False, controller=True, self_driven=False)

	indx = drives_self_driven[all_controllers.index(controller)] & self_driven 
	self_driven_by_controller=[cnt_fns[i] for i in range(len(indx)) if indx[i]]
	self_driven_by_controller = only_subspaces(params, self_driven_by_controller)
	
	original_target = params['target']

	return data, mech_graph, self_driven_by_controller, original_target 

def controller_to_self_driven(params, F, self_driven_by_controller, controller, data, mech_graph):
	for self_driven_fn in self_driven_by_controller:
		self_driven = logic.from_pyEda(self_driven_fn)
		params['target'] = logic.F_to_str(self_driven).replace('(','').replace(')','')
		for clause in self_driven:
			name, label = get_name_and_label([clause])
			if (name in mech_graph):
				mech_graph.nodes[name]['self_driven']=True
			else:
				mech_graph.add_node(name,label=label,target=False, controller=False, self_driven=True)
		name, label = get_name_and_label(self_driven)
		seq = get_rsc_seq(params, F)
		if self_driven != controller:
			if not is_found(params, [controller], seq[-1]):
				data['controller -> self-driven'] += [(name, 0)]
			else:
				max_order_seen, finished = multipath(params,  F, mech_graph, [controller], seq, 0)
				data['controller -> self-driven'] += [(name, max_order_seen)]
		
def self_driven_to_self_driven(params, F, self_driven_by_controller, data, mech_graph):
	for self_driven_fn in self_driven_by_controller:
		self_driven = logic.from_pyEda(self_driven_fn)
		params['target'] = logic.F_to_str(self_driven).replace('(','').replace(')','')
		name, label = get_name_and_label(self_driven)
		seq = get_rsc_seq(params, F)

		if not is_found(params, self_driven, seq[-1]):
			data['self-driven -> self-driven'] += [(name, 0)]
		else:
			max_order_seen, finished = multipath(params,  F, mech_graph, self_driven, seq, 0)
			data['self-driven -> self-driven'] += [(name, max_order_seen)]

def self_driven_to_target(params, F, self_driven_by_controller, original_target, data, mech_graph):

	params['target'] = original_target
	name, label = get_name_and_label(logic.str_to_F(original_target))
	if (name in mech_graph): 
		mech_graph.nodes[name]['target']=True
	else:
		mech_graph.add_node(name,label=label,target=True, controller=False, self_driven=False)
	seq = get_rsc_seq(params, F)
	#for i in range(len(seq)):
	#	print("step",i,':',seq[i],'\n')
	
	for self_driven_fn in self_driven_by_controller:
		self_driven = logic.from_pyEda(self_driven_fn)
		name, label = get_name_and_label(self_driven)

		if not is_found(params, self_driven, seq[-1]):
			data['self-driven -> target'] += [(name, 0)]
		else:
			max_order_seen, finished = multipath(params,  F, mech_graph, self_driven, seq, 0) #trim via seq[:-1*period]?
			data['self-driven -> target'] += [(name, max_order_seen)]


#####################################################################################################

def get_rsc_seq(params, F):
	fn, fn_prev = RSC.build_target_function(params, F)
	seq = [sorted_clauses(logic.str_to_F(params['target']))]
	reps = 1
	if params['approx']:
		reps = params['ensemble_repeats']
	for r in range(params['max_recursion']):
		fn_next = logic.to_pyEda([['1']])
		cont= False
		for s in range(reps):
			fn2, cont2 = RSC.step(params, F, fn, fn_prev, r)
			fn_next = (fn_next) & (fn2)
			cont = cont | cont2
		fn = logic.reduce(params,fn_next)
		#print("\nget_Rsc_seq fn=",fn_next)
		seq.append(logic.from_pyEda(fn))
		if not cont:
			break
	return seq

def multipath(params, F, mech_graph, clauses, seq, i):
	# takes a set of clauses, and add to mech_graph with each, then recurse
	max_order_seen, finished = 0, True
	next_clauses = []

	#print("at seq step",i,'with',seq[len(seq)-i-1]) # assuming this is in F form
	for Ci in clauses:
		Ci_fn = logic.to_pyEda([Ci])
		candidates = seq[len(seq)-i-1]
		exclude = [False for _ in range(len(candidates))]
		for k in range(1, params['analysis_order']+1):
			next_subspaces, parent_fns, subspace_indices = get_subspaces(params, candidates, exclude, k, F)
			#print("\n\n\t at i,k=",(i,k),'next_subspaces=',next_subspaces)
			#if i<len(seq)-2:
			#	assert(0)
			max_order_seen = find_implied_subspaces(params, Ci, Ci_fn, next_subspaces, parent_fns, i, F, next_clauses, mech_graph, k, max_order_seen, subspace_indices, exclude)

	max_order_seen, finished = check_if_finished_then_recurse(params, F, mech_graph, next_clauses, seq, i, max_order_seen, finished)

	return max_order_seen, finished 

def get_subspaces(params, candidates, exclude, k, F):
	subspaces, parent_fns = [], []
	subspace_indices = [] # to know what to exclude if find something driving it
	for indices in combos(len(candidates), k):
		if all(not exclude[i] for i in indices):
			subspace = [candidates[i] for i in indices]
			parent_fn, reduction_matters = build_candidate_parent_fn(params, subspace, F)

			if (subspace not in subspaces) and ((k==1) or reduction_matters):
				#print("adding subspace:", subspace, 'with function', parent_fn)
				subspaces += subspace
				parent_fns += [parent_fn for _ in range(k)] # each clause of subspace given this function seperately
				subspace_indices += [indices for _ in range(k)]
	return subspaces, parent_fns, subspace_indices

def build_candidate_parent_fn(params, subspace, F):
	fn = logic.to_pyEda([['0']])
	for clause in subspace:
		clause_fn = logic.to_pyEda([['1']])
		for xj in clause:
			f_xj = logic.str_to_fn(F[xj])
			clause_fn = clause_fn & f_xj
		clause_fn = logic.reduce(params, clause_fn)
		fn = fn | clause_fn
	fn.simplify()
	unred_clauses = logic.from_pyEda(fn)
	parent_fn = logic.reduce(params, fn)
	red_clauses = logic.from_pyEda(parent_fn)
	if unred_clauses != red_clauses:
		reduction_matters = True 
	else:
		reduction_matters = False 

	return parent_fn, reduction_matters

def find_implied_subspaces(params, Ci, Ci_fn, next_subspaces, parent_fns, i, F, next_clauses, mech_graph, order, max_order_seen, subspace_indices, exclude):
	next_Ci = [] # next clause implied by Ci at this order
	for j in range(len(next_subspaces)):
		Cj = next_subspaces[j]
		if Cj not in [['0'],['1']]:
			if logic.implies(params, Ci_fn, parent_fns[j]):
				next_Ci += [Cj]
				if Cj not in next_clauses:
					next_clauses += [Cj]
				for indx in subspace_indices[j]:
					exclude[indx] = True
				max_order_seen = max(max_order_seen, order)
				#print("\nfinding edge label from", Cj,'and',parent_fns[j],'\n\n\n')
				edge_label = None
				#if order == 1:
				#	edge_label = None 
				#else:
				#	edge_label = logic.fn_to_str(build_candidate_parent_fn(params,[Cj],F)[0]).replace('(','').replace(')','')
				add_an_edge(mech_graph, Ci, Cj, order, edge_label)
	return max_order_seen

def check_if_finished_then_recurse(params, F, mech_graph, next_clauses, seq, i, max_order_seen, finished):
	if (len(seq)-i-1 == 0):
		finished = True 
	elif len(next_clauses) == 0: #(next_clauses[0] in [[True],[False]]):
		finished = False
	else:
		mx, fnshd = multipath(params, F, mech_graph, next_clauses, seq, i+1)
		max_order_seen = max(max_order_seen, mx)
		finished = finished & fnshd
	if not finished:
		max_order_seen = 0
	return  max_order_seen, finished


def add_an_edge(G, curr, nextt, order, detail):
	curr_name, curr_label = get_name_and_label([curr])
	next_name, next_label = get_name_and_label([nextt])
	if curr_name not in G:
		G.add_node(curr_name,label=curr_label,target=False, controller=False, self_driven=False)
	if next_name not in G:
		G.add_node(next_name,label=next_label,target=False, controller=False, self_driven=False)
	if (curr_name, next_name) not in G.edges():
		G.add_edge(curr_name, next_name,value=order, detail=detail)

def sorted_clauses(F):
	for i in range(len(F)):
		F[i] = sorted(F[i])
	return F

def get_name_and_label(fn_list):
	name, label = '', ''
	for clause in sorted(fn_list):
		i=0
		if len(name) > 0:
			name += ' | '
			label += '|'
		for ele in sorted(clause):
			if i > 0:
				name += ' & '
				label += '&'
			name += str(ele)
			label += str(ele)
			i+=1
	return name, label

def combos(maxint, size):
	return [list(c) for c in itertools.combinations(range(maxint), size)]

def only_subspaces(params, fns):
	# keep subspaces that are not satisfied (logically implied) by another subspace
	# for example, the subspace (A+B) is not kept if the subspace (B) is also considered
	keep = [not any(logic.implies(params, fns[j], fns[i]) for j in range(len(fns)) if j != i) for i in range(len(fns))]
	return [f for f, k in zip(fns, keep) if k]


def is_found(params, F, terminal_logic):
	found=True
	terminal_logic = logic.to_pyEda(terminal_logic)
	reduced_fn = logic.reduce(params, terminal_logic)
	reduced_clauses = logic.from_pyEda(reduced_fn)
	for clause in F:
		found = found & (clause in reduced_clauses)
	return found