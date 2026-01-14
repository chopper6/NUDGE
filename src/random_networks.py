'''
random network generation

'''

import numpy as np
import sys, math
import networkx as nx, random as rd
import logic
import util, plot
from time import time
from copy import deepcopy

import matplotlib
from matplotlib import rcParams, cm
rcParams['font.family'] = 'serif'
from matplotlib import pyplot as plt


#######################

# some default parameters, if needed

NET_PARAMS = {
		'n': int(2**3),
		'edges_per_node': 2, #1.7,
		'random_num_edges': False, # instead of 'edges_per_node', uniform random in (2,n)
		'enforce_edge_ratio':True, # maintain exact edges_per_node
		'enforce_all_edges':True, # logic is constructed such that each input matters
		'graphType': 'ER', # options: ER, SF, PLT, WS, EBA
		#'EBA' has scale free in- and out-degs, whereas SF has only out-degs
		'preferential_edges': False, # rm and adding edges prop to their degree
		'monotonic': False,
		'method': 'TT', 
		# METHODS:
		#	'TT' generates a random truth table, slowest but generally the best. Not implemented with 'monotonic'. Typically fails for n>8
		#	'+1' generates one clause at a time, until the constraints are satisfied
		#	'kmaxTT' generates 'max_clauses' number of random clauses
		#  	'kTT' generates a number of random clauses in [1, max_clauses]
		#  	'kTTpartial' as KTT, except not all nodes are necessarily included in each initial clause
		'max_clauses': 4,
		'enforce_io':True,
		'num_in': 2, 
		'num_out': 2,
		'oneWCC': True, # single weakly connected component
		'oneWCCfromSCC': False, # first build oneSCC, then add inputs and outputs
									# note that total number of nodes will be 'n'+'num_in'+'num_out' in this case
		'oneSCC': False, # single strongly connected component

		# These determine the output when "display_net()" function is called
		'plot_degree_distrib':0, 
		'plot_topology':0,
		'print_info':0,

		'approx':False,
	}

#######################

def single_random_function_example():
	params = {
		'approx': False,
		'monotonic': False, # must be False for 'TT' method
		'method': 'kTTpartial',
		'max_clauses': 10, #only used for 'kmaxTT', 'kTT', and 'kTTpartial' methods
		'enforce_all_edges': True
	}
	# node name must be strings
	# string name cannot be a number (causes problems when reducing the function with pyeda)
	inputs = ['A','B','C','D','E','F','G','H','I','J']
	node = 'Y' # this doesn't really matter, but if no inputs, then function will be ['Y']
	fn = random_logic_node(params, node, inputs)
	print("final function is:",fn)



#######################


def logic_graph(params, verbose=False):
	# assigns random logic to each node, assuming each edge is significant
	# to do so, generates random truth table, builds logic using espresso
	# if an edge is irrelevant (for ex AB + B => B), then rebuilds truth table

	assert(params['method']!='+1' or params['enforce_all_edges']) # if using sequential addition of input clauses ('+1'), must enforce all edges
	assert(~params['oneSCC'] or params['num_in']==params['num_out']==0 or ~params['enforce_io'])

	if params['random_num_edges']:
		params['edges_per_node'] = rd.randint(2,params['n']) # unfortunately with only 1 edge_per_node can get infinite loop too easily

	G_topology = base_graph(params)
	if verbose and params['print_info']:
		print("\ninitial network:")
		display_net(G_topology, params)

	G, F, fail = constrain_and_add_logic(G_topology, params)

	if verbose and params['print_info']:
		print("\nfinal network:")
		display_net(G_topology, params)

	return G, F, fail

def constrain_and_add_logic(G_topology, params):
	G, fail = fit_constraints(G_topology, params)
	if fail:
		return None, fail
	F = random_logic(G_topology, params)
	return G, F, fail


def display_net(G, params):	
	if params['print_info']:
		print("\tedge:node = ", len(G.edges())/len(G.nodes()))
		print('\t#nodes=',len(G.nodes()))
		degcent, betcent = round(np.mean(list(nx.degree_centrality(G).values())),5), round(np.mean(list(nx.betweenness_centrality(G).values())),5)
		print('\tavg centralities: degree, btwn =', degcent, ',',betcent)
	if params['plot_topology']:
		plt.figure(figsize=(14, 8))
		nx.draw(G) #, pos=nx.spring_layout(G), with_labels = True)
		plt.show()
	if params['plot_degree_distrib']:
		plot.nx_plot(G)


def base_graph(params):
	# types: fast_gnp_random_graph(n, p), watts_strogatz_graph(n, k), barabasi_albert_graph(n, m), random_regular_graph(d, n)
	#	reminder use create_using=nx.DiGraph() or direct=True for all
	#		some do not have this arg, so get edges, then manually assign direction
	# other weird graph types could add: lobster, caterpillar (poss good for debug?), random_powerlaw_tree
	
	n, graphType, edges_per_node = params['n'], params['graphType'], params['edges_per_node']

	pr_edge = edges_per_node/n
	if graphType=='ER':
		G=nx.gnp_random_graph(n,pr_edge,directed=True)
	elif graphType=='SF':
		G = nx.scale_free_graph(n) 
		#G = nx.Graph(G) # if want both in- and out- degs to be power laws lol
		G = nx.DiGraph(G) # wooow..otherwise adds duplicate edges wtf

	elif graphType=='EBA': # extended barabasi albert
		G = nx.extended_barabasi_albert_graph(n, 1, .4, .2)
		# (n,m,p,q) where p is pr rd edge, q is pr rd rewires, m is number of each
		# p=q=0 is same as SF (except undirected initially)
		G = undirected_to_directed(G)
	elif graphType=='WS': # watts-strogatz
		G = nx.connected_watts_strogatz_graph(n, 4, .2, tries=10000) 
		# (n,k,p) where k is #nghs to connected to and p is pr of rewiring
		G = undirected_to_directed(G)
	elif graphType=='PLT': 
		G = nx.random_powerlaw_tree(n, gamma=3,tries=10000)
		G = undirected_to_directed(G)

	labels = alpha_labels(n)
	G = nx.relabel_nodes(G, dict(zip(range(n), labels)))
	return G

def alpha_labels(n):
    def to_alpha(k):
        s = ""
        k += 1
        while k:
            k, r = divmod(k - 1, 26)
            s = chr(65 + r) + s
        return s
    return [to_alpha(i) for i in range(n)]

def fit_constraints(G, params):

	if params['enforce_io']:
		fail = enforce_io(G, params)
		if fail:
			return None, True
	if params['enforce_edge_ratio']:
		fail = enforce_edge_ratio(G, params)
		if fail:
			return None, True
	fail = enforce_single_CC(G, params) # checks params, so will ignore if not oneSCC or oneWCC
	if fail:
		return None, True
	if params['oneWCCfromSCC']:
		add_io(G, params)
		n = params['n'] + params['num_in'] + params['num_out']
	else:
		n= params['n']

	assert(len(G.nodes) == n)
	label_list  = util.alphabet_labels(n)
	labels = {i:label_list[i] for i in range(n)}
	#labels = {i:int_to_str(i) for i in range(params['n'])}
	G = nx.relabel_nodes(G,labels)
	return G, False


def random_logic(G, params):
	F = {}
	for node in G.nodes():
		in_edges = list(G.in_edges(node))
		inputs = [in_edges[i][0] for i in range(len(in_edges))]
		F[node] = random_logic_node(params, node, inputs)
	return F



def random_logic_node(params, node, inputs):
	# uses params keys: monotonic, method, max_clauses, enforce_all_edges
	loop=0
	raw_clauses = []
	if params['monotonic']:
		signs = np.random.choice([0,1],len(inputs))
	else:
		signs = None
	if len(inputs)==0:
		return [[node]]
	while True and len(inputs):
		if params['method']=='TT':
			assert(not params['monotonic']) # monotonic not implemented for TT method
			raw_clauses = rd_TT(inputs)
		elif params['method'] in ['kTT','kTTpartial']:
			for _ in range(np.random.randint(1,min(params['max_clauses'], 2**(len(inputs)))+1)): # note +1 since return int in [low, high)
				if params['method'] == 'kTTpartial':
					clause = rd_clause(inputs,params['monotonic'],partial=True, signs=signs)
				else:
					clause = rd_clause(inputs,params['monotonic'],partial=False, signs=signs)
				if clause not in raw_clauses:
					raw_clauses += [clause]
		elif params['method']=='kmaxTT':
			for _ in range(min(params['max_clauses'], 2**(len(inputs))+1)):					
				clause = rd_clause(inputs,params['monotonic'],partial=False, signs=signs)
				if clause not in raw_clauses:
					raw_clauses += [clause]
		elif params['method']=='+1':
			clause = rd_clause(inputs,params['monotonic'],partial=True, signs=signs)
			if clause not in raw_clauses:
				raw_clauses += [clause]
			#print('rawc=',raw_clauses,'inputs=',inputs,' newclause=',clause)
			
		else:
			assert(0) # unrecognized 'method' in params 
		assert(len(raw_clauses)>0)
		fn = logic.from_pyEda(logic.reduce(params, logic.to_pyEda(raw_clauses)))

		if fn not in [True, False]: # cannot be constant or input
			assert(params['enforce_all_edges']) # have not updated alternative, and reqs passing nx G as arg
			if params['enforce_all_edges'] and all_edges_included(inputs, fn):
				return fn
			#elif not params['enforce_all_edges']:
			#	if len(fn)>0:
			#		rm_redundant_edges(G,inputs,fn,node)
			#		break
		if loop>10**4:
			assert(0) # seems to be infinite loop 
		loop+=1


############# Constraints ################


def enforce_io(G, params):
	# just removes and adds edges to remove or add inputs (likewise for outputs)
	# no attempt to try and preserve number of edges...but probably should consider
	# could subsequently add edges (in a SF or ER manner), being careful not to add in edges to inputs or out edges from outputs
	ignoreIO_for_new_edges = True 
	num_in, num_out = params['num_in'], params['num_out']
	if params['oneWCCfromSCC']:
		num_in=num_out=0
	loop=0
	while True:
		curr_outputs = [node for node, out_degree in G.out_degree() if out_degree == 0]
		curr_inputs = [node for node in G.nodes if all(src == node for src, _ in G.in_edges(node))]
		curr_both = list(set(curr_outputs) & set(curr_inputs)) # note no nodes that are both in and out_degrees allowed

		excess_inputs = len(curr_inputs) - num_in 
		excess_outputs = len(curr_outputs) - num_out

		if excess_inputs == 0 and excess_outputs == 0 and len(curr_both)==0:
			break 

		if len(curr_both)>0:
			for node in curr_both:
				if excess_inputs >= excess_outputs:
					fail = add_rd_edge(G, params, target=node, ignoreIO = ignoreIO_for_new_edges) 
				else:
					fail = add_rd_edge(G, params, source=node, ignoreIO = ignoreIO_for_new_edges) 

				if fail:
					return fail

		if excess_inputs < 0: # add inputs by removing edges
			rd_indices = np.random.randint(len(G.nodes),size=abs(excess_inputs))
			for indx in rd_indices:
				while indx in curr_inputs or indx in curr_outputs: 
					indx = np.random.randint(len(G.nodes)) 
				node = list(G.nodes)[indx]
				G.remove_edges_from(list(G.in_edges(node)))
		elif excess_inputs > 0: # rm inputs by adding edges
			input_indices = np.random.randint(len(curr_inputs),size=excess_inputs) 
			for indx in input_indices: 
				source = curr_inputs[indx]
				fail = add_rd_edge(G,  params, source=source, ignoreIO = ignoreIO_for_new_edges) 
				if fail:
					return fail

		if excess_outputs < 0: # add inputs by removing edges
			rd_indices = np.random.randint(len(G.nodes),size=abs(excess_outputs))
			for indx in rd_indices:
				while indx in curr_inputs or indx in curr_outputs: 
					indx = np.random.randint(len(G.nodes)) 
				node = list(G.nodes)[indx]
				G.remove_edges_from(list(G.out_edges(node)))
		elif excess_outputs > 0: # rm inputs by adding edges
			output_indices = np.random.randint(len(curr_outputs),size=excess_outputs) 
			for indx in output_indices: 
				target = curr_outputs[indx]
				fail = add_rd_edge(G,  params, target=target, ignoreIO = ignoreIO_for_new_edges)  
				if fail:
					return fail
		loop+=1
		if loop>10000:
			#assert(0) # seems to be an infinite loop \:
			return True 
	return False

def enforce_edge_ratio(G, params):
	edges_per_node = params['edges_per_node']
	target_num_edges = int(round(len(G.nodes())*edges_per_node))
	
	diff = target_num_edges - len(G.edges())

	if diff > 0: # need more edges
		for i in range(diff):
			fail = add_rd_edge(G, params)
			assert(not fail) # TODO should return and handle instead
	elif diff < 0:
		for i in range(abs(diff)):
			fail = rm_rd_edge(G,params)
			if fail:
				return fail

	assert(target_num_edges == len(G.edges()))
	fail = False 
	return fail

def enforce_single_CC(G,params):
	# typically AFTER enforcing edge ratio and IO, so also should maintain those

	if params['oneSCC'] or params['oneWCCfromSCC']:
		fn = nx.strongly_connected_components
	elif params['oneWCC']:
		fn = nx.weakly_connected_components
	else:
		return # skip
	loop=0
	while len(list(fn(G)))>1:
		CCs = list(fn(G))
		#print("CCs=",CCs,'\nedges',G.edges())
		source, target = rd.choice(list(CCs[0])), rd.choice(list(CCs[1]))
		if params['oneSCC'] or params['oneWCCfromSCC']:
			while nx.has_path(G,source,target):
				source, target, loop = repick(CCs, loop)
		elif params['oneWCC'] and params['num_in']+params['num_out'] > 0:
			while G.out_degree(source)==0 or G.in_degree(target)==0:
				source, target, loop = repick(CCs, loop)
		G.add_edge(source,target)
		fail = rm_rd_edge(G,params)
		if fail:
			return fail 
	fail = False 
	return fail

def add_io(G, params):
	assert(params['oneWCCfromSCC'])
	num_edges = max(1,params['edges_per_node'])
	loopy=0

	for i in range(params['num_in']):
		node = i+params['n']
		G.add_node(node) 
		for j in range(num_edges):
			cont=True 
			while cont:
				target = np.random.randint(params['n'])
				cont = (node, target) in G.out_edges(node)
				G.add_edge(node,target)
				loopy= util.loop_debug(params['n'], loopy, expo=4)
	for i in range(params['num_out']):
		node = i+params['n']+params['num_in']
		G.add_node(node) 
		for j in range(num_edges):
			cont=True 
			while cont:
				source = np.random.randint(params['n'])
				cont = (source, node) in G.in_edges(node)
				G.add_edge(source, node)
				loopy= util.loop_debug(params['n'], loopy, expo=4)



def repick(CCs, loop):
	indices = np.random.randint(len(CCs),size=2)
	source, target = rd.choice(list(CCs[indices[0]])), rd.choice(list(CCs[indices[1]]))
	#print('s,t=',source,target)
	loop+=1
	if loop>10**2:
		assert(0) # seems to be an infinite loop \:
	return source, target, loop


############# Random Selection ################


def add_rd_edge(G, params, source=None, target=None, ignoreIO = False):
	# will make sure 
	# 1) edge does not already exist
	# 2) source is not an output node (unless source given)
	# 3) target is not an input node (unless source given)
	# "ignoreIO" will skip 2-3

	# to add: preferential-attchmt method of edges
	#	just add list of in_degs for pr of selection

	loop=0
	orig_source, orig_target = source, target
	nodes = list(G.nodes())
	if params['preferential_edges']: 
		in_degs, out_degs = G.in_degree(nodes), G.out_degree(nodes)
		target_p = np.array([in_degs[nodes[i]] for i in range(len(nodes))])
		source_p = np.array([out_degs[nodes[i]] for i in range(len(nodes))])
		if not ignoreIO: 
			#such that all nodes have at least some pr
			target_p +=1
			source_p +=1
		target_p = target_p / np.sum(target_p)
		source_p = source_p / np.sum(source_p)
	else:
		target_p=source_p=None

	cont=True
	while cont==True:

		if source is None:
			source = np.random.choice(nodes,p=source_p)
			while len(G.out_edges(source))==0 and not ignoreIO: # output node
				source = np.random.choice(nodes,p=source_p)
		if target is None:
			target = np.random.choice(nodes,p=target_p)
			while len(G.in_edges(target))==0 and not ignoreIO: # output node
				target = np.random.choice(nodes,p=target_p)

		if (source,target) not in G.edges():
			G.add_edge(source, target)
			cont=False # just break should suffice but anyhow
			break
		else:
			source, target = orig_source, orig_target

		loop+=1
		if loop>len(G.nodes)*100:
			#assert(0) # seems to be an infinite loop \:
			return True # ie fail 
	return False


def rm_rd_edge(G, params):
	# constrained to not change #inputs and outputs

	edges = list(G.edges())
	if params['preferential_edges']:
		# (being very careful to maintain the order such that p[i] corresponds to edges[i])
		in_degs = G.in_degree([edges[i][1] for i in range(len(edges))])
		target_p = np.array([in_degs[edges[i][1]] for i in range(len(edges))])-1

		# note -1 since don't want to changeIO (i.e. must be at least 2 edges to rm 1)
		out_degs = G.out_degree([edges[i][0] for i in range(len(edges))])
		source_p = np.array([out_degs[edges[i][0]] for i in range(len(edges))])-1

		target_p = target_p / np.sum(target_p)
		source_p = source_p / np.sum(source_p)

		#p = 1-target_p
		#p = (1-target_p)*(1-source_p) # pref to rm edges with low source AND low target
		p = 1-target_p*source_p # pref to rm edges with low source OR low target
		p = p / np.sum(p)
	else:
		p = None  
	edge_indx = np.random.choice([i for i in range(len(edges))], p=p)
	edge = edges[edge_indx]
	loop=0
	while len(G.out_edges(edge[0]))==1 or len(G.in_edges(edge[1]))==1:
		edge_indx = np.random.choice([i for i in range(len(edges))], p=p)
		edge = edges[edge_indx]

		loop+=1
		if loop>len(G.nodes)*100:
			fail = True 
			return fail

	G.remove_edge(edge[0],edge[1])
	fail=False
	return fail


def rd_clause(nodeNames,monotonic,partial=False, signs=None):
	k=len(nodeNames)
	nodeNames = nodeNames.copy()
	not_str = '!'
	nodeNames = [name.replace(not_str,'') for name in nodeNames]

	dtype = util.get_uint_dtype(2**k)
	rdint = np.random.randint(1,2**k, dtype=dtype) # note rd choice[0,1] would use rd in reals anyway

	if signs is None:
		signs = util.int_to_bool(rdint,k) # note that this is reverse order
		assert(not monotonic)
	assert(len(signs)<=k)

	clause = []
	if partial: 
		include_this_input = util.int_to_bool(rdint,k) 
		for j in range(len(signs)):
			if include_this_input[j]:
				if signs[j]:
					clause+=[str(nodeNames[j])]
				else:
					clause+=[not_str+str(nodeNames[j])]	
	else:
		assert(not monotonic) # need to clean this up, but using all nodes only makes sense if fn is non-monotonic
		for j in range(len(signs)):
			if signs[j]:
				clause+=[str(nodeNames[j])]
			else:
				clause+=[not_str+str(nodeNames[j])]
	assert(len(clause)>0)
	return clause

def rd_TT(nodeNames):
	# for each binary str in 2^k, rd flip and add if true
	# i.e. unreduced net form

	fn = []
	k=len(nodeNames)
	not_str = '!'

	while len(fn)==0:
		if k<64:
			rdflips = np.random.rand(2**k) # note rd choice[0,1] would use rd in reals anyway
		else:
			rdflips = int(np.random.uniform(2**k))
		for i in range(1,2**k):
			if rdflips[i] > .5:
				clause = []
				bools = util.int_to_bool(i,k) # note that this is reverse order
				#print(len(bools),k,i, bools)
				assert(len(bools)<=k)
				for j in range(len(bools)):
					if bools[j]:
						clause+=[str(nodeNames[j])]
					else:
						clause+=[not_str+str(nodeNames[j])]
				assert(len(clause)>0)
				fn += [clause]
	return fn

############# Util ################


def all_edges_included(inputs, fn):
	which = which_edges_included(inputs, fn)
	return np.all(np.array(which))

def which_edges_included(inputs, fn, not_str='!'):
	inputs = [inp.replace(not_str,'') for inp in inputs]
	included = {inp:False for inp in inputs}
	for clause in fn:
		for ele in clause:
			ele = ele.replace(not_str,'')
			included[ele] = True 
	return [x for x in included.values()]

def rm_redundant_edges(G,inputs,fn,target_node):
	which = which_edges_included(inputs, fn)
	for i in range(len(which)):
		if not which[i]:
			G.remove_edge(inputs[i],target_node)

def undirected_to_directed(G):
	to_rm = []
	G = nx.DiGraph(G) # for each undirected edges makes 2 directed edges
	for edge in G.edges():
		if edge[0] < edge[1]:
			if rd.random()<.5:
				to_rm += [edge]
			else:
				to_rm += [(edge[1],edge[0])]
	G.remove_edges_from(to_rm)
	return G


def int_to_str(num):
   bin_str = format(num, '04b')
   bin_str = bin_str.replace('0','a')
   bin_str = bin_str.replace('1','b')
   return bin_str


##############################################################


if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python3 rdze.py [build | function]")
	if sys.argv[1] == 'build':
		params = NET_PARAMS
		G, F, fail =logic_graph(params, verbose=True)
		print("generated network with logic:")
		for k in F.keys():
			print(k,'\t',logic.F_to_str(F[k]))
	elif sys.argv[1] == 'function':
		single_random_function_example()
	else:
		assert(0) # unknown 3rd argument