'''
Mean field simulation of boolean networks, following:
Parmer, Thomas, Luis M. Rocha, and Filippo Radicchi. "Influence maximization in Boolean networks." Nature communications 13.1 (2022): 3457.

This implementation optionally parallelizes the simulation with the Cupy library on GPU.
'''

PARAMS = {
	'network_file': './models/toy.bnet',
	'meanfield_time_steps': 100,
	'meanfield_epsilon': 1e-3, # if the most a node changes is less than this, will exit early
	'init':{'A':1},
}

##############################################################

import sys, os, itertools, math
from timeit import default_timer as timer
from pyeda.inter import exprvars
import util, logic, RSC
from copy import deepcopy

CUPY, cp = util.import_cp_or_np(try_cupy=1)

##############################################################

def find_controllers(params, Forig):
	if not params['trim_mf']:
		approx = params['approx']
		params['approx'] = False
		F = RSC.load_F(params)[0]
		params['approx'] = approx 
	else:
		F = Forig
	mf_controllers = find_temporary_controllers(params, Fstr=F)
	return mf_controllers

def find_temporary_controllers(params, Fstr=None):
	tstart, F, K, n, kmax, Aorig, TT, M, Y, nodes = init(params, Fstr=Fstr)
	# M tracks which nodes are inputs to each other node (adjacency list as ints)
	# TT are all possible states of input rows in {0,1}^max_degree
	# Y is the value of each node at the corresponding row in the truth table
	# Aorig is the initial average value of each node
	# F is the set of functions that define the network
	# K is a mask representing in degree of each node
	# kmax is the maximum in-degree in the network
	# node is a list of node names
	# n is the number of nodes

	node_name = params['target'].replace('!','')
	target_val = int('!' not in params['target'])
	target_index = nodes.index(node_name)

	# check null control
	A, iters = core_loop(params, Aorig.copy(), TT, M, Y, K, n)
	Aavg, iters = core_loop(params, A, TT, M, Y, K, n, calc_avg=True)
	if math.isclose(Aavg[target_index], target_val, abs_tol=params['meanfield_epsilon']):
		return [['1']] # null control
	elif math.isclose(Aavg[target_index], 1-target_val, abs_tol=params['meanfield_epsilon']):
		return [['0']] # control not possible


	controllers, solo_cnt = [], []
	for i in range(n):
		for cnt_vals in [[0,0],[1,1]]:
			if good_controller(params, TT, M, Y, K, n, nodes, Aorig, target_index, target_val, i, i, cnt_vals):
				controllers += [get_controller(nodes, i, i, cnt_vals)]
				solo_cnt += [i]

	if (len(controllers) == 0) and (len(F) <= params['mf_pair_cap']):
		for i in range(n):
			if i not in solo_cnt:
				for j in range(i+1, n):
					for cnt_vals in [[0,0],[0,1],[1,0],[1,1]]:
						#if compl_index(nodes,i)!=j: # can't control both A and !A
						if good_controller(params, TT, M, Y, K, n, nodes, Aorig, target_index, target_val, i, j, cnt_vals):
							controllers += [get_controller(nodes, i, j, cnt_vals)]
	return controllers 

def good_controller(params, TT, M, Y, K, n, nodes, Aorig, target_index, target_val, i, j, cnt_vals):
	A=Aorig.copy()
	A[i] = cnt_vals[0]
	A[j] = cnt_vals[1]
	A, iters = core_loop(params, A, TT, M, Y, K, n)
	Aavg, iters = core_loop(params, A, TT, M, Y, K, n, calc_avg=True)
	assert(cp.all(Aavg <= 1.001) and cp.all(Aavg > -.0001))
	return math.isclose(Aavg[target_index], target_val, abs_tol=params['meanfield_epsilon'])

def get_controller(nodes, i, j, cnt_vals):
	#namei, vali = get_name_val(nodes,i)
	if i==j:
		#return {namei:vali}
		sign =get_sign(cnt_vals[0])
		return [sign+nodes[i]]
	else:
		#namej, valj = get_name_val(nodes,j)
		#return {namei: vali, namej: valj}
		signi, signj = get_sign(cnt_vals[0]),get_sign(cnt_vals[1])
		return [signi+nodes[i], signj+nodes[j]]

def get_sign(cnt_val):
	if cnt_val==0:
		return '!'
	else:
		return ''


##############################################################

def run(params, Fstr=None):
	# if pass Fstr, will build from that rather than params['network_file']
	# Fstr should be a dict of str corresponding to node fns,
	tstart, F, K, n, kmax, A, TT, M, Y, nodes = init(params, Fstr=Fstr)
	A, iters = core_loop(params, A, TT, M, Y, K, n)
	#finish(params, A, iters)
	return A, nodes

def core_loop(params, A, TT, M, Y, K, n, calc_avg=False):
	if calc_avg:
		Aavg = A.copy()
	for i in range(params['meanfield_time_steps']):
		Aprev = A.copy() #deepcopy(A)
		A = step(params, Aprev, TT, M, Y, K, n)
		if calc_avg:
			Aavg += A
		if stop(params, A, Aprev):
			if calc_avg:
				Aavg/=(i+2)
				return Aavg, i
			else:
				return A, i 
	if calc_avg:
		Aavg/=(i+2)
		return Aavg, i
	else:
		return A, i 

def step(params, Aprev, TT, M, Y, K, n):

	#less parallel version:
	'''
	A=cp.zeros(n)
	for i in range(n):
		print("i=",i)
		for j in range(len(TT)):
			RHS = cp.prod(K[i,:]*(cp.power(Aprev[M[i,:]],TT[j,:]))*(cp.power(1-Aprev[M[i,:]],1-TT[j,:]))+(1-K[i,:]), axis=0) 
			A[i] += Y[i,j]*RHS
			print("\tA[i]+=",Y[i,j],'*',RHS,'\nKi',K[i])
	'''
	A = cp.sum(Y * cp.prod(K[:, None, :] * cp.power(Aprev[M][:, None, :], TT[None, :, :]) * cp.power(1 - Aprev[M][:, None, :], 1 - TT[None, :, :]) + (1 - K[:, None, :]), axis=2), axis=1)
	#print("A=\n",A)
	assert(cp.all(A < 1 + 1e-6) and cp.all(A > - 1e-6))
	return A

def stop(params, A, Aprev):
	if (cp.max(cp.abs(A-Aprev)) < params['meanfield_epsilon']):
		#print("early exit, diff=",cp.abs(A-Aprev))
		return True 
	else:
		return False

##############################################################


def init(params, Fstr=None):
	tstart = timer()
	if Fstr is not None:
		F, K, n, kmax, nodes = convert_from_F(params, Fstr)
	else:
		F, K, n, kmax, nodes = read_bnet(params) # dict of functions, num nodes, max-indegree, ordered node names
	# note that F is a pyeda function, whereas in RSC F is a string (for easier replacement)
	A = build_A0(params,n,nodes,F)  # estimated Average of each node, begins as 1/2 unless specified in params['init']
	TT = cp.array(list(itertools.product([0, 1], repeat=kmax)), dtype=cp.int8)[:, ::-1] # all possible input strings for a node
	# note TT is ordered sT earlier cols change more frequently
	M = build_map(F, kmax, nodes, n) # Map which of each nodes <= k inputs correspond to which other nodes (indices of A)
	Y = build_Y(TT, F, K, M, n, kmax, nodes) # which rows of TT are on for each node Y
	return tstart, F, K, n, kmax, A, TT, M, Y, nodes

def convert_from_F(params, Fstr):
	# instead of reading a bnet
	F, kmax = {}, 0
	for k in Fstr.keys():
		if '!' not in k: # skip complements
			F[k] = logic.str_to_fn(Fstr[k])
			count = len(F[k].support)
			kmax = max(kmax, count)
	K, n, nodes = build_nodes_n_degs(F, kmax)
	return F, K, n, kmax, nodes

def read_bnet(params):
	F, kmax = {}, 0
	with open(params['network_file'], 'r') as file:
		for line in file:
			line = line.strip() 
			if not line:
				continue 
			k, fn_str = line.split(',', 1)
			fn_str = fn_str.replace('\t','')
			assert(k not in F.keys() and '!' + k not in F.keys())
			F[k] = logic.str_to_fn(fn_str)
			count = len(F[k].support)
			kmax = max(kmax, count)
	K, n, nodes = build_nodes_n_degs(F, kmax)
	return F, K, n, kmax, nodes

def build_nodes_n_degs(F, kmax):
	n = len(F)
	nodes = list(F.keys())
	in_degs = cp.array([len(F[nodes[i]].support) for i in range(n)],dtype=int)
	K = cp.zeros((n,kmax),dtype=cp.int8)
	for i in range(n):
		K[i,:in_degs[i]]=1
	return K, n, nodes

def build_A0(params,n,nodes,F):
	A = cp.ones(n,dtype=float)/2
	for k in F.keys():
		if F[k].equivalent(1):
			A[nodes.index(k)]=1
		elif F[k].equivalent(0):
			A[nodes.index(k)]=0
	#for k in params['init']:
	#	A[nodes.index(k)] = params['init'][k]
	return A

def build_map(F, kmax, nodes, n):
	M = cp.zeros((n,kmax),dtype=util.get_uint_dtype(n))  
	for i in range(n):
		variables = sorted(v.name for v in frozenset(F[nodes[i]].support))
		# wow, this absurd way of getting the variable names is NEC to avoid some very bizarre behavior!
		for j in range(len(variables)):
			#M[i,-j-1] = nodes.index(variables[j])
			M[i,j] = nodes.index(variables[j])
	return M

def build_Y(TT, F, K, M, n, kmax, nodes):
	Y = cp.zeros((n,len(TT)),dtype=cp.int8)
	for i in range(n):
		fn =F[nodes[i]]
		num_vars = len(fn.support)
		vardict = {str(v): v for v in fn.support}
		if fn.is_one():
			Y[i,0] = 1 
		elif not fn.is_zero():
			for j in range(len(TT)):
				if j < int(2**num_vars): # note if j > 2**k_i, Y[i,j]=0
					assgnt = {vardict[nodes[int(M[i,k])]]:int(TT[j,k]) for k in range(num_vars)}
					Y[i,j] = int(fn.restrict(assgnt))
					#print('fn',fn,'assngt',assgnt,'val',Y[i,j])
	return Y

##############################################################

def finish(params, A, last_step):
	print('final avgs:\n',cp.round(A, 3))
	# do something final with time and last step

##############################################################	

if __name__ == "__main__":
	# for debugging
	run(PARAMS)