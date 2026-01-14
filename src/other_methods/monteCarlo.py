'''
From a random set of initial states, explicitly simulate for many steps until in attractor
Optionally parallized with the Cupy library on GPU.
'''

import itertools
import util
from . import simulate

CUPY, cp = util.import_cp_or_np(try_cupy=1)

def find_all_min(params, G):
	# uses brute force sim to find all transient control sets of size <= max_controllers
	# which drive target_node to desired sign (1 or 0)
	controllers = []
	target_node, sign = get_target(params)
	for i in range(params['max_monte_carlo_controllers']+1):
		add_controllers_of_set_size(params, G, controllers, target_node, i, sign)
		if len(controllers) > 0:
			break # only want min sized controllers 
	return controllers

def get_target(params):
	if '!' in params['target']:
		sign=0
		target_node = params['target'].replace('!','')
	else:
		sign=1
		target_node = params['target']
	return target_node, sign

def add_controllers_of_set_size(params, G, controllers, target_node, size, sign):
	# complexity is (n choose k)(2^k), where n is the number of nodes and k is size
	nodeNames = [node.name for node in G.nodes]
	for subset in itertools.combinations(nodeNames, size):
		for assignment in itertools.product([0, 1], repeat=size):
			controller_assignment = {subset[i]:assignment[i] for i in range(size)}
			controller_name = get_controller_name(controller_assignment)
			if not subset_already_saved(controllers, controller_name):
				when = 'init'
				params['init'], params['mutations'] = {},{}
				params[when] = controller_assignment
				G.prepare(params)
				avg, xf = simulate.measure(params, G)

				if cp.all(avg[:,G.nodeNums[target_node]] == sign):
					controllers += [controller_name]
				elif cp.all(avg[:,G.nodeNums[target_node]] == 1-sign) and controller_name == ['1']:
					controllers += [['0']] # control not possible if nominal average is opposite of target


def subset_already_saved(controllers, controller):
	controller_set = set(controller)
	return any(controller_set == set(c) for c in controllers)
	
def get_controller_name(controller_assignment):
	if len(controller_assignment)==0: #null controller:
		return ['1']
	controller_name = []
	for k in controller_assignment:
		if controller_assignment[k]==0:
			controller_name+=['!'+k]
		else:
			controller_name+=[k]
	return controller_name