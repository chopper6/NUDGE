'''
Measures the robustness of controllers
Defined by the number of perturbations needed to escape the desired phenotype (relative distance)
'''

from . import logic
import numpy as np
import itertools, sys


def calc_relative_distances(params, indices, cnt_fns, drives_self_reinforced):
	relative_distance = np.zeros(len(indices),dtype=int) 

	j=0
	for i in indices: # only the specified controllers
		set_of_controllers = convert_to_sets(drives_self_reinforced[i], cnt_fns)
		relative_distance[j] = min_hitting_set_size(set_of_controllers)
		j+=1

	return relative_distance

def pick_min_robust_controllers(params, controllers, cnt_fns, drives_self_reinforced):
	if controllers[0] in [['0'],['1']]:
		return controllers, controllers, 'NA'
	min_lng = min(len(lst) for lst in controllers)
	k = len(controllers)
	min_controllers = [lst for lst in controllers if len(lst) == min_lng]
	min_indices = [i for i, lst in enumerate(controllers) if len(lst) == min_lng]

	# find max relative distance among min_controllers
	relative_distances = calc_relative_distances(params, min_indices, cnt_fns, drives_self_reinforced)
	max_distance = -2 # of a min sized controller
	min_robust_controllers = []
	for i in range(len(min_indices)):
		if relative_distances[i] > max_distance:
			min_robust_controllers = [min_controllers[i]]
			max_distance = relative_distances[i]
		elif relative_distances[i] == max_distance:
			min_robust_controllers += [min_controllers[i]]

	return min_controllers, min_robust_controllers, [int(d) for d in relative_distances]

##########################################################################################

def convert_to_sets(mask, cnt_fns):
	set_of_sets = []
	for i in range(len(mask)):
		if mask[i]:
			cnts = logic.from_pyEda(cnt_fns[i])
			cnts_1D = list(set(x for sub in cnts for x in sub)) # [['A'],['B']] --> ['A','B'] 
			set_of_sets += [cnts_1D]
			#set_of_sets += [controllers[i]] # only first order
	return set_of_sets

def min_hitting_set_size(set_of_sets):
	family = [set(s) for s in set_of_sets]
	if not family:
		return 0
	universe = sorted(set().union(*family))
	if not universe:
		return 0

	for k in range(len(universe) + 1):
		for chosen in itertools.combinations(universe, k):
			chosen_set = set(chosen)
			if all(chosen_set & s for s in family):
				return k
