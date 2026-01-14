'''
Analyzes controllers in terms of robustness and mechanism
'''
import logic, ensemble, RSC, robustness, mechanism, util
import sys, itertools
import numpy as np

def robustness_and_mechanism(params, F, terminal_logic):
	controllers = logic.from_pyEda(terminal_logic)
	rob, mech = (params['measure_robustness'] and (controllers[0] not in [['1'],['0']])), (params['find_mechanism'] and (controllers[0] not in [['1'],['0']]))

	if rob or mech:
		drives, self_driven, drives_self_driven, cnt_fns = calc_driven_and_self_driven(params, F, controllers)

	if rob:
		min_controllers, min_robust_controllers, relative_heights = robustness.pick_min_robust_controllers(params, controllers, cnt_fns, drives_self_driven)
		best_controllers = min_robust_controllers
	else:
		min_controllers, min_robust_controllers, relative_heights, best_controllers = organize_controllers_without_robustness(params, controllers)
	
	if mech:
		mechanism_data = {}
		for controller in best_controllers:
			cnt_name = logic.F_to_str([controller]).replace('(','').replace(')','')
			mechanism_data[cnt_name] = mechanism.find_mechanism(params, F, controllers, controller, cnt_fns, drives_self_driven, self_driven)
	else:
		mechanism_data = None

	return {
		'all controllers':controllers,
		'relative heights':relative_heights,
		'min controllers': min_controllers,
		'min robust controllers': min_robust_controllers,
		'mechanism data': mechanism_data,
		'best controllers': best_controllers
	}

########################################################################################		

def calc_driven_and_self_driven(params, F, controllers):
	# note that this takes all controllers, not just minimal ones
	assert(controllers[0] not in [['0'],['1']]) # no relative height if is

	cnt_fns = generate_cnt_fns(controllers, params['analysis_order']) # combinations of controllers if order > 1, otherwise same as controllers (in function form)
	drives, self_driven, drives_self_driven = calc_driven(params, F, cnt_fns)
	return drives, self_driven, drives_self_driven, cnt_fns

def calc_driven(params, F, cnt_fns):
	og_target = params['target']
	k=len(cnt_fns)
	D = np.zeros((k,k), dtype=bool) # D[i,j]=1 --> controller i drivers controller j
	for i in range(k):
		params['target'] = logic.fn_to_str(cnt_fns[i]) 
		terminal_fn = ensemble.terminal_logic_predicate(params, F)
		if not (terminal_fn.is_zero() or terminal_fn.is_one()):
			for j in range(k):
				D[j,i] = logic.implies(params, cnt_fns[j], terminal_fn)
	self_driven = np.diag(D) # whether each controller is self-driven
	D_self_driven = D & self_driven[None] # drives something self-driven. zero all columns of D where self_driven==0, only care if drive a self-driven node
	params['target'] = og_target
	#check_for_oscil_motif(params, self_driven, cnt_fns)
	return D, self_driven, D_self_driven

def check_for_oscil_motif(params, self_driven, cnt_fns):
	for i in range(len(cnt_fns)):
		fn_i = logic.from_pyEda(cnt_fns[i])
		if self_driven[i] and len(fn_i) > 1:
			no_SD_subspace = True
			for j in range(len(cnt_fns)):
				if i!=j and self_driven[j]: 
					if logic.implies(params, self_driven[j], self_driven[i]):
						no_SD_subspace = False
						break
			if no_SD_subspace:
				print("\n\tFound oscilating motif:",cnt_fns[i],'\n')
				return True 
	return False

def generate_cnt_fns(controllers, order):
	if order is None or order < 1:
		return []

	max_combination_size = min(order, len(controllers))
	cnt_fns = []
	for combination_size in range(1, max_combination_size + 1):
		for index_tuple in itertools.combinations(range(len(controllers)), combination_size):
			selected_controllers = [controllers[i] for i in index_tuple]
			cnt_fns.append(logic.to_pyEda(selected_controllers))

	return cnt_fns

def organize_controllers_without_robustness(params, controllers):
	min_lng = min(len(lst) for lst in controllers)
	min_controllers =  [lst for lst in controllers if len(lst) == min_lng]
	min_robust_controllers, relative_heights= None, None
	best_controllers = min_controllers
	return min_controllers, min_robust_controllers, relative_heights, best_controllers

def main(param_file):

	params = util.load_yaml(param_file)
	if not (params['measure_robustness'] | params['find_mechanism']):
		sys.exit("Analyze either measures robustness or finds a mechanism, check parameters 'measure_robustness' and 'find_mechanism'.")
	tstart, F, fn, fn_prev = RSC.init(params)
	terminal_logic = ensemble.terminal_logic_predicate(params, F)
	controllers = logic.from_pyEda(terminal_logic)
	controller = logic.str_to_F(params['controller'])[0] 

	if controllers[0] in [['0'],['1']]:
		sys.exit("Terminal logic predicate is a constant, possibly due to approximation error.")
	if controller not in controllers:
		sys.exit("Controller "+ str(controller) + " was not found in the terminal logic conjunction " + str(terminal_logic))

	controller_index = [controllers.index(controller)]
	drives, self_driven, drives_self_driven, cnt_fns = calc_driven_and_self_driven(params, F, controllers)
	
	results = {}
	if params['measure_robustness']:
		print("Measuring robustness...")
		results['relative height'] = robustness.calc_relative_heights(params, controller_index, cnt_fns, drives_self_driven)[0]
	
	if params['find_mechanism']:
		print("Finding mechanism...")
		cnt_name = params['controller']
		results['mechanism'] = mechanism.find_mechanism(params, F, controllers, controller, cnt_fns, drives_self_driven, self_driven, plot_it=True)
	
	util.pretty_print(results)

#################################################################################

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python analyze.py PARAM_FILE")
	main(sys.argv[1])