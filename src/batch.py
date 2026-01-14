'''
Compares NUDGE to other methods on a batch of networks
'''

import NUDGE, RSC, logic, util
from other_methods import meanField, ldoi, monteCarlo, simulate, net
import random, sys, os, csv
from timeit import default_timer as timer
CUPY, cp = util.import_cp_or_np(try_cupy=1)

def main(param_file, run_title):
	params = util.load_yaml(param_file)
	params, tstart, data, files, targets = init(params, run_title)
	i=0
	for file in files:
		if os.path.basename(file) in targets.keys():
			i+=1
			print("\n#" + str(i) + ": starting network file",file)
			params, G, F, reduced = init_one(params, file)
			#if not os.path.basename(file) in targets.keys():
			#	target = [random.choice(G.nodeNames)]
			#	print("\trandom target =",target)
			if not reduced or not params['skip_reduced_networks']:
				target = targets[os.path.basename(file)]
				print("\ttargets from csv =",target)
				eval_one(params, G, F, data, target)
			else:
				print("\tskipping due to large in-degree")
	finish(params, data, run_title, tstart, targets)

#################################################################################

def init(params, run_title):
	tstart = timer()
	data = {}
	if params['measure_robustness']:
		params['methods'] = ['NUDGE','NUDGE_ROBUST']
		data['NUDGE Errors'], data['NUDGE_ROBUST Errors'] = [], []
		data['robustness'] = {} 
	else:
		params['methods'] = []
		for k in ['NUDGE','LDOI','MEAN-FIELD','MONTE_CARLO']:
			if params['include_'+k]:
				params['methods'] += [k]
				data[k + ' Errors'] = []
	init_csvs(params, run_title)
	files = load_network_folder(params)
	targets = load_network_targets(params['targets_csv'])

	return params, tstart, data, files, targets

def load_network_folder(params):
	network_files, mutations = [], {}
	for filename in os.listdir(params['network_dir']):
		file = os.path.join(params['network_dir'], filename)
		if (os.path.isfile(file)) and ((not params['cutoff']) or (len(network_files)<params['cutoff'])):
			network_files += [file]
	return network_files

def load_network_targets(csv_file):
	result = {}
	with open(csv_file, newline='') as f:
		reader = csv.DictReader(f)
		for row in reader:
			net = row['Network']
			tar = row['Target']
			if net not in result:
				result[net] = []
			result[net].append(tar)
	return result

def init_one(params, file):
	params['network_file'] = file
	F, reduced = RSC.load_F(params) 
	assign_IO_from_F(params, F)

	sim_params = simulate.get_sim_params(params)
	params = {**sim_params, **params}
	params['features'] = [{'id':'avg', 'active' : 1, 'print' : True, 'when' : 'equilibrium', 'attractor' : True, 'update_freq' : 1}]
	G = net.Net(params) 
	G.prepare(params)
	return params, G, F, reduced

#################################################################################

def assign_IO_from_F(params, F):
	assign_inputs_from_F(params, F)
	assign_outputs_from_F(params, F)

def assign_inputs_from_F(params, F):
	params['inputs'] = []
	for k in F.keys():
		if '!' not in k:
			if F[k] in [k, '0','1','(' + k + ')']:
				params['inputs'] += [k]

def assign_outputs_from_F(params, F):
	params['outputs'] = []
	found = {}
	for k in F.keys():
		if '!' not in k:
			found[k] = False
	for k in F.keys():
		if '!' not in k:
			fn = logic.str_to_fn(F[k])
			for ele in fn.support:
				found[str(ele)]=True 
	for k in F.keys():
		if '!' not in k:
			if not found[k]:
				params['outputs'] += [k]

#################################################################################

def eval_one(params, G, F, data, targets):
	for target in targets:
		params['target'] = target

		for k in params['methods']:
			controllers, skip, time = find_controllers(params, k, G, F,data)
			errors = avg_control_error(params, controllers, G, skip, k)
			write_to_csv_and_data(params,data,controllers,k,errors, time)

				
		params['init'], params['mutations'] = {},{} # wipe clean

def find_controllers(params, name, G, F, data):
	name = name.upper()
	tstart = timer()
	if name == 'NUDGE':
		params['find_mechanism'] = False
		result = NUDGE.control(params, F=F)
		controllers = result['best controllers']
		skip = (controllers[0] == ['0'])
		#has_no_inputs = any(all(inp not in ctrl for inp in params['inputs'])for ctrl in controllers)
	elif name == 'LDOI':
		controllers = ldoi.find_controllers(params, F)
		skip = (len(controllers) == 0)
	elif name == 'MEAN-FIELD':
		controllers =  meanField.find_controllers(params, F)
		skip = (len(controllers) == 0)
	elif name == 'MONTE_CARLO':
		controllers = monteCarlo.find_all_min(params, G)
		skip = (len(controllers) == 0)
	else:
		print("Unrecognized name:",name,'\n')
		assert(0) # unrecognized method name
	tend = timer()
	time = round((tend-tstart)/60,5)
	return controllers, skip, time


def avg_control_error(params, controllers, G, skip, name):
	if not skip:
		errors = []
		num_checked = 0
		#num_samples = min(len(controllers),params['error_msmt_samples'])
		for r in range(len(controllers)):
			cnt = controllers[r]
			one, zero = edge_cases(cnt)
			if not zero:
				num_checked += 1
				errors += [measure_error(params, G, cnt, one, name)]
		if num_checked == 0:
			return 'NA'
		else:
			return errors
	else:
		return 'NA'

def edge_cases(controller):
	if (controller == {}) or (controller == None) or (len(controller)==0):
		return True, True
	elif controller == ['1']:
		return True, False 
	elif controller == ['0']:
		return False, True 
	else:
		return False, False

def measure_error(params, G, cnt, one_cnt, name):
	if '!' in params['target']:
		sign = 0
		output = params['target'].replace('!','')
	else:
		sign = 1
		output = params['target']
	prep_controller(params, G, cnt, one_cnt)
	avg = sim_with_control(params, G)
	bias = cp.mean(avg[:,G.nodeNums[output]])
	error = bias_to_error(bias, sign)
	if CUPY:
		error = error.item()
	return error

def prep_controller(params, G, controller, one_cnt):
	params['init'], params['mutations'] = {},{}
	cnt_dict = convert_cnt_to_dict(controller)
	#print("\t cnt",controller,'dict',cnt_dict,'\n\n')
	if not one_cnt:
		when = 'init'
		params[when] = cnt_dict
	G.prepare(params)

def sim_with_control(params, G):
	avg = simulate.measure(params, G)[0]
	return avg

def bias_to_error(bias, sign):
	if CUPY:
		bias = bias.get()
	error = abs(float(sign) - bias)
	return error

def convert_cnt_to_dict(controller):
	#if controller in [['0'],['1']]:
	#	return {} # null control
	d = {}
	for c in controller:
		if '!' in c:
			val =0
			c=c.replace('!','')
		else:
			val=1 
		d[c]=val 
	return d

def init_csvs(params, run_title):
	for name in params['methods']:
		params['csv_file_'+name] = params['output_folder'] + '/' + run_title + '_' + name + '_RESULTS.csv'
		headers = ['Network','Target','Controllers','Errors','Minutes']
		
		if (os.path.exists(params['csv_file_'+name])) and (run_title.lower() != 'debug'):
			sys.exit("CSV files already exist, change run name.\n")
		util.init_csv(params['csv_file_'+name], headers)

	if params['measure_robustness']:
		robutness_headers = ['Network','Target','All Controllers','Minimal Controllers','Relative Heights','Robust Controllers','Robustness Matters','Robustness Reliable']
		params['csv_file_robustness'] = params['output_folder'] + '/' + run_title + '_' + 'robustness.csv'
		util.init_csv(params['csv_file_robustness'], robutness_headers)


def write_to_csv_and_data(params, data, controllers, name, errors, time):
	name = name.upper()
	row = [params['network_file'],params['target'], controllers, errors, time]
	util.append_csv(params['csv_file_'+name], row)
	data[name + ' Errors'] += [errors]


#################################################################################

def finish(params, data, run_title, tstart, targets):
	tend = timer()
	print("\nTime elapsed: ", round((tend-tstart)/3600,3),'hours')

	d= {'data':data,'params':params,'run_title':run_title,'targets':targets}
	util.pickle_it(d, params['output_folder'] + run_title, use_timestamp=True)


#################################################################################

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Usage: python batch.py PARAM_FILE RUN_TITLE")
	main(sys.argv[1], sys.argv[2])