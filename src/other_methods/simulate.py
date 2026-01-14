'''
Simulate dynamics of networks into steady states
and calculates averages of each node in those steady states

'''
import util
CUPY, cp = util.import_cp_or_np(try_cupy=1)

################################################################################################

def measure(params, G, x0=None):
	# Runs transient (no avg) then equilibrium (collects avg); returns (avg, x_final)
	if x0 is None:
		x0 = get_init_sample(params, G)
	x_mid, _ = run_lap(params, x0, G, collect_avg=False)
	x_fin, avg = run_lap(params, x_mid, G, collect_avg=True)
	return avg, x_fin


################################################################################################


def get_sim_params(params):
	return {
		"num_samples": params['monte_carlo_samples'],
		"time_steps": params['monte_carlo_time_steps'],
		"update_rule": "sync",
		"savefig": 0,
		"outputs": [],
		"inputs": [],
		"init": {},
		"mutations": {},
		"clause_bin_size": 8,
		"mpl_backend": False,
		'ignore_input_sampling':True,
	}



def get_init_sample(params, G):
	p = 0.5
	x0 = cp.random.choice(a=[0,1], size=(params['num_samples'], G.n), p=[p,1-p]).astype(bool, copy=False)
	if 'inputs' in params and params.get('input_state_indices'):
		assert(0) # jp unused and if so, del
		input_idx = G.input_indices()
		for i, s in enumerate(G.get_input_sets()):
			a, b = params['input_state_indices'][i]
			x0[a:b, input_idx] = cp.array(s)
	return apply_setting_to_x0(params, G, x0)

def apply_setting_to_x0(params, G, x0):
	for k, v in params.get('mutations', {}).items():
		x0[:, G.nodeNums[k]] = v
	for k, v in params.get('init', {}).items():
		x0[:, G.nodeNums[k]] = v
	return x0

def run_lap(params, x0, G, collect_avg=False):
	node_dtype = util.get_node_dtype(params)
	x = cp.array(x0, dtype=node_dtype).copy()
	avg = cp.zeros_like(x, dtype=cp.float64) if collect_avg else None
	for i in range(params['time_steps']):
		xn = step_core(x, G, params)
		x, _ = apply_update_rule(params, G, x, xn)
		if collect_avg:
			avg += x 
	if collect_avg:
		avg /= float(params['time_steps'])
	return x, avg

def step_core(x, G, params):
	X = cp.concatenate((x, cp.logical_not(x[:, :G.n])), axis=1)
	clauses = cp.all(X[:, G.Fmapd['nodes_to_clauses']], axis=2)
	partials = cp.any(clauses[:, G.Fmapd['clauses_to_threads']], axis=3)
	x_next = cp.sum(cp.matmul(cp.swapaxes(partials, 0, 1), G.Fmapd['threads_to_nodes']), axis=0)
	return x_next.astype(util.get_node_dtype(params))

def apply_update_rule(params, G, x, x_next):
	ur = params['update_rule']
	if ur == 'sync':
		return x_next, cp.ones(x.shape, dtype=bool)
	if ur == 'psync':
		mask = (cp.random.rand(params['num_samples'], G.n) > 0.5)
		return (mask & x_next) | (~mask & x), mask
	if ur == 'async':
		mask = cp.zeros((params['num_samples'], G.n), dtype=bool)
		mask[cp.arange(params['num_samples']), cp.random.randint(0, G.n, size=params['num_samples'])] = True
		return (mask & x_next) | (~mask & x), mask
	raise SystemExit("ERROR: 'update_rule' not recognized")

#################################################################################

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python simulate.py PARAM_FILE")
	params = util.load_yaml(param_file)
	G = net.Net(params) 
	avg, x_fin = measure(params, G)
	print("Nodes=",G.nodeNames,'\n\nAvg=',avg,'\n')