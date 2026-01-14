'''
Implementation of Recursive Self-Composite (RSC) and its approximation to find terminal logic
'''

import activity, logic, util
import sys
from timeit import default_timer as timer


#################################################################	

def find_terminal_logic(params):
	tstart, F, fn, fn_prev = init(params)
	final_logic, max_recursion = core_loop(params, F, fn, fn_prev)
	finish(params, tstart, fn, final_logic)
	return final_logic, max_recursion

def core_loop(params, F, fn, fn_prev):
	for r in range(params['max_recursion']):
		fn, cont = step(params, F, fn, fn_prev, r)
		if not cont:
			break
	max_recursion = (r >= params['max_recursion']-1)
	if max_recursion:
		final_logic = [logic.to_pyEda([['0']])] # if have not converged by max recursion, terminal logic is unknown
		# TODO: log max recursion somewhere
	else:
		final_logic = build_final_logic(fn, fn_prev, params)
	return final_logic, max_recursion


################################################################################

def step(params, F, fn, fn_prev, r):
	update(params, r, fn, fn_prev)
	print("at step",r+1,'fn=',fn)
	fn = substitute(fn, F)
	fn = logic.reduce(params, fn)
	found = check_if_terminal(fn, fn_prev, params, r)
	return fn,not found


#################################################################

def init(params, F=None):
	tstart = timer()
	if F is None:
		F, reduced = load_F(params)
	fn, fn_prev = build_target_function(params, F)
	return tstart, F, fn, fn_prev

def load_F(params):
	F =  {}
	reduced=False
	with open(params['network_file'], 'r') as file:
		for line in file:
			line = line.strip() 
			if not line:
				continue 
			k, val = line.split(',', 1)
			val = val.replace('\t','')
			assert(k not in F.keys() and '!' + k not in F.keys())
			
			if params['approx'] and params['approx_orig_fns']:
				fn = logic.str_to_fn(val)
				if logic.fn_too_complex(params, fn, max_vars=params['max_initial_variables']):
					reduced = True
				fn = logic.reduce(params, fn, max_vars=params['max_initial_variables'])
				val = logic.fn_to_str(fn)
			update_F(params, F, k, val)

	apply_mutations(params, F)
	return F, reduced

def build_target_function(params, F):
	
	fn = logic.str_to_fn('(' + params['target'] + ')')
	for node in fn.support:
		assert(str(node) in F.keys()) # all nodes in output function must be in the network
	fn_prev = [None for _ in range(params['max_cycle_length'])]		

	if params['approx'] and params['approx_orig_fns']:
		fn = logic.reduce(params, fn)
		# TODO: warn if original target function is reduced!
	return fn, fn_prev

def update_F(params, F,  k, fn_str):
	F[k] = '('+fn_str+')'
	#compl_fn= logic.str_to_fn('!'+'(' + fn_str + ')')
	fn = logic.str_to_fn(fn_str)
	compl_fn = logic.not_to_dnf(params, fn)
	compl_fn_str = logic.fn_to_str(compl_fn)
	F['!' + k] = compl_fn_str
	fn_list = logic.str_to_F(fn_str)

def apply_mutations(params, F):
	if params['use_mutations']:
		for k in params['mutations']:
			assert(k in F.keys())
			F[k] = str(params['mutations'][k])
			F['!'+k] = str(1-int(params['mutations'][k]))

def update(params, r, fn, fn_prev):
	if params['verbose_rsc'] and r%params['print_freq']==0 and r>0:
		print("at step",r)
	fn_prev[1:] = fn_prev[:-1] # shift up by 1
	fn_prev[0] = fn

def substitute(fn, F):
	# assumes fn is in DNF
	fn = logic.fn_to_str(fn)
	s=''
	clauses = fn.split(' | ')
	if len(clauses) > 1:
		s += '('
	i=0
	for clause in clauses:
		if i!=0:
			s+=') | ('
		else:
			s+='('
		j=0
		eles = clause.split(' & ')
		for ele in eles:
			if j!=0:
				s+=' & '
			ele = ele.replace(' ','').replace('(','').replace(')','')
			s+='(' + F[ele] + ')' # replacement occurs here
			j+=1
		i+=1
	s+=')'
	if len(clauses) > 1:
		s += ')'
	fn = logic.str_to_fn(s)
	return fn

def check_if_terminal(fn, fn_prev, params, r):
	if fn.equivalent(fn_prev[0]) or fn.equivalent(1) or fn.equivalent(0):
		if params['verbose_rsc']:
			print("Found fixed-point terminal logic at step",r)
		return True
	for i in range(1,params['max_cycle_length']):
		if fn.equivalent(fn_prev[i]):		#fn==fn_prev[i] can miss if two functions are reduced differently
			if params['verbose_rsc']:
				print("Found cyclic terminal logic at step",r,"with cycle length",i+1)
			return True
	if r+1==params['max_recursion'] and params['verbose_rsc']:
		print("Reached max recursion without finding terminal logic")
	return False

def build_final_logic(fn, fn_prev, params):
	if fn.equivalent(fn_prev[0]) or fn.equivalent(1) or fn.equivalent(0):
		return [fn]
	for i in range(1,params['max_cycle_length']):
		if fn.equivalent(fn_prev[i]):		#fn==fn_prev[i] can miss if two functions are reduced differently
			final_logic = []
			for j in range(i+1):
				final_logic += [substitute_mutants(params,  fn_prev[j])]
			return final_logic # prev used: fn_prev[:i+1]
	return [logic.to_pyEda([['0']])] # hit max recursion without solution # should have found final logic

def substitute_mutants(params, fn):
	# uses pyeda.compose instead of explicitly use F[!x] since only constants
	if params['use_mutations']:
		keys = [k for k in fn.support if str(k) in params['mutations'].keys()]
		fn = fn.compose({k:params['mutations'][str(k)] for k in keys})
		fn = logic.reduce(params, fn)
	return fn

def finish(params, tstart, fn, final_logics):
	if params['verbose_rsc']:
		fn_str = logic.fn_to_str(fn)
		print("\nterminal logics =",final_logics)
		tend = timer()
		print("\nTime elapsed: ", round((tend-tstart)/3600,3),'hours')
	return final_logics

def main(param_file):
	params = util.load_yaml(param_file)
	terminal_logic = find_terminal_logic(params)[0]
	print("Terminal logic of",params['target'],"= ",terminal_logic)

#################################################################################

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python RSC.py PARAM_FILE")
	main(sys.argv[1])
