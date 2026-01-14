'''
Builds ensemble of terminal logic and takes their conjunction to find controllers and increase confidence
'''

import logic, activity, RSC, util
import sys, os
CUPY, cp = util.import_cp_or_np(try_cupy=1)

def terminal_logic_predicate(params, F):
	if not params['approx']:
		f = build_terminal_logic_predicate(params, F)
	else:
		f = build_combined_ensemble_f(params, F)
	return f

def build_combined_ensemble_f(params, F):
	f=logic.str_to_fn('1')
	for r in range(params['ensemble_repeats']):
		f2 = build_terminal_logic_predicate(params, F)
		combo_fn = (f) & (f2)
		f = logic.reduce(params, combo_fn)
	return f

def build_terminal_logic_predicate(params, F):
	fn, fn_prev = RSC.build_target_function(params, F) # reset fn and fn_prev
	#fs, max_recursion = RSC.core_loop(params, F, fn, fn_prev)
	fs, max_recursion = RSC.find_terminal_logic(params)
	predicate = combine_oscils(params, fs)
	return predicate

def combine_oscils(params, fns):
	andfn = logic.str_to_fn('1')
	for i in range(len(fns)): # reduce one at a time to avoid computational explosion
		andfn = (andfn) & (fns[i])
		andfn = logic.reduce(params, andfn)
	return andfn