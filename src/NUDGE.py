'''
Finds controllers and optionally analyzes their robustness and mechanism, as specified in params.yaml
'''

import RSC, ensemble, analyze, util
import sys

#################################################################################

def control(params, F=None):

	tstart, F, fn, fn_prev = RSC.init(params, F=F)
	terminal_logic = ensemble.terminal_logic_predicate(params, F)
	result = analyze.robustness_and_mechanism(params, F, terminal_logic)
	return result

def main(param_file):

	params = util.load_yaml(param_file)
	results = control(params)
	if params['verbose_nudge']:
		util.pretty_print(results)

#################################################################################

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python POKE.py PARAM_FILE")
	main(sys.argv[1])
