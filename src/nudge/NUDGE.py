'''
Finds controllers and optionally analyzes their robustness and mechanism, as specified in params.yaml
'''

from . import RSC, ensemble, analyze, util, colomoto_io
import sys

#################################################################################

def control(params, F=None):
	'''
	Returns (strategies, details):
		strategies: the best controllers as algorecell_types reprogramming strategies, using the original node names
		details: dict with all controllers, relative distances, minimal controllers, mechanism data and best controllers
	'''
	util.set_seed(params)

	tstart, F, fn, fn_prev = RSC.init(params, F=F)
	terminal_logic = ensemble.terminal_logic_predicate(params, F)
	details = analyze.robustness_and_mechanism(params, F, terminal_logic)
	strategies = colomoto_io.to_strategies(details['best controllers'], params.get('original_names'))
	return strategies, details

def main(param_file):

	params = util.load_yaml(param_file)
	strategies, details = control(params)
	if params['verbose_nudge']:
		util.pretty_print(details)

#################################################################################

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python -m nudge.NUDGE PARAM_FILE")
	main(sys.argv[1])
