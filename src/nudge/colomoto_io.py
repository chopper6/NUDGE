'''
Bridge to CoLoMoTo: networks are read with colomoto.minibn and controllers are returned as algorecell_types strategies
'''

from colomoto import minibn
import algorecell_types as art


def read_bnet(path):
	bn = minibn.BooleanNetwork.load(path)
	order = list(bn)
	renamed = bn.sanitize_names()
	for name in [n for n in bn if n[0].isdigit()]:
		bn.rename(name, 'n' + name)
		original = next((k for k, v in renamed.items() if v == name), name)
		renamed[original] = 'n' + name
	rules = {}
	for k in order:
		f = bn[renamed.get(k, k)]
		rules[renamed.get(k, k)] = str(int(f is bn.ba.TRUE)) if f in (bn.ba.TRUE, bn.ba.FALSE) else str(f)
	return rules, {v: k for k, v in renamed.items()}

def to_strategies(controllers, original_names=None):
	names = original_names or {}
	strategies = art.ReprogrammingStrategies()
	for cnt in controllers:
		if cnt == ['0']:
			continue
		state = {} if cnt == ['1'] else {names.get(lit.lstrip('!'), lit.lstrip('!')): int(not lit.startswith('!')) for lit in cnt}
		strategies.add(art.FromAny(art.InstantaneousPerturbation(state)))
	return strategies
