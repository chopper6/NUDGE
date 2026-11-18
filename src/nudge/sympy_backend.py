'''
SymPy backend for Boolean functions (the default backend)

Functions are kept in pure python as a set of DNF clauses, where each clause is a frozenset of literals (name, value).
This makes combining, restricting, sampling and comparing functions fast, and provides the subset of the PyEDA
expression interface used by NUDGE, so the rest of the package works with either backend.

SymPy is only called for exact two-level logic minimization (Quine-McCluskey via sympy.logic.SOPform), which
receives only the variables of the (already approximated) function. With approximation, logic.reduce() first discards
variables until at most params['max_variables'] (or 'max_init_variables') remain, and minimize_dnf() refuses anything larger.
'''

import itertools

EXACT_ENUMERATION_LIMIT = 16 # beyond this many variables, minterm enumeration is infeasible and only absorption is applied
CACHE_SIZE = 100000 # minimized functions are cached, since RSC repeatedly reduces the same functions
_warned = [False]
_cache = {}


class Var(str):
	'''A variable name, with the .name attribute that PyEDA variables also have.'''
	__slots__ = ()

	@property
	def name(self):
		return str(self)


TRUE = frozenset([frozenset()])
FALSE = frozenset()


################################################################################
# clause set operations

def _absorb(clauses):
	kept = []
	for c in sorted(set(clauses), key=len):
		if not any(k <= c for k in kept):
			kept.append(c)
	return frozenset(kept)


def _consistent(c):
	return len({n for n, _ in c}) == len(c)


def _or(a, b):
	return _absorb(a | b)


def _and(a, b):
	return _absorb(c1 | c2 for c1 in a for c2 in b if _consistent(c1 | c2))


def _not(a):
	result = TRUE
	for c in a:
		result = _and(result, frozenset(frozenset([(n, 1 - v)]) for n, v in c))
	return result


def _restrict(a, mapping):
	out = []
	for c in a:
		keep, falsified = [], False
		for n, v in c:
			if n in mapping:
				if mapping[n] != v:
					falsified = True
					break
			else:
				keep.append((n, v))
		if not falsified:
			out.append(frozenset(keep))
	return _absorb(out)


def _tautology(a):
	# Shannon expansion on the most frequent variable
	if frozenset() in a:
		return True
	if not a:
		return False
	counts = {}
	for c in a:
		for n, _ in c:
			counts[n] = counts.get(n, 0) + 1
	v = max(counts, key=counts.get)
	return _tautology(_restrict(a, {v: 0})) and _tautology(_restrict(a, {v: 1}))


def _implies(a, b):
	return all(_tautology(_restrict(b, dict(c))) for c in a)


################################################################################

class DNF:
	__slots__ = ('c',)

	def __init__(self, clauses):
		self.c = clauses

	def __and__(self, other): return DNF(_and(self.c, _coerce(other)))
	def __rand__(self, other): return DNF(_and(_coerce(other), self.c))
	def __or__(self, other): return DNF(_or(self.c, _coerce(other)))
	def __ror__(self, other): return DNF(_or(_coerce(other), self.c))
	def __invert__(self): return DNF(_not(self.c))

	@property
	def support(self):
		return frozenset(Var(n) for c in self.c for n, _ in c)

	def simplify(self): return self
	def to_dnf(self): return self
	def is_dnf(self): return True
	def is_one(self): return frozenset() in self.c
	def is_zero(self): return not self.c

	def equivalent(self, other):
		if other is None:
			return False
		o = _coerce(other)
		if self.c == o:
			return True
		return _implies(self.c, o) and _implies(o, self.c)

	def compose(self, mapping):
		return DNF(_restrict(self.c, {str(k): int(v) for k, v in mapping.items()}))

	def restrict(self, mapping):
		# fast path for a full assignment (as in activity sampling): evaluate the clauses directly
		try:
			value = any(all(mapping[n] == v for n, v in c) for c in self.c)
		except KeyError:
			return self.compose(mapping)
		return _ONE if value else _ZERO

	def __int__(self):
		if self.is_one():
			return 1
		if self.is_zero():
			return 0
		raise ValueError('function is not constant: ' + str(self))

	def __str__(self):
		# same format as PyEDA, e.g. Or(And(A, ~B), C)
		if self.is_one():
			return '1'
		if self.is_zero():
			return '0'
		clauses = []
		for c in self.c:
			lits = [s for _, s in sorted((n, ('' if v else '~') + n) for n, v in c)]
			clauses.append(lits[0] if len(lits) == 1 else 'And(' + ', '.join(lits) + ')')
		clauses.sort()
		return clauses[0] if len(clauses) == 1 else 'Or(' + ', '.join(clauses) + ')'

	def __repr__(self):
		return str(self)


_ONE, _ZERO = DNF(TRUE), DNF(FALSE)


def _coerce(x):
	if isinstance(x, DNF):
		return x.c
	if isinstance(x, (bool, int)) or type(x).__name__ in ('bool_', 'int64', 'int32'):
		return TRUE if int(x) else FALSE
	raise TypeError('cannot convert ' + repr(x) + ' to a Boolean function')


################################################################################

def _tokens(s):
	toks, i = [], 0
	while i < len(s):
		ch = s[i]
		if ch.isspace():
			i += 1
		elif ch.isalnum() or ch == '_':
			j = i
			while j < len(s) and (s[j].isalnum() or s[j] in '_.'):
				j += 1
			toks.append(s[i:j])
			i = j
		else:
			toks.append(ch)
			i += 1
	return toks


def expr(s):
	'''Parses a string such as "(A & !B) | C" into DNF (also accepts ~ for NOT, and the constants 0 and 1).'''
	if not isinstance(s, str):
		return DNF(_coerce(s))
	toks = _tokens(s)
	pos = [0]

	def peek():
		return toks[pos[0]] if pos[0] < len(toks) else None

	def take():
		pos[0] += 1
		return toks[pos[0] - 1]

	def parse_or():
		a = parse_and()
		while peek() == '|':
			take()
			a = _or(a, parse_and())
		return a

	def parse_and():
		a = parse_unary()
		while peek() == '&':
			take()
			a = _and(a, parse_unary())
		return a

	def parse_unary():
		tok = take()
		if tok in ('~', '!'):
			return _not(parse_unary())
		if tok == '(':
			a = parse_or()
			assert take() == ')', 'unbalanced parentheses in ' + s
			return a
		if tok in ('0', '1'):
			return TRUE if tok == '1' else FALSE
		assert tok[0].isalpha() or tok[0] == '_', 'unexpected token ' + tok + ' in ' + s
		return frozenset([frozenset([(tok, 1)])])

	a = parse_or()
	assert pos[0] == len(toks), 'could not parse ' + s
	return DNF(a)


def implies(a, b):
	'''Whether function a logically implies function b.'''
	return _implies(_coerce(a), _coerce(b))


def minimize_dnf(fn, max_vars=None):
	'''Exact minimal DNF of fn, computed by sympy. If max_vars is given, fn must not have more variables than that.'''
	if fn.is_one() or fn.is_zero() or len(fn.c) == 1:
		return fn
	names = sorted({n for c in fn.c for n, _ in c})
	if max_vars is not None and len(names) > max_vars:
		raise ValueError('function with ' + str(len(names)) + ' variables passed to sympy, but the limit is ' + str(max_vars))
	if len(names) > EXACT_ENUMERATION_LIMIT:
		if not _warned[0]:
			print('Warning: function with', len(names), 'variables is too large for exact minimization with sympy, only absorption is applied. Consider approx: true or logic_backend: "pyeda".')
			_warned[0] = True
		return fn
	if fn.c in _cache:
		return DNF(_cache[fn.c])
	result = _minimize(fn, names)
	if len(_cache) >= CACHE_SIZE:
		_cache.clear()
	_cache[fn.c] = result
	return DNF(result)


def _minimize(fn, names):
	idx = {n: i for i, n in enumerate(names)}
	minterms = set()
	for c in fn.c:
		fixed = {idx[n]: v for n, v in c}
		free = [i for i in range(len(names)) if i not in fixed]
		for bits in itertools.product((0, 1), repeat=len(free)):
			m = [0] * len(names)
			for i, v in fixed.items():
				m[i] = v
			for i, v in zip(free, bits):
				m[i] = v
			minterms.add(tuple(m))
	if len(minterms) == 2 ** len(names):
		return TRUE

	import sympy
	from sympy.logic import SOPform
	symbols = [sympy.Symbol(n) for n in names]
	result = SOPform(symbols, [list(m) for m in sorted(minterms)])
	return _from_sympy(result)


def _from_sympy(e):
	from sympy.logic.boolalg import And, Or, Not, BooleanTrue, BooleanFalse
	if isinstance(e, BooleanTrue):
		return TRUE
	if isinstance(e, BooleanFalse):
		return FALSE
	clauses = e.args if isinstance(e, Or) else (e,)
	out = []
	for c in clauses:
		lits = c.args if isinstance(c, And) else (c,)
		out.append(frozenset((l.args[0].name, 0) if isinstance(l, Not) else (l.name, 1) for l in lits))
	return _absorb(out)
