'''
Logic reduction, using either SymPy (default, exact minimization) or PyEDA (optional, espresso heuristic)

3 representations of a function:
F: [['A','B'],['C']]
str: '(A & B) | C'
function object (a PyEDA expression, or a SymPy expression wrapped to provide the same interface)

The backend is chosen by params['logic_backend'] ('sympy' or 'pyeda'), see set_backend().
'''
import numpy as np
from . import activity

BACKENDS = ('sympy', 'pyeda')
_backend = None
_eda = None # pyeda.inter, only imported if the pyeda backend is selected

def set_backend(name='sympy'):
	global _backend, _eda
	name = (name or 'sympy').lower()
	if name not in BACKENDS:
		raise ValueError("logic_backend must be one of " + str(BACKENDS) + ", not '" + str(name) + "'")
	if name == 'pyeda' and _eda is None:
		try:
			from pyeda import inter as eda
		except ImportError as err:
			raise ImportError("logic_backend 'pyeda' requires PyEDA (pip install pyeda), which only supports python 3.9 or earlier. Use logic_backend 'sympy' instead.") from err
		_eda = eda
	_backend = name

def backend_from_params(params):
	set_backend(params.get('logic_backend', 'sympy'))

def get_backend():
	if _backend is None:
		set_backend('sympy')
	return _backend

def _expr(s):
	if get_backend() == 'pyeda':
		return _eda.expr(s)
	from . import sympy_backend
	return sympy_backend.expr(s)

################################################################################

def reduce(params, fn, max_vars = None):
	fn = fn.simplify()
	if params['approx']:
		if max_vars is None:
			max_vars = params['max_variables']
		while fn_too_complex(params, fn, max_vars):
			fn = activity.discard_one_variable(params, fn)
	else:
		max_vars = None # exact: no limit on the number of variables
	fn=exactly_reduce(fn, max_vars)
	return fn

def exactly_reduce(fn, max_vars=None):
	fn = fn.simplify() # this can massively help reduction time, maybe due to to_dnf()
	if get_backend() == 'sympy':
		from . import sympy_backend
		return sympy_backend.minimize_dnf(fn, max_vars) # exact minimization by sympy, raises an error if fn has more than max_vars variables
	fn = fn.to_dnf() # this is naively implemented and may cause memory errors...
	if fn.equivalent(1):
		return _expr(1)
	elif fn.equivalent(0):
		return _expr(0)
	else:
		fn, = _eda.espresso_exprs(fn)
		return fn

def fn_too_complex(params, fn, max_vars):
	if get_backend() == 'sympy':
		if fn.is_one() or fn.is_zero(): # constants are already reduced to a single empty clause or no clauses
			return False
	elif (fn.equivalent(1) or fn.equivalent(0)):
		return False # ofc not too big
	varbs = list(fn.support)
	if len(varbs) > max_vars:
		return True 
	else:
		return False

################################################################################
	
def implies(params, a, b):
	# checks if a => b
	if isinstance(b, np.bool_):
		return b
	if params['approx'] and not isinstance(a, np.bool_):
		a=reduce(params, a)
		b=reduce(params, b)
	if get_backend() == 'sympy':
		from . import sympy_backend
		return sympy_backend.implies(a, b) # tautology check by Shannon expansion, avoids minimizing the tautology
	tautology = not_to_dnf(params, a) | b
	return reduce(params, tautology).is_one()

def fn_to_str(fn):
	# function object to string
	F = from_pyEda(fn)
	fn_str = F_to_str(F)
	return fn_str

def str_to_fn(fn_str):
	fn = _expr(fn_str.replace('!','~'))
	return fn

def not_to_dnf(params, fn, cap=64):
	# takes function object in dnf form
	# and computes its complement dnf
	F = from_pyEda(fn)
	if F == [['0']]:
		return to_pyEda([['1']])
	elif F == [['1']]:
		return to_pyEda([['0']])
	Fflipped = [[lit[1:] if lit.startswith('!') else '!' + lit for lit in clause] for clause in F] # '!A'-> 'A', 'A'-> '!A' 
	Fcompl = [[Fflipped[0][i]] for i in range(len(Fflipped[0]))]
	for i in range(1, len(Fflipped)):
		Fcompl_next = []
		for j in range(len(Fcompl)):
			for k in range(len(Fflipped[i])):
				clause = Fcompl[j] + [Fflipped[i][k]]
				Fcompl_next += [clause]
		if (len(Fcompl_next) >= cap) or (i+1==len(Fflipped)):
			fn = to_pyEda(Fcompl_next)
			fn = reduce(params, fn)
			Fcompl_next = from_pyEda(fn)
		Fcompl = Fcompl_next

	fn_compl = to_pyEda(Fcompl)
	return fn_compl


################################################################################


def F_to_str(fn):
	s = ''
	i=0
	if (len(fn) > 1): 
		s += '('
	for clause in fn:
		j=0
		if i!=0:
			s+=') | ('
		else:
			s+= '('
		for ele in clause:
			if j!=0:
				s+=' & '
			s+=ele 
			j+=1
		i+=1
	s+= ')'
	if len(fn) > 1:  
		s+=')'
	return s 

def str_to_F(s):
	# assumes dnf
	fn = []
	clauses = s.split(' | ') 
	for clause in clauses:
		fn_clause = []
		eles = clause.split(' & ')
		for ele in eles:
			ele = ele.replace(')','').replace('(','').strip()
			fn_clause += [ele]
		fn += [sorted(fn_clause)]
	fn = sorted(fn)
	return fn 

def from_pyEda(fn):
	newF = []
	fn = str(fn).replace('Or(','')
	assert('Or(' not in fn) #since this is DNF, should only be one Or which is rm'd in prev line
	if 'And' in fn:
		clauses = mini_parser(fn)
	else:
		clauses = fn.split(',')

	for clause in clauses:
		eles = clause.split(', ')
		newClause = []
		for ele in eles:
			newEle = ''
			if '~' in ele:
				newEle += '!'
				ele = ele.replace('~','')
			while ' ' in ele or ')' in ele:
				ele = ele.replace(' ','')
				ele = ele.replace(')','')
			newEle += ele
			if ele != '':
				newClause += [newEle]
		if newClause != []:
			newClause = sorted(newClause)
			newF += [newClause]
	newF = sorted(newF)
	return newF

def mini_parser(s):
	parts = []
	while 'And(' in s:
		before, and_part = s.split('And(', 1)
		if before:
			parts.extend(before.rstrip(',').split(','))  # Split before "And"
		and_content, s = and_part.split(')', 1)
		parts.append(and_content)  # Keep "A,B" as one unit
		s = s.lstrip(',')  # Remove leading comma after "And(...)"
	
	if s:
		parts.extend(s.split(','))  # Split remaining parts normally
	
	return [p.strip() for p in parts]


def to_pyEda(fn):
    # takes my function format and converts to pyEda 
    fnStr = ''
    i=0
    for clause in fn:
        j=0
        if i!=0:
            fnStr += ' | '
        for ele in clause:
            if j!=0:
                fnStr += ' & '
            if '!' in ele:
                fnStr += '~'
                ele = ele.replace('!','')
            fnStr += ele
            j+=1
        i+=1
    if fnStr == '':
        fnStr = '0'
    return _expr(fnStr)

