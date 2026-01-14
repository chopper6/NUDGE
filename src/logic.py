'''
Logic reduction using espresso heuristic, implementation by PyEDA

3 representations of a function:
F: [['A','B'],['C']]
str: '(A & B) | C'
pyeda function object

'''
from pyeda import inter as eda
import numpy as np
import activity

################################################################################

def reduce(params, fn, max_vars = None):
	fn = fn.simplify()
	if params['approx']:
		if max_vars is None:
			max_vars = params['max_variables']
		while fn_too_complex(params, fn, max_vars):
			fn = activity.discard_one_variable(params, fn)
	fn=exactly_reduce(fn)
	return fn

def exactly_reduce(fn):
	fn = fn.simplify() # this can massively help reduction time, maybe due to to_dnf()
	fn = fn.to_dnf() # this is naively implemented and may cause memory errors...
	if fn.equivalent(1):
		return eda.expr(1)
	elif fn.equivalent(0):
		return eda.expr(0)
	else:
		fn, = eda.espresso_exprs(fn)
		return fn

def fn_too_complex(params, fn, max_vars):
	if (fn.equivalent(1) or fn.equivalent(0)):
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
	tautology = not_to_dnf(params, a) | b
	return reduce(params, tautology).is_one()

def fn_to_str(fn):
	# pyeda function form to string
	F = from_pyEda(fn)
	fn_str = F_to_str(F)
	return fn_str

def str_to_fn(fn_str):
	fn = eda.expr(fn_str.replace('!','~'))
	return fn

def not_to_dnf(params, fn, cap=64):
	# takes pyeda fn in dnf form
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
    return eda.expr(fnStr)

