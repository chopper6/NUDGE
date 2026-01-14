'''
Activity approximation, and its use to trim Boolean functions
'''
import re, random, math
import logic
from pyeda.boolalg.expr import Variable, expr, exprvar

######################################################################

def discard_one_variable(params, fn):
	activity = approx_activity(params, fn)
	min_activity_var = min(activity, key=activity.get) 
	fn = randomly_assign_var(fn, min_activity_var)
	fn = fn.simplify()
	return fn

def randomly_assign_var(f, var):
	return f.compose({var: random.choice([0,1])})  

def get_variables(fn):
	if isinstance(fn,Variable):
		return [fn]
	else:
		return list(fn.support)

def get_sample(params, variables):
	return {v:random.randint(0,1) for v in variables} 

def approx_activity(params, fn):
	variables = get_variables(fn)
	num_vars = len(variables)
	activities = {var:0 for var in variables}
	for i in range(params['activity_samples']):
		sample= get_sample(params, variables)
		y = int(fn.restrict(sample))

		for v in variables:
			samplej = sample.copy() 
			samplej[v] = 1-samplej[v]# flip only j
			yj = int(fn.restrict(samplej))
			if y!=yj:
				activities[v]+=1
	for v in variables:
		activities[v] /= params['activity_samples']
	return activities
