'''
set of misc functions		
most important for uses purposes is the ability to turn Cupy off for ALL files with TURN_CUPY_OFF_OVERRIDE

'''


import os, math, pickle, sys, itertools, csv, yaml
from datetime import datetime, date
import numpy as np
from string import ascii_uppercase

TURN_CUPY_OFF_OVERRIDE = 0 #turns off CUPY for everything if True
if (TURN_CUPY_OFF_OVERRIDE):
	print("\nWARNING: Cupy is off, see util.py\n")


def import_cp_or_np(try_cupy = True, np_warnings = False):
	if try_cupy and not TURN_CUPY_OFF_OVERRIDE:
		try:
			import cupy as cp
			CUPY = True
		except ImportError:
			import numpy as cp
			CUPY = False
			if not np_warnings:
				import warnings
				warnings.filterwarnings("ignore")
	else:
		import numpy as cp
		CUPY = False
		if not np_warnings:
			import warnings
			warnings.filterwarnings("ignore")

	return CUPY, cp

# unfortunately this has to be after import_cp_or_np() definition
CUPY, cp = import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed
# need to import here for copy_to_larger_dim


#############################################################################################

def load_yaml(yaml_file):
	with open(yaml_file, "r") as f:
		data = yaml.safe_load(f)
	return data

def istrue(d,keys):
	# searches if dictionary d[key0][key1]...[keyn] is True
	if isinstance(keys, list):
		curr_dict = d 
		for k in keys:
			if not k in curr_dict.keys():
				return False
			else:
				curr_dict = curr_dict[k]
		return curr_dict
	else: # assume there is only 1 key then
		return keys in d and d[keys]

def pretty_print(dictionary):
	if dictionary is None or dictionary=={}:
		print("Empty")
	else:
		w = max(len(k) for k in dictionary if dictionary[k] is not None)
		for k,v in dictionary.items():
			if v is not None:
				print("{:<{}} : {}".format(k, w, v))

def init_csv(filename, headers):
	# overwrites if file exists
	# headers is a list 
	with open(filename, mode="w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(headers)


def append_csv(filename, row):
	# row is a list
	with open(filename, mode="a", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(row)

def str2list(string):
	# assumes int
	l = []
	for s in string:
		if s not in ['[',']',',',' ']:
			l+=[int(s)]
	return l

def loop_debug(N, loop, expo=4, parent_fn=None):
	if loop > max(N**expo,N*(expo**expo)): # N*expo^2 in case N is very small like 1
		if parent_fn is not None:
			sys.exit("ERROR: infinite loop in",parent_fn,"!")
		else:
			sys.exit("ERROR: infinite loop!")
	return loop+1

def get_node_dtype(params):
	if istrue(params,['PBN','active']) and istrue(params,['PBN','float']):
		return float 
	else:
		return 'bool' 

def infinite_loop_checker(current_loop, max_loops):
	if current_loop > max_loops:
		print("\nERROR: infinite loop inferred after exceeding",max_loops,'loops.\n')
		assert(0)
	return current_loop+1

def get_uint_dtype(max_val):
	# renamed from 'get_np_int_dtype'
	if max_val<254:
		return np.uint8
	elif max_val<65533:
		return np.uint16
	elif max_val<4294967295: # 2^32
		return np.uint32
	elif max_val<18446744073709551615: # 2^64
		return np.uint64
	else:
		print("Max value too large:",max_val)
		assert(0) # value too large for uint ):

def get_int_dtype(max_val):
	if max_val<127:
		return np.int8
	elif max_val<32767:
		return np.int16
	elif max_val<2147483647: # 2^32
		return np.int32
	elif max_val<9223372036854775807: # 2^64
		return np.int64
	else:
		print("Max value too large:",max_val)
		assert(0) # value too large for int ):

def int_to_bool(num, num_nodes):
   bin_str = format(num, '0' +str(num_nodes) + 'b')
   #print('2bool:',num,bin_str, [x == '1' for x in bin_str[::-1]])
   return [x == '1' for x in bin_str[::-1]]

def copy_to_larger_dim(x, num_copies):
	# array will be 1 dim larger, with num_copies of the original array
	# for example, copying array of shape (4,5) with 3 copies would result in an array of shape (3,4,5)
	# could also try cp.newaxis?
	return cp.tile(x,num_copies).reshape(tuple([num_copies])+x.shape)


def char_in_str(strg,indx,new):
	tmp = list(strg)
	tmp[indx]=str(new)
	return "".join(tmp)

def unique_axis0(X):
	# cupy version shamelessly ripped from: https://stackoverflow.com/questions/58662085/is-there-a-cupy-version-supporting-axis-option-in-cupy-unique-function-any
	if not CUPY:
		return cp.unique(X,axis=0)
	else: # unique with axis not implemented in cupy yet
		if len(X.shape) != 2:
			raise ValueError("Input array must be 2D.")
		sortarr     = X[cp.lexsort(X.T[::-1])]
		mask        = cp.empty(X.shape[0], dtype=cp.bool_)
		mask[0]     = True
		mask[1:]    = cp.any(sortarr[1:] != sortarr[:-1], axis=1)
		return sortarr[mask]

def print_rows(X,title=None):
	if title is not None:
		print(title)
	for x in X:
		print(x)

def alphabet_labels(n):
	alphabet = list(ascii_uppercase)
	power = math.ceil(math.log(n,26)) 
	labels = list(itertools.product(alphabet, repeat=power))
	labels = [''.join(l) for l in labels]
	return labels

###### FROM LIGHT SIMULATION #########



def true_in_dict(d,key):
	if key in d and d[key]:
		return True
	else:
		return False


def none(x):
	if x in [None,'none','None',0,'0','False',False,'',' ']:
		return 1
	else:
		return 0

def bool_true(x):
	if x in [0,'0','False',False,'false','noway','gtfofh']:
		return False
	elif x in [1,'1','True',True,'true','fosho','nodoubt']:
		return True

def timestamp():
	now = datetime.now()
	curr_date = str(date.today())#.strip('2022-')
	curr_time = str(datetime.now().strftime("%H-%M-%S"))
	tstamp = curr_date+'_'+curr_time
	return tstamp

def pickle_it(data, file, use_timestamp=False, verbose=False):
	if use_timestamp:
		tstamp = timestamp()
		if '.pickle' in file:
			file_parts = file.split('.pickle')
			file = file_parts[0] + '_' + str(tstamp) + '.pickle' + file_parts[1]
		else:
			file = file + '_' + str(tstamp) + '.pickle'
	if verbose:
		print("pickling data to file:",file)
	with open(file,'wb') as f:
		pickle.dump(data,f)

def load_pickle(file):
	with open(file, 'rb') as f:
		data = pickle.load(f)
	return data

def check_build_dir(dirr):
	if not os.path.exists(dirr):
		print("\nCreating new directory for output at: " + str(dirr) + '\n')
		os.makedirs(dirr)

def safe_div_array(A,B):
	# a is numerator, b is divisor
	assert(len(A) == len(B))
	z=[]
	for i in rng(A):
		if B[i] == 0: z+=[0]
		else: z+=[A[i]/B[i]]
	return z


def get_timestamp():
	now = datetime.now()
	curr_date = str(date.today()).strip('2020-')
	curr_time = str(datetime.now().strftime("%H-%M-%S"))
	tstamp = curr_date+'_'+curr_time
	return tstamp

