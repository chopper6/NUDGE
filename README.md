TODO: 
- change prints to file logging, including running just RSC or just NUDGE
- later: add relv data used in figures to output, and img folder (force add to git)
- double check no f-string usage (else py 3.6+)
- for Ferio: putting up notebook for results, so that reader can follow the process in a clearer context

# NUDGE: Natural Dynamics Control of Gene Regulatory Networks

[general description]

## Installation

This package was designed in python 3.9 and should be compatible between python 3.6 and 3.9. Unfortunately the library that implements logic reduction, PyEDA, is no longer maintained and breaks with later versions of python.

```bash
pip install -r requirements.txt
```

For GPU acceleration, one can optionally install cupy, which requires Cuda drivers (see https://cupy.dev/). Without cupy installed, numpy will automatically be used. If one want to run numpy even though cupy is installed, set TURN_CUPY_OFF_OVERRIDE=1 at the top of src/util.py. GPU accelerates monte carlo simulation, LDOI, and mean-field methods, but is not used in the primary NUDGE method.

## Usage

[not sure how to summarize use cases, likely want to include how certain figures were made]


### RSC: Recursive Self-Composite 

Logs terminal logic of a node, which maps initial states to the state of that node in all attractors.

Usage: `python src/RSC.py PARAM_FILE`

Output: [currently print, will change to file output]

example PARAM_FILE: /input/params.yaml

Only the "RSC parameters" are used. These parameters will specify the target network, the target node, and parameters for RSC and its approximation. Note that nodes being mutated (fixed to a certain state), can also be specified in the params file.


### NUDGE: Phenotype Kernel for Ephemeral Control

Finds ephemeral controllers for a node, distinguishes minimal robust controllers, and highlights the mechanisms of the controllers.

Usage: `python src/NUDGE.py PARAM_FILE`

Output: [currently print a dict, will change to file output?]

Parameter file is the same as the previous case, except both "RSC parameters" and "NUDGE parameters" are used.


### Batch

Compares NUDGE with other methods on a batch of networks. 
[specify figure 2 ect]

Usage: `python src/batch.py PARAM_FILE RUN_TITLE`

Output: csv for the controllers found by each method in output folder, and a pickle of all objects from the run

Parameter file is the same as the previous cases, and all parameters are used. RUN_TITLE is any valid string, and will be included in all output files. RUN_TITLE cannot be used if an output already exists with that name, except for 'debug'.


### Analyze

Detailed analysis of a controller specified in PARAM_FILE 'controller', in terms of robustness (if PARAM_FILE specifies 'include robustness'), and its mechanism of action (if PARAM_FILE specifies 'find_mechanism'). 

Usage: `python src/analyze.py PARAM_FILE`


### Plotting

[need to update this]

Usage: `python src/plot.py PLOT_TITLE`

Output: violin plots of error and time, and venn diagram of controllers found

Where PLOT_TITLE is any valid string that will be included in the output files, and existing names can be reused.
[also add a few to params?]


### Preprocessing 

Usage: `python plot.py PLOT_TITLE [sort | targets]`

The sorts networks 

[TODO: a little preprocessing like target selection]
