# NUDGE: Natural Dynamics Control of Gene Regulatory Networks

Identifies single-time interventions from Boolean network models to ensure a desired phenotype while preserving natural regulatory dynamics. Free to use, modify, and distribute as per MIT license, please cite or acknowledge this work if used.

## Installation

This package was designed in python 3.9 and should be compatible between python 3.6 and 3.9. Unfortunately the library that implements logic reduction, PyEDA, is no longer maintained and breaks with later versions of python.

```bash
pip install -r requirements.txt
```

For GPU acceleration, one can optionally install cupy, which requires CUDA drivers (see https://cupy.dev/). Without cupy installed, numpy will automatically be used. If one want to run numpy even though cupy is installed, set TURN_CUPY_OFF_OVERRIDE=1 at the top of src/util.py. GPU accelerates monte carlo simulation, LDOI, and mean-field methods, but is not used in the primary NUDGE method.

## Usage

There are several primary use cases where this package can be called from the command line, as summarized below. Instead of command line arguments, this package uses a paramter file. See `input/params.yaml` for all parameters, their default values, and an explanation of each.

### RSC: Recursive Self-Composite 

Prints terminal logic of a node, which maps initial states to the state of that node in all attractors.

Usage: `python src/RSC.py PARAM_FILE`

example PARAM_FILE: /input/params.yaml

Only the "RSC parameters" are used. These parameters will specify the target network, the target node, and parameters for RSC and its approximation. Note that nodes being mutated (fixed to a certain state), can also be specified in the params file.


### NUDGE: Phenotype Kernel for Ephemeral Control

Finds ephemeral controllers for a node, distinguishes minimal robust controllers, and highlights the mechanisms of the controllers.

Usage: `python src/NUDGE.py PARAM_FILE`

Parameter file is the same as the previous case, except both "RSC parameters" and "NUDGE parameters" are used.


### Analyze

Detailed analysis of a controller specified in PARAM_FILE 'controller', in terms of robustness ('include_robustness' is true), and its mechanism of action (if 'find_mechanism' is true). Optionally a visualization of the mechanism, although visual clarity is limited for large networks.

Usage: `python src/analyze.py PARAM_FILE`


### Batch comparison

Compares NUDGE with other methods on a batch of networks. 

Usage: `python src/batch.py PARAM_FILE RUN_TITLE`

Output: csv for the controllers found by each method in output folder, and a pickle of all objects from the run. Example output can be seen in the `output/` folder.

Parameter file is the same as the previous cases, and all parameters are used. RUN_TITLE is any valid string, and will be included in all output files. RUN_TITLE cannot be used if an output already exists with that name, except for 'debug'.


### Plotting Batch Comparison

Plots images corresponding to the output of a batch comparison run (Figure 3 and S3 in original paper), specifically violin plots of error and time, and venn diagram of the controllers found.

Usage: `python src/plot.py`

Parameters corresponding to the input and output files are hard-coded at the top of `plot.py`. Average values for errors and times are printed output, along with p-values between NUDGE and other methods for errors.

### Batch Analysis for Warped Dynamics

Analyze deleterious effects and emergence of de novo attractors on a batch of networks

Usage: `python src/batch_scan.py --params PARAM_FILE --bnet_folder FOLDER_OF_NETWORKS`

Output: pickle file (.pkl) of deleterious effects and emergence of de novo attractors for each network in `/output/` and a corresponding scatter plot in `/output/img/`.


### Information on Notebooks

If you prefer to view the results from a Jupyter notebook, we have provided two notebooks that were used to generate our results for Fig. 4, 5, and 6:

1. Figure4to6.ipynb generates Fig. 4A to 4D, Fig. 5A to 5D , and Fig. 6A to 6C.
2. WrappedDynamicsExample_Figure4to6.ipynb generates Fig. 4E, Fig. 5E , and Fig. 6D.
3. Guideline_analysis.ipynb generates the guidelines for when approximation is required.
4. Heterogeneous_path.ipynb generates the comparison between heterogeneous paths in biological and randomized networks.