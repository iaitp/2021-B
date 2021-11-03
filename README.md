# Data HarPy CDT Team project

A Jupyter extension that analyses your data and shows you less computational expensive models to pick for your task. See [link](dataharpy/README.md)

## Tooling
- Jupyter 6.* with extensions: this is basically our frontend
- Conda: Python environment mgr
- Invoke: Python build tool to simplify frequently used tasks


## Development Environment Setup
We're using a conda environment for the development of this extension. This makes it ease for all devs to have the same setup.

**Create Conda Environment**
- install conda (miniconda is probably enough)
- in the project dir run `conda env create -f environment.yml`

**Activate Conda Environment**
- `conda activate 2021-B`
*Make sure you also use this env in your IDE*

**Change and/or Update Conda Environment**
1. Add any new package you need to the environment.yml file
2. Update the conda environment by running `conda env update -n 2021-B --file environment.yml  --prune`

*It's a good idea to this everytime after you pulled as somebody might have changed the environment*

## Install and Update the Extension

_You can skip the first step if you've never installed the extension before_
_These commands need to be run with the project's conda env active and in the root folder of the project_
1. `inv uninstall-dataharpy`
2. `inv install-dataharpy`

## Run Jupyter
Jupyter is already part of the environment. To start it run 
- `conda activate 2021-B`
- `jupyter notebook`

## Credits
<TODO>

--------------------
## Invoke

_We can add any build tasks to the tasks.py file to make things simple for others to run_
 - Documentation: https://docs.pyinvoke.org/en/stable/index.html
 - Show available tasks: `invoke --list` or `inv --list`
 

