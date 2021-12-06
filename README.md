# Data HarPy Interactive AI CDT Team project

---
## Group Members

| [Alex Davies](https://github.com/neutralpronoun) | [Isabella Degen](https://github.com/isabelladegen) | [Phillip Sloan](https://github.com/phillipSloan) | [Tony Fang](https://github.com/) | 
| - | - | - | - |

---

### 1. [Introduction](#introduction)
### 2. [Background](Report/Background/README.md)
### 3. [System Implementation](Report/System_Implementation/README.md)
### 4. [Evaluation](Report/Evaluation/README.md)
### 5. [Conclusion](Report/Conclusion/README.md)

# Introduction

A Jupyter extension that analyses your data and shows you less computational expensive models to pick for your task. See [link](dataharpy/README.md)

## Tooling
- Jupyter 6.* with extensions: this is basically our frontend
- Conda: Python environment mgr
- Invoke: Python build tool to simplify frequently used tasks

## Development Environment Setup
We're using a conda environment for the development of this extension. This makes it ease for all devs to have the same setup.

### Quick setup

Just run the build.sh script using bash:

`bash build.sh`

you might need to modify the file to make it executable or run with sudo:

`
chmod u+x build.sh

sudo bash build.sh
`

### Manual setup

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

## Using the plugin
Once you're in a notebook, prepare a pandas dataframe.

In this version feature columns can be any name, but your target feature ("Y") should be named "target".

Once you have a dataframe, click the plugin's button on the Jupyter toolbar (a bar chart icon), and you'll be prompted to select a dataframe to use.
Select a dataframe and press analyse, and dataharpy will generate some useful tabs with information about your data.

Note: Data is normalised by default when `.analyse(x1,x2)` is called. Calculated values (e.g. per-class mean separation) are therefore calculated using the normalised data.

## Output information

The first tab allows you to plot features against each other, along with histograms aligned with the scatter plot. If the same feature is selected twice then the output is a per-class violin plot.

The second tab uses the same `.compare(x1,x2)` method, and finds the most useful features in the context of your data, using an sklearn random forest.

Other tabs display useful per-class and per-feature information.

## Credits
<TODO>

--------------------
## Invoke

_We can add any build tasks to the tasks.py file to make things simple for others to run_
 - Documentation: https://docs.pyinvoke.org/en/stable/index.html
 - Show available tasks: `invoke --list` or `inv --list`
 

