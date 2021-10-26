# Data HarPy

Jupyter extension that analyses your data and shows you less computational expensive models (greener) to pick for your task


## Setup
**Create Conda Environment**

We should all be able to reproduce the same python and notebook environment using a conda environment.

- install conda (miniconda is probably enough)
- to create the environment for this project run the following in the project dir `conda env create -f environment.yml`

**Update the environment**
1. Edit the environment.yml file to add a new package
2. update your conda environment by running `conda env update -n 2021-B --file environment.yml  --prune`
3. If someone else has added a new package we can just step 2 locally and have the same environment

**To run Jupyter notebook**
- Jupyter is alreay installed in the conda environment
- to start it simply activate the environment and run `conda activate 2021-B`
