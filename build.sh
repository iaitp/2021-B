#!/bin/bash
conda env create -f environment.yml
eval "$(conda shell.bash hook)"
conda activate 2021-B
inv install-dataharpy
jupyter notebook "Extension Test.ipynb"