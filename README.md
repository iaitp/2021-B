# Data HarPy Interactive AI CDT Team project

---
### Group Members

| [Alex Davies](https://github.com/neutralpronoun) | [Isabella Degen](https://github.com/isabelladegen) | [Phillip Sloan](https://github.com/phillipSloan) | Tony Fang | 

### Contents
 1. [Overview](#overview)
 2. [Setup](#setup)
 3. [Integration](#integration)
 4. [Frontend](#frontend)
 5. [Backend](#backend)
 6. [Future Work](#futurework)

# <a name="overview">Overview #

### Motivation
Data HarPy is a Jupyter notebook extension that analyses and visualises data and recommends less computational expensive models to 
pick for your task. At the moment Data HarPy works for classification tasks.

### Description
Data HarPy extends the Jupyter notebook toolbar with an additional button. On click of the button an ipywidgets
cell is inserted into the notebook. The pandas dataframe to be analysed can be selected via the dropdown in the cell. 
The data analysis part of the extension is written in pure python and can therefore also be used by itself without
the need to install the  Jupyter notebook extension itself.

### Project Overview
At the beginning of the project we brainstormed different ideas in the area of sustainable AI. We wanted
to work on something that allowed us to learn more about AI and sustainability. We felt that the understanding 
of the shape of data was a key part to help decide what ML algorithm could be fit to classify the data.
Data HarPy is the beginning of such a data analysis and ML algorithm selection tool. It also helps
with feature analysis.

We split the work for Data HarPy into three major parts: 
- Jupyter notebook integration and development environment setup
- ipywidget frontend
- data analysis 'backend'

The different parts of Data HarPy allowed us to work on all of them in parallel. We were meeting
weekly to discuss what we've done and decide what to work on next. We also during the week paired on
some of the tasks to be done.

We followed 'trunk-based development' without feature branches and pull request. Instead we each merge our work into the
main branch frequently, whenever we had a bit of working new functionality on our local machine. 
This meant that we'd commit to our local branch, do 
a `git pull --rebase`, resolve the potential merge conflicts on our machine before pushing back 
to the remote main branch. Further reading on different source control workflows
[here](https://martinfowler.com/articles/branching-patterns.html).

We used a conda environment to keep the Python environments across our computers consistent.

### Repository Structure 
```bash.
├── Extension\ Test.ipynb
├── LICENSE
├── Notebook_example.ipynb
├── README.md
├── backend_widgets.py
├── build.sh
├── dataharpy
│   ├── README.md
│   ├── dataharpy.js
│   └── dataharpy.yaml
├── environment.yml
├── functions_backend.py
└── tasks.py
```

### Tooling
- Jupyter 6.*  and ipywidgets: Frontend
- Conda: Python environment mgr
- [Invoke](https://docs.pyinvoke.org/en/stable/index.html): Python build tool to simplify frequently used tasks


# <a name="setup">Setup #

If you want use Data HarPy you can follow the [Quick Setup](#quicksetup).
For development on Data HarPy follow [Developer Setup](#devsetup).

At the moment both assume that you are using [conda](https://docs.conda.io/en/latest/) to manage your python environment.

## <a name="quicksetup"></a>Quick Setup
- clone this git repo
- run `./build.sh` in the root of the repository

This will setup and activate a new conda environment and start the Jupyter notebook for you.

You might need to modify the file to make it executable or run with sudo:
- `chmod u+x build.sh`
- `sudo bash build.sh`

## <a name="devsetup"></a>Developer Setup
- Clone this git repo
- In the project root run `conda env create -f environment.yml`
- Activate env `conda activate 2021-B`
- Install Data HarPy `inv uninstall-dataharpy`
- Run Jupyter `jupyter notebook`

### Keep Conda Environment up to date
If you need a new Python package you need to add the package to the `environement.yml` file. Then
you can run (in your active conda env)  `conda env update -n 2021-B --file environment.yml  --prune`

*It's a good idea to this everytime you pull changes from main as somebody else might have changed the environment*


### Changing the Data HarPy extension

If you make changes to the Javascript code of the extension you need to reinstall it. To do so
run:
- `inv uninstall-dataharpy`
- `inv install-dataharpy`

*It's a good idea to this everytime you pull changes from main as somebody else might have changed the extension*

# <a name="integration">Integration #

# <a name="frontend">Frontend #

# <a name="backend">Backend #

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


--------------------
# # <a name="futurework">Future Work #
Work that needs doing

### Environment
- Installation via Conda Forge and PiPY
- Extension for Jupyter Lab

### Frontend
- User Testing and adaption of user interface based on these outcomes
- Easier controls directly in the toolbar instead of a cell

### Backend
- User testing of current functionality and extension  based on that
- Extension for tasks beyond classification

--------------------
## Invoke

_We can add any build tasks to the tasks.py file to make things simple for others to run_
 - Documentation: https://docs.pyinvoke.org/en/stable/index.html
 - Show available tasks: `invoke --list` or `inv --list`
 

