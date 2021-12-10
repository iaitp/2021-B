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

### Project Management Overview
At the beginning of the project we brainstormed different ideas in the area of sustainable AI. We wanted
to work on something that allowed us to learn more about AI and sustainability. We felt that the understanding 
of the shape of data was a key part to help decide what ML algorithm could be fit to classify the data.
Data HarPy is the beginning of such a data analysis and ML algorithm selection tool. It also helps
with feature analysis.

We split the work for Data HarPy into three major parts: 
- Jupyter notebook integration and development environment setup
- ipywidget frontend
- data analysis 'backend'

The different parts of Data HarPy allowed us to work on all of them in parallel. We used a Trello board to keep track 
of ideas and tasks and we were meeting weekly to discuss what we've done and decide what to work on next. 
During the week we also paired on some of the tasks.

We followed 'trunk-based development' without feature branches and pull request. Instead, we each merged our work into the
main branch whenever we had a bit of working new functionality on our local machine. 
This meant that we'd commit to our local branch, do 
a `git pull --rebase`, resolve the potential merge conflicts on our machine before pushing back 
to the remote main branch. Further reading on different source control workflows:
[Martin Fowler Source Control Workflows](https://martinfowler.com/articles/branching-patterns.html).

We used a conda environment to keep the Python environments across our computers consistent.

### Repository Structure 
```bash.
├── Extension\ Test.ipynb -> Notebook to test the extension with
├── LICENSE
├── Notebook_example.ipynb -> Notebook that directly calls the backend
├── README.md
├── build.sh -> quick setup
├── dataharpy -> Jupyter extension
│   ├── README.md
│   ├── dataharpy.js
│   └── dataharpy.yaml
├── environment.yml -> Conda environment file
├── backend_widgets.py -> backend
├── functions_backend.py -> backend
└── tasks.py -> Invoke file to install and uninstall the extention
```

### Tooling
- Jupyter 6.*  and ipywidgets: Frontend
- Conda: Python environment mgr
- Python 3.9.7: Backend
- [Invoke](https://docs.pyinvoke.org/en/stable/index.html): Python build tool to simplify frequently used tasks


# <a name="setup">Setup #

If you want to use Data HarPy you can follow the [Quick Setup](#quicksetup).
For manual install and more details on development on Data HarPy follow [Developer Setup](#devsetup).

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

### Extending the build file
Any useful command for developers can be added to the `tasks.py` file to make things simpler for 
other to run.

To show what tasks are available run: `invoke --list` or `inv --list`

# <a name="integration">Integration #

Currently, Data HarPy integrates with Jupyter notebooks and adds a new button to the toolbar:

![Data HarPy Toolbar Extension](https://github.com/iaitp/2021-B/blob/main/img/jupyterextension.png?raw=true)

All the code for the extension lives in the dataharpy subfolder of the repository. To figure out how to
extend the notebook we looked at the documentation and existing extensions.

Useful links:
- [Extending Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/stable/extending/index.html)
- [Existing Extensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions/) *This is also the repository we'd need to push to in order to make our extension part of the existing nbextensions*
- [Blog post on: Ways of extending the Jupyter Notebook](https://mindtrove.info/4-ways-to-extend-jupyter-notebook/) *This is a tad out of date but still useful*

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


# # <a name="futurework">Future Work #
Work that needs doing

### Integration
- Installation via Conda Forge and PiPY
- Add extension to the existing nbextensions which would allow installation via Jupyter extension manager
- Extension for Jupyter Lab

### Frontend
- User Testing and adaption of user interface based on these outcomes
- Easier controls directly in the toolbar instead of a cell

### Backend
- User testing of current functionality and extension  based on that
- Extension for tasks beyond classification



