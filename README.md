# Data HarPy Interactive AI CDT Team project #

### Group Members

| [Alex Davies](https://github.com/neutralpronoun) | [Isabella Degen](https://github.com/isabelladegen) | [Phillip Sloan](https://github.com/phillipSloan) | Tony Fang | 

### Contents
 1. [Overview](#overview)
 2. [Setup](#setup)
 3. [Jupyter Integration](#integration)
 4. [Frontend and Backend ](#frontendandbackend)
 5. [Future Work](#futurework)

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
- data analysis backend

The different parts of Data HarPy allowed us to work on all of them in parallel. We used a Trello board to keep track 
of ideas and tasks, and we were meeting weekly to discuss what we've done and decide what to work on next. 
During the week we also paired on some tasks.

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

# <a name="integration">Jupyter Integration #

Currently, Data HarPy integrates with Jupyter notebooks and adds a new button to the toolbar:

![Data HarPy Toolbar Extension](https://github.com/iaitp/2021-B/blob/main/img/jupyterextension.png?raw=true)

All the code for the extension lives in the dataharpy subfolder of the repository. To figure out how to
extend the notebook we looked at the documentation and existing extensions.

Useful links:
- [Extending Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/stable/extending/index.html)
- [Existing Extensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions/) *This is also the repository we'd need to push to in order to make our extension part of the existing nbextensions*
- [Blog post on: Ways of extending the Jupyter Notebook](https://mindtrove.info/4-ways-to-extend-jupyter-notebook/) *This is a tad out of date but still useful*

# <a name="frontendandbackend">Frontend and Backend #

IPython Widgets are used as user interface and to visualise the data analysis. 
The cell that gets inserted to the notebook has
a dropdown that allows the user to select any panda's dataframe in their notebook:

![Data HarPy Select dataframe](https://github.com/iaitp/2021-B/blob/main/img/selectdataframe.jpg?raw=true)

*In this version feature columns can have any name, but the target feature ("Y") needs to be named "target".*

Pressing analyse will generate an analysis interface with some useful tabs of information about your data as
well as a recommendation tab on what simple algorithm could be used to separate your features for classification.

![Data HarPy Analysis Interface](https://github.com/iaitp/2021-B/blob/main/img/analysisinterface.jpg?raw=true)

Note that the data is normalised by default when `.analyse(x1,x2)` is called. 
Calculated values (e.g. per-class mean separation) are therefore calculated using the normalised data.

The **Compare Features** tab plots features against each other, along with histograms aligned with the scatter plot. 
If the same feature is selected twice then the output is a per-class violin plot.

The **Best Features** tab uses the same `.compare(x1,x2)` method, and finds the most useful features in the data,
using an sklearn random forest.

The **Calculations** and **Functions** tabs display useful per-class and per-feature information.

The **Recommendations** tab shows a what algorithm could be used to separate the selected features. It 
uses the following model to come up with a recommendation:

![Data HarPy Recommendations](https://github.com/iaitp/2021-B/blob/main/img/recommendations.png?raw=true)

*Note that the python scripts are also well documented for users who want to directly call the python functiosn
instead of using the extension.

# <a name="futurework">Future Work #
There's loads of ways to improve Data HarPy. One key activity is to do user testing beyond ensuring 
that the installation works on various different computers. Like we've done. The user testing would guide us
which of the following work is most important

### Integrations and Installation
- Installation via Conda Forge and PiPY
- Add extension to the existing nbextensions which would allow installation via Jupyter extension manager
- Extension for Jupyter Lab

### Frontend
- Simplifying the controls
- Add the dataframe selection dropdown to toolbar instead of into a cell
- Visualisation on how the algorithm recommendation has been done

### Backend
- Extension for tasks beyond classification



