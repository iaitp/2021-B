# Imports
from sys import getsizeof
from IPython import get_ipython
from IPython.core.magics.namespace import NamespaceMagics
import ipywidgets as widgets
import pandas as pd

# Getting the kernel in an object
_nms = NamespaceMagics()
_Jupyter = get_ipython()
_nms.shell = _Jupyter.kernel.shell

# Using the %who command from IPython to get a list of the current variables
values = _nms.who_ls()

# Storing a list of the DataFrames in the kernel
locals = []
for v in values:
    if type(eval(v)).__name__ == "DataFrame":
        locals.append(v)

# Drop down IPyWidget showing the Dataframes
dropdown = widgets.Dropdown(
    options=locals,
    description='Number:',
    disabled=False,
)

# Buttons choice for the user to select different types of analysis
btn1 = widgets.Button(description='Classification')
btn2 = widgets.Button(description='Regression')

def btn1_eventhandler(obj):
    print('Hello from the {} button!'.format(obj.description))

def btn2_eventhandler(obj):
    print('Hello from the {} button!'.format(obj.description))


btn1.on_click(btn1_eventhandler)
btn2.on_click(btn2_eventhandler)
