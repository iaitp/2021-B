from sys import getsizeof
from IPython import get_ipython
from IPython.core.magics.namespace import NamespaceMagics

_nms = NamespaceMagics()
_Jupyter = get_ipython()
_nms.shell = _Jupyter.kernel.shell

def get_vars():
    values = _nms.who_ls()
    locals = []

    for v in values:
        if type(eval(v)).__name__ == "DataFrame":
            locals.append(v)
