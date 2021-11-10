define([
    'base/js/namespace'
], function(
    Jupyter
) {
    function load_ipython_extension() {
        console.log('[DataHarPy] Loading extension with current notebook:', Jupyter.notebook);

        var show_data_harpy = function () {

            Jupyter.notebook.insert_cell_below('code').set_text("from sys import getsizeof\nfrom IPython import get_ipython\nfrom IPython.core.magics.namespace import NamespaceMagics\nimport ipywidgets as widgets\nimport pandas as pd\n_nms = NamespaceMagics()\n_Jupyter = get_ipython()\n_nms.shell = _Jupyter.kernel.shell\nvalues = _nms.who_ls()\nlocals = []\nfor v in values:\n\tif type(eval(v)).__name__ == 'DataFrame':\n\t\tlocals.append(v)\ndropdown = widgets.Dropdown(options=locals,vdescription='Number:', disabled=False)\nfrom backend_widgets import analyser\nimport numpy as np\nanalyser_btn = widgets.Button(description='Analyse Data')\ndef analyser_btn_eventhandler(obj):\n\tglobal analyse\n\tanalyse = analyser(eval(dropdown.value), 'target')\nanalyser_btn.on_click(analyser_btn_eventhandler)")
            Jupyter.notebook.select_next()
            Jupyter.notebook.execute_cell()
            Jupyter.notebook.get_selected_cell().set_text("display(dropdown)\ndisplay(analyser_btn)")
            Jupyter.notebook.execute_cell()
            Jupyter.notebook.get_selected_cell().set_text("# Select a DataFrame to analyse:")
        };


        var data_harpy_button = {
            icon: 'fa-bar-chart', // a font-awesome icon
            help    : 'Show Data HarPY',
            help_index : 'zz',
            handler : show_data_harpy
        };
        var prefix = 'dataharpy';
        var action_name = 'show-data-harpy';

        var full_action_name = Jupyter.keyboard_manager.actions.register(data_harpy_button, action_name, prefix);
        Jupyter.toolbar.add_buttons_group([full_action_name]);
    }

    return {
        load_ipython_extension: load_ipython_extension
    };
});
