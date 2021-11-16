define([
    'base/js/namespace'
], function(
    Jupyter
) {
    function load_ipython_extension() {
        console.log('[DataHarPy] Loading extension with current notebook:', Jupyter.notebook);

        var show_data_harpy = function () {

            Jupyter.notebook.insert_cell_below('code').set_text("from sys import getsizeof\nfrom IPython import get_ipython\nfrom IPython.core.magics.namespace import NamespaceMagics\nimport ipywidgets as widgets\nimport pandas as pd\n_nms = NamespaceMagics()\n_Jupyter = get_ipython()\n_nms.shell = _Jupyter.kernel.shell\nvalues = _nms.who_ls()\nlocals = []\nfor v in values:\n\tif type(eval(v)).__name__ == 'DataFrame':\n\t\tlocals.append(v)\ndata_dropdown = widgets.Dropdown(options=locals,description='Data:', disabled=False)\nfrom backend_widgets import analyser\nimport numpy as np\nselect_data_btn = widgets.Button(description='Select Data')\nanalyser_btn = widgets.Button(description='Analyse Data')\ndef select_data_btn_eventhandler(obj):\n\tglobal selected_data\n\tglobal col_dropdown\n\tselected_data = eval(data_dropdown.value)\n\tcol_dropdown = widgets.Dropdown(options=selected_data.columns,description='Target:', disabled=False)\n\tdisplay(col_dropdown)\n\tdisplay(analyser_btn)\ndef analyser_btn_eventhandler(obj):\n\tglobal analyse\n\tanalyse = analyser(selected_data, col_dropdown.value)\tanalyser_btn.on_click(analyser_btn_eventhandler)\nselect_data_btn.on_click(select_data_btn_eventhandler)\ndisplay(data_dropdown)\ndisplay(select_data_btn)")
            Jupyter.notebook.select_next()
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
