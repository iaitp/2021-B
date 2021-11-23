define([
    'base/js/namespace'
], function(
    Jupyter
) {
    function load_ipython_extension() {
        console.log('[DataHarPy] Loading extension with current notebook:', Jupyter.notebook);

        var show_data_harpy = function () {

            Jupyter.notebook.insert_cell_below('code').set_text("from sys import getsizeof\n" +
                "from IPython import get_ipython\n" +
                "from IPython.core.magics.namespace import NamespaceMagics\n" +
                "import ipywidgets as widgets\n" +
                "import pandas as pd\n" +
                "_nms = NamespaceMagics()\n" +
                "_Jupyter = get_ipython()\n" +
                "_nms.shell = _Jupyter.kernel.shell\n" +
                "values = _nms.who_ls()\n" +
                "locals = []\n" +
                "for v in values:\n" +
                "\tif type(eval(v)).__name__ == 'DataFrame':\n" +
                "\t\tlocals.append(v)\n" +
                "data_dropdown = widgets.Dropdown(options=locals,description='Data:', disabled=False)\n" +
                "from backend_widgets import analyser\n" +
                "import numpy as np\n" +
                "select_data_btn = widgets.Button(description='Select Data')\n" +
                "analyser_btn = widgets.Button(description='Analyse Data')\n" +
                "def select_data_btn_eventhandler(obj):\n" +
                "\tglobal selected_data\n" +
                "\tglobal col_dropdown\n" +
                "\tselected_data = eval(data_dropdown.value)\n" +
                "\tcol_dropdown = widgets.Dropdown(options=selected_data.columns,description='Target:', disabled=False)\n" +
                "\tdisplay(col_dropdown)\n" +
                "\tdisplay(analyser_btn)\n" +
                "def analyser_btn_eventhandler(obj):\n" +
                "\tglobal analyse\n" +
                "\tanalyse = analyser(selected_data, col_dropdown.value)\n" +
                "analyser_btn.on_click(analyser_btn_eventhandler)\n" +
                "select_data_btn.on_click(select_data_btn_eventhandler)\n" +
                "display(data_dropdown)\n" +
                "display(select_data_btn)")
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
