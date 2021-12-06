define([
    'base/js/namespace'
], function(
    Jupyter
) {
    function load_ipython_extension() {
      // Checker to ensure the .js script is loading into your browser.
        console.log('[DataHarPy] Loading extension with current notebook:', Jupyter.notebook);

        // function that executes when the Data HarPy button is clicked.
        var show_data_harpy = function () {
            //insert_cell_below
            Jupyter.notebook.insert_cell_below('code').set_text("from sys import getsizeof\n" +
                "from IPython import get_ipython\n" +
                "from IPython.core.magics.namespace import NamespaceMagics\n" +
                "import ipywidgets as widgets\n" +
                "import pandas as pd\n" +
                //importing the Magics from iPython to allow who_ls to be ran
                "_nms = NamespaceMagics()\n" +
                // Gets the global IPython InteractiveShell instance
                "_Jupyter = get_ipython()\n" +
                "_nms.shell = _Jupyter.kernel.shell\n" +
                //provides a list of all variables within the jupyter IPython Kernel
                "values = _nms.who_ls()\n" +
                "locals = []\n" +
                //loop through the variables and store all of the pd DataFrames
                "for v in values:\n" +
                "\tif type(eval(v)).__name__ == 'DataFrame':\n" +
                "\t\tlocals.append(v)\n" +
                // Creating a widget for a drop down box to allow user to select
                // the DataFrame they want to analyse
                "data_dropdown = widgets.Dropdown(options=locals,description='Data:', disabled=False)\n" +
                //importing our backend python scripts
                "from backend_widgets import analyser\n" +
                "import numpy as np\n" +
                // Buttons users click to select dataframe / analyse data respectively
                "select_data_btn = widgets.Button(description='Select Data')\n" +
                "analyser_btn = widgets.Button(description='Analyse Data')\n" +
                // function is executed when the select data btn is clicked
                "def select_data_btn_eventhandler(obj):\n" +
                "\tglobal selected_data\n" +
                "\tglobal col_dropdown\n" +
                // eval(data_dropdown.value) provides the actual instance of the dataframe
                "\tselected_data = eval(data_dropdown.value)\n" +
                // from the dataframe, this dropdown allows a specific column to
                // be selected for analysis
                "\tcol_dropdown = widgets.Dropdown(options=selected_data.columns,description='Target:', disabled=False)\n" +
                // when the select data button is clicked then the second drop dropdown
                // and button are revealed.
                "\tdisplay(col_dropdown)\n" +
                "\tdisplay(analyser_btn)\n" +
                //function creates an analyser object which we created
                "def analyser_btn_eventhandler(obj):\n" +
                "\tglobal analyse\n" +
                //creates an analyse object
                "\tanalyse = analyser(selected_data, col_dropdown.value)\n" +
                "analyser_btn.on_click(analyser_btn_eventhandler)\n" +
                "select_data_btn.on_click(select_data_btn_eventhandler)\n" +
                //ensuring the analyser and drop down are visible when the
                // Data HarPy button is pressed.
                "display(data_dropdown)\n" +
                "display(select_data_btn)")
            // selects the cell we've just inserted all this code into
            Jupyter.notebook.select_next()
            // executes the cell
            Jupyter.notebook.execute_cell()
            // replaces all the code with a nice message instead
            Jupyter.notebook.get_selected_cell().set_text("# Select a DataFrame to analyse:")
        };

        // framework to instantiate the button on the jupyter notebook dashboard
        var data_harpy_button = {
            icon: 'fa-bar-chart', // The bar chart icon used (font-awesome icon)
            help    : 'Show Data HarPY', //this is displayed when you hover
            help_index : 'zz',
            handler : show_data_harpy // calls this function on click
        };
        //making
        var prefix = 'dataharpy';
        var action_name = 'show-data-harpy';
        // registers the button with Jupyter Notebook
        var full_action_name = Jupyter.keyboard_manager
                                      .actions
                                      .register(data_harpy_button, action_name, prefix);
        //Adds button to toolbar
        Jupyter.toolbar.add_buttons_group([full_action_name]);
    }

    return {
        load_ipython_extension: load_ipython_extension
    };
});
