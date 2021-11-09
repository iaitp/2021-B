define([
    'base/js/namespace'
], function(
    Jupyter
) {
    function load_ipython_extension() {
        console.log('[DataHarPy] Loading extension with current notebook:', Jupyter.notebook);

        var show_data_harpy = function () {
            //console.log("[DataHarPY] Show button clicked")
            Jupyter.notebook.insert_cell_below('code').set_text("# This is our cell\nHello");

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
