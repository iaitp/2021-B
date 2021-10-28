define([
    'base/js/namespace',
    'jquery',
], function(Jupyter, $) {
    "use strict";
    let bold = false;
    let italic = false;
    let underline = false;
    var load_extension = function() {
        Jupyter.toolbar.add_buttons_group([
            Jupyter.keyboard_manager.actions.register ({
                'help'   : 'Show Data HarPY',
                'icon'   : 'harpies-1649016-1398890',
                'handler': function () {
//                    implement
                }
            }, 'text-bold', 'bold'),
        ]);
    };
    var extension = {
        load_ipython_extension : load_extension
    };
    return extension;
});