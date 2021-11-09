import ipywidgets as widgets
from IPython import get_ipython

dropdown = widgets.Dropdown(
    options=locals,
    description='Number:'
)


btn = widgets.Button(description='Click Me')
btn2 = widgets.Button(description='Regression')
analyser_btn = widgets.Button(description='Create Analyser')



def analyser_btn_eventhandler(obj):
    # otherwise local to the function
    global analyse
    analyse = analyser(eval(dropdown.value), "target")


def btn_eventhandler(obj):
#     call function here
    print('Hello from the {} button!'.format(obj.description))

def btn2_eventhandler(obj):
#     call function here
    print('Hello from the {} button!'.format(obj.description))


btn1.on_click(btn1_eventhandler)
btn2.on_click(btn2_eventhandler)
analyser_btn.on_click(analyser_btn_eventhandler)
