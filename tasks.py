from invoke import task

@task
def list_extensions(c):
    c.run("jupyter nbextension list; jupyter serverextension list; jupyter bundlerextension list")

@task
def uninstall_dataharpy(c):
    # c.run("jupyter nbextension disable dataharpy/dataharpy --sys-prefix")
    c.run("jupyter nbextension uninstall dataharpy --sys-prefix")

@task
def install_dataharpy(c):
    c.run("jupyter nbextension install dataharpy --sys-prefix")
    c.run("jupyter nbextension enable dataharpy/dataharpy --sys-prefix")

