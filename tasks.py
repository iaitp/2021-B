from invoke import task

@task
def install_nbextensions(c):
    c.run("jupyter contrib nbextension install --sys-prefix")
