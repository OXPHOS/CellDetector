import os
import shutil


def check_path(path, children=[]):
    """
    Examine whether one directory exist.
    If yes, clean the directory.
    If no, make the directory
    
    :param path: the path to the directory to be checked
    :param children: child dirs of the path dir
    :return: 
    """
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

    if children:
        for child in children:
            os.makedirs(os.path.join(path, child))