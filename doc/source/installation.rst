Installation
============

    *Tested on Ubuntu 20.04, Linux-5.8.0*


It is advised to install the package in a new virtual environment using python 3.9. We use anaconda::

    conda create -n OCDenv python=3.9
    conda activate OCDenv

Then from the root of the OCD_modeling source repository (where the `setup.py` is located), type::

    pip install -r ./doc/source/requirements.txt