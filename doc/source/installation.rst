Installation
============

    *Tested on Ubuntu 20.04, Linux-5.8.0*

Conda
-----

It is advised to install the package in a new virtual environment using python 3.9. We use anaconda::

    conda create -n OCDenv python=3.9
    conda activate OCDenv

Then from the root of the OCD_modeling source repository (where the `setup.py` is located), type::

    pip install -r ./doc/source/requirements.txt


Docker
------

We created a docker image with all the necessary pre-installed dependencies. ::

    docker pull sebnaze/ocd-modeling:0.3

To have the docker container automatically start a Jupyter server and run the demo, 
you can simply run ``docker run -it -p 8899:8899 sebnaze/ocd-modeling:0.3`` and then 
type ``localhost:8899/`` in your browser address bar.


