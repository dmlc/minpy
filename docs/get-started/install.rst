MinPy installation guide
========================
.. image:: https://badge.fury.io/gh/dmlc%2Fminpy.svg
    :target: https://badge.fury.io/gh/dmlc%2Fminpy

There are generally three steps to follow:

* Install MXNet
* Setup Python package and environment
* Install Minpy

Docker installation guide is available at :doc:`/get-started/docker`.

Install MXNet
-------------

The full guide of MXNet is `here  <http://mxnet.readthedocs.io/en/latest/how_to/build.html>`_ to build and install MXNet.
Below, we give the common steps for Linux and OSX.

On Ubuntu/Debian
^^^^^^^^^^^^^^^^
Install the dependencies and build mxnet by
::

    sudo apt-get update
    sudo apt-get install -y build-essential git libatlas-base-dev libopencv-dev
    git clone --recursive https://github.com/dmlc/mxnet
    cd mxnet; make -j$(nproc)

On OSX
^^^^^^
Do the following instead. 
::

    brew update
    brew tap homebrew/science
    brew info opencv
    brew install opencv
    brew info openblas
    brew install openblas
    git clone --recursive https://github.com/dmlc/mxnet
    cd mxnet; cp make/osx.mk ./config.mk; make -j$(sysctl -n hw.ncpu)

Setup Python and its environment
--------------------------------

Refer to MXNet installation document for `Python package installation. <http://mxnet.readthedocs.io/en/latest/how_to/build.html>`_. One of the most common mistake is forget to set the environment variable. Suppose you have installed ``mxnet`` under your home directory and is running bash shell. Put the following line in your `~/.bashrc` (or `~/.bash_profile`)

::

    export PYTHONPATH=~/mxnet/python

Install Minpy
-------------

MinPy releases are uploaded to PyPI. Just use ``pip`` to install after you install MXNet.

::

    pip install minpy

Don't forget to upgrade once in a while to use the latest features!

For developers
^^^^^^^^^^^^^^

Currently MinPy is going through rapid development (but we do our best
to keep the APIs stable). So it is adviced to do an editable
installation.  Change directory into where the Python package lies and
run ``python setup.py develop`` if you are in a virtual environment,
or ``python setup.py develop --user`` if you are using your system
Python packages. This will ensure a symbolic link to the project, so
you do not have to install a second time when you update this
repository.
