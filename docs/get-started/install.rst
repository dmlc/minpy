MinPy installation guide
========================
.. image:: https://badge.fury.io/gh/dmlc%2Fminpy.svg
    :target: https://badge.fury.io/gh/dmlc%2Fminpy

There are generally three steps to follow:

1. Install MXNet
2. Setup Python package and environment
3. Install MinPy

..
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

It turns out that installing ``openblas`` is necessary, in addition to modify the makefile, to fix `one of the build issues <https://github.com/dmlc/mxnet/issues/572>`_.

Setup Python and its environment
--------------------------------

Refer to MXNet installation document for `Python package installation. <http://mxnet.readthedocs.io/en/latest/how_to/build.html>`_. One of the most common problem a beginner runs into is not setting the environment variable to tell Python where to find the library. Suppose you have installed ``mxnet`` under your home directory and is running bash shell. Put the following line in your ``~/.bashrc`` (or ``~/.bash_profile``)

::

    export PYTHONPATH=~/mxnet/python:$PYTHONPATH

Install MinPy
-------------

Minpy prototypes a pure Numpy interface. To make the interface consistent, please make sure Numpy version >= 1.10.0 before install Minpy.

MinPy releases are uploaded to PyPI. Just use ``pip`` to install after you install MXNet.

::

    pip install minpy

Don't forget to upgrade once in a while to use the latest features!

For developers
^^^^^^^^^^^^^^

Currently MinPy is going through rapid development (but we do our best
to keep stable APIs). So it is adviced to do an editable
installation.  Change directory into where the Python package lies. If
you are in a virtual environment, run ``python setup.py develop``. If
you are using your system Python packages, then run ``python setup.py develop --user``.
This will ensure a symbolic link to the project, so
you do not have to install a second time when you update this
repository.


Docker images for MinPy
-----------------------

Optionally, you may build MinPy/MXNet using `docker <http:www.docker.com>`_.

Build images
^^^^^^^^^^^^

Build Docker images using the ``Dockerfile`` found in the ``docker``
directory.

Just run the following command and it will build MXNet with CUDA
support and MinPy in a row::

    docker build -t dmlc/minpy ./docker/

Start a container
^^^^^^^^^^^^^^^^^

To launch the docker, you need to install `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ first.

Then use ``nvidia-docker`` to start the container with GPU access. MinPy is
ready to use now!

::

    $ nvidia-docker run -ti dmlc/minpy python
    Python 2.7.6 (default, Jun 22 2015, 17:58:13)
    [GCC 4.8.2] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import minpy as np
    >>> ...


Train a model on MNIST to check everything works
^^^^^^^

::

    nvidia-docker run dmlc/minpy python dmlc/minpy/examples/basics/logistic.py --gpus 0
