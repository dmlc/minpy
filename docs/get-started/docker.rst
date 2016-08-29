Docker images for MinPy
=======================

Build images
------------

Build Docker images using the ``Dockerfile`` found in the ``docker``
directory.

Just run the following command and it will build MXNet with CUDA
support and MinPy in a row.

::
    docker build -t dmlc/minpy ./docker/

Start a container
-----------------

To launch the docker, you need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) first.

Then use ``nvidia-docker`` to start the container with GPU access. MinPy is
ready to use now!::

    $ nvidia-docker run -ti dmlc/minpy python
    Python 2.7.6 (default, Jun 22 2015, 17:58:13)
    [GCC 4.8.2] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import minpy as np
    >>> ...


Train a model on MNIST to check everything works
-----------------

::
    nvidia-docker run dmlc/minpy python dmlc/minpy/examples/basics/logistic.py --gpus 0
