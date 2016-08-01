MinPy installation guide
========================

For regular users
-----------------

.. image:: https://badge.fury.io/gh/dmlc%2Fminpy.svg
    :target: https://badge.fury.io/gh/dmlc%2Fminpy

Docker installation guide is available at :doc:`/get-started/docker`.

MinPy releases are uploaded to PyPI. Just use ``pip`` to install.

::

    pip install minpy

Don't forget to upgrade once in a while to use the latest features!

For developers
--------------

Currently MinPy is going through rapid development (but we do our best
to keep the APIs stable). So it is adviced to do an editable
installation.  Change directory into where the Python package lies and
run ``python setup.py develop`` if you are in a virtual environment,
or ``python setup.py develop --user`` if you are using your system
Python packages. This will ensure a symbolic link to the project, so
you do not have to install a second time when you update this
repository.
