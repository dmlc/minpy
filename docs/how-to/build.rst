How to build and release
========================

Working on a development version of MinPy is most convenient when
installed in editable mode. Refer to :doc:`/get-started/install` for
instructions. This way changes will appear directly in the installed
package.

MinPy also needs MXNet to run its GPU operators. There is also a
regular and a development installtion for MXNet. Please refer to
MXNet's document for details. MXNet has some C++ parts, so it requires
compilation. But MinPy is pure Python, so a ``pip`` installation is
enough.

When contributing, make sure coding convention is consistent. Run
``yapf -i <python-file>`` to auto format source files. `Google Python
Style Guide <https://google.github.io/styleguide/pyguide.html>`_ is a
good stop for advice.

MinPy version numbers conform to `Semver <http://semver.org/>`_
rules. To ensure consistency between version numbers on PyPI and git
tags, there are a few utility scripts at the root of the
repositry.

First run ``./install_hooks.sh`` once to install some Git hooks. This
will automatically tag your commits upon version change, and push them
if necessary. Now after you make some changes, run ``./bump_version (major |
minor | patch)`` to increment the version number. Then make a commit
and push to upstream. If your Git version is not too old, it will push the tags
along the commit. Otherwise you would have to push the tags manually.

Travis CI will test and build the commit, and if there is a tag, release new version to PyPI.

