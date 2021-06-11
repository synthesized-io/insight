Installation
============

The Synthesized SDK is a python 3 compatible package that is installed using the provided pre-built wheel (``.whl``)
file. Currently, wheels can be provided for Python 3.6, 3.7 and 3.8 on both Windows and Linux x86_64 platforms.

.. _installation-label:

It is assumed that you have an existing Python 3 installation that is compatible with the provided wheel file.
Additionally, it is recommended to work within a clean Python environment (e.g created using ``virtualenv`` or ``venv``)
to ensure the compatibility of all dependencies.

Before starting, ensure that ``pip``, ``setuptools`` and ``wheel`` are installed and up to date

.. tabs::

    .. group-tab:: Linux

        .. code-block:: bash

            python3 -m pip install --upgrade pip setuptools wheel

    .. group-tab:: Windows

        .. code-block:: powershell

            py -m pip install --upgrade pip setuptools wheel


Next, install the package using ``pip``:

.. tabs::

    .. group-tab:: Linux

        .. code-block:: bash

            python3 -m pip install synthesized-<version>-linux_x86_64.whl


    .. group-tab:: Windows

        Note: `Visual Studio Build Tools <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019>`__
        must be pre-installed before completing this step.

        .. code-block:: powershell

            py -m pip install synthesized-<version>-win_amd64.whl


Setting the License Key
-----------------------

To use the Synthesized SDK, a valid license key is required. This can be set
as an environment variable, for example:

.. tabs::

    .. group-tab:: Linux

        .. code-block:: bash

            export SYNTHESIZED_KEY="XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX"

    .. group-tab:: Windows

        .. code-block:: powershell

            $Env:SYNTHESIZED_KEY="XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX"

Alternatively, the key can be copied to a permanent hidden folder:

.. tabs::

    .. group-tab:: Linux

        .. code-block:: bash

            mkdir ~/.synthesized
            echo YOUR_LICENSE_KEY > ~/.synthesized/key

    .. group-tab:: Windows

        .. code-block:: powershell

            mkdir ~/.synthesized
            echo YOUR_LICENSE_KEY > ~/.synthesized/key


Testing the installation
------------------------

To test the installation is correct, import the synthesized module in the python interpreter

.. tabs::

    .. group-tab:: Linux

        .. code-block:: bash

            python3 -c "import synthesized; print(synthesized.__version__)"

    .. group-tab:: Windows

        .. code-block:: powershell

            py -c "import synthesized; print(synthesized.__version__)"


Dependencies
------------

=======================================================================  ==========================
Package                                                                   Minimum supported version
=======================================================================  ==========================
`tensorflow <https://setuptools.readthedocs.io/en/latest/>`__                   2.2.1
`tensorflow-probability <https://www.tensorflow.org/probability/>`__            0.10.1
`numpy <https://numpy.org>`__                                                   1.18.4
`scipy <https://www.scipy.org/>`__                                              1.5.4
`scikit_learn <https://scikit-learn.org>`__                                     0.23.2
`pandas <https://pandas.pydata.org/>`__                                         1.1.5
`seaborn <https://seaborn.pydata.org/>`__                                       0.11.0
`pyemd <https://pypi.org/project/pyemd/>`__                                     0.5.1
`faker <https://faker.readthedocs.io/>`__                                       5.0.1
`simplejson <https://simplejson.readthedocs.io/>`__                             3.17.2
`pyyaml <https://pyyaml.org/>`__                                                5.3.1
`rstr <https://pypi.org/project/rstr/>`__                                       2.2.6
=======================================================================  ==========================

For Python 3.6 compatibility, the following are also required:

=======================================================================  ==========================
Package                                                                   Minimum supported version
=======================================================================  ==========================
`dataclasses <https://pypi.org/project/dataclasses/>`__                          0.6
=======================================================================  ==========================

Additional Technical Details
----------------------------

There is no explicit limit for the size of a dataset; this is limited by the size of RAM.

The library can potentially leverage a GPU, but it is not required.
