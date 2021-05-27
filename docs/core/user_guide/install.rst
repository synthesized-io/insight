Installation
============

The Synthesized SDK is a python package that can be installed using a pre-built  (``.whl``). It is currently
built and tested for **Python 3.6**.

Linux Installation
------------------

It’s possible to install Synthesized SDK even without access to the Internet. However, you will still need to download
and install the required Python packages.

Install the required system packages:

.. code-block:: bash

    yum update -y && yum install -y python3 python3-devel gcc-c++

Update Python & install prerequisites

.. code-block:: bash

    python3 -m pip install --upgrade pip setuptools

Install required Python packages

.. code-block:: bash

    python3 -m pip install --no-deps -r requirements.txt

The following steps do not require the Internet connection.

Install the Synthesized SDK

.. code-block:: bash

    python3 -m pip install --no-deps synthesized-<version>.whl


Dependencies
------------

==================================================================== ==========================
Package                                                               Minimum supported version
==================================================================== ==========================
`tensorflow <https://setuptools.readthedocs.io/en/latest/>`__                   2.4.1
`tensorflow-probability <https://www.tensorflow.org/probability/>`__            0.12.2
`tensorflow-privacy <https://github.com/tensorflow/privacy>`__                  0.12.2
`numpy <https://numpy.org>`__                                                   1.19.5
`scipy <https://www.scipy.org/>`__                                              1.5.4
`scikit_learn <https://scikit-learn.org>`__                                     0.23.2
`pandas <https://pandas.pydata.org/>`__                                         1.1.5
`seaborn <https://seaborn.pydata.org/>`__                                       0.11.0
`pyemd <https://pypi.org/project/pyemd/>`__                                     0.5.1
`faker <https://faker.readthedocs.io/>`__                                       5.0.1
`simplejson <https://simplejson.readthedocs.io/>`__                             3.17.2
`dataclasses <https://pypi.org/project/dataclasses/>`__                         0.6
`pyyaml <https://pyyaml.org/>`__                                                5.3.1
`rstr <https://pypi.org/project/rstr/>`__                                       2.2.6
==================================================================== ==========================


Setting the License Key
-----------------------

To use the Synthesized SDK, a valid license key is required. This can be set
as an environment variable, for example:

.. code-block:: bash

   export SYNTHESIZED_KEY=XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX

For convenience this can be set in ~/.bashrc or /etc/profile (for all users).

Alternatively, the key can be copied to a permanent hidden directory

.. code-block:: bash

   mkdir ~/.synthesized
   echo YOUR_LICENSE_KEY | cat > ~/.synthesized/key


Testing the installation
------------------------

To test the installation is correct, import the synthesized module in the python interpreter

.. code-block:: bash

   python3 -c "import synthesized; print(synthesized.__version__)"


Additional Technical Details
----------------------------

There is no explicit limit for the size of a dataset and it is rather limited by the size of RAM.

The library can potentially leverage a GPU, but it is not required.
