Build Command
=============

The ``build`` command creates a gradient dataset from your input data.

Usage
-----

.. code-block:: bash

   python -m your_package build [OPTIONS]

Description
-----------

.. autoclass:: your_package.cli.Build
   :members:
   :undoc-members:

Configuration Options
---------------------

The build command uses ``IndexConfig`` for configuration:

.. autoclass:: your_package.data.IndexConfig
   :members:
   :undoc-members:

Examples
--------

Basic build:

.. code-block:: bash

   python -m bergson build \
       --run_path runs/index

Notes
-----

.. important::
   Either ``save_index`` must be True or ``skip_preconditioners`` must be False.
