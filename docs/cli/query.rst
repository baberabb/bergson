Query Command
=============

The ``query`` command queries an existing gradient dataset.

Usage
-----

.. code-block:: bash

   python -m your_package query [OPTIONS]

Description
-----------

.. autoclass:: your_package.cli.Query
   :members:
   :undoc-members:

Configuration Options
---------------------

Query Configuration
~~~~~~~~~~~~~~~~~~~

.. autoclass:: your_package.data.QueryConfig
   :members:
   :undoc-members:

Index Configuration
~~~~~~~~~~~~~~~~~~~

.. autoclass:: your_package.data.IndexConfig
   :members:
   :undoc-members:

Examples
--------

Basic query:

.. code-block:: bash

   python -m bergson query \
       runs/query \
       --query_path runs/index \
       --scores_path runs/query/scores \
       --save_index false

Notes
-----

.. warning::
   If the index path already exists and ``save_index`` is True, the command
   will fail to prevent overwriting existing gradients.
