Command Line Interface
======================

Overview
--------

This package provides a command-line interface for building and querying gradient datasets.

Quick Start
-----------

Build a gradient dataset:

.. code-block:: bash

   python -m bergson build --run_path runs/index

Query a dataset on the fly:

.. code-block:: bash

   python -m bergson query runs/query --query_path runs/index --scores_path runs/query/scores

Commands
--------

.. toctree::
   :maxdepth: 2

   build
   query
