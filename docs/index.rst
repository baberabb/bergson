Bergson Documentation
=====================

Bergson is a library for tracing the memory of deep neural nets with gradient-based data attribution techniques.

Installation
------------

.. code-block:: bash

   pip install bergson

Quick Start
-----------

Build an index of gradients:

.. code-block:: bash

   bergson build runs/quickstart --model EleutherAI/pythia-14m --dataset NeelNanda/pile-10k --truncation

Load the gradients:

.. code-block:: python

   from pathlib import Path
   from bergson import load_gradients

   gradients = load_gradients(Path("runs/quickstart"))

API Reference
--------------

.. toctree::
   :maxdepth: 4

   cli

.. toctree::
   :maxdepth: 2

   api


Content Index
------------------

* :ref:`genindex`
