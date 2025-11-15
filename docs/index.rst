Bergson Documentation
=====================

Bergson is a library for tracing the memory of deep neural nets with gradient-based data attribution techniques.

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   api

Installation
------------

.. code-block:: bash

   pip install bergson

Quick Start
-----------

Build an index of gradients:

.. code-block:: bash

   python -m bergson build runs/quickstart --model EleutherAI/pythia-14m --dataset NeelNanda/pile-10k --truncation

Command Line Interface
----------------------

.. autoprogram:: bergson.cli:get_parser()
   :prog: bergson

Content Index
------------------

* :ref:`genindex`

Deep API Reference
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api
