Bergson Documentation
---------------------

Bergson is a library for tracing the memory of deep neural nets with gradient-based data attribution techniques.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install bergson

Quick Start
~~~~~~~~~~~~

Build an index of gradients:

.. code-block:: bash

   python -m bergson build runs/quickstart --model EleutherAI/pythia-14m --dataset NeelNanda/pile-10k --truncation

Content Index
~~~~~~~~~~~~

* :ref:`genindex`
