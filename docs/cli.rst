Command Line Interface
======================

Bergson provides a command-line interface with two main commands: ``build`` and ``query``.

Usage
-----

.. code-block:: bash

   bergson {build,query} [OPTIONS]

Commands
--------

Build
~~~~~

Build a gradient dataset.

.. code-block:: bash

   bergson build [OPTIONS]

.. autoclass:: bergson.__main__.Build
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
^^^^^^^^^^^^^

.. autoclass:: bergson.data.IndexConfig
   :members:
   :undoc-members:

**Example:**

.. code-block:: bash

   bergson build runs/my-run \
       --cfg.model EleutherAI/pythia-14m \
       --cfg.dataset NeelNanda/pile-10k \
       --cfg.save_index true

Query
~~~~~

Query the gradient dataset.

.. code-block:: bash

   bergson query [OPTIONS]

.. autoclass:: bergson.__main__.Query
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
^^^^^^^^^^^^^

Query Configuration:

.. autoclass:: bergson.data.QueryConfig
   :members:
   :undoc-members:

Index Configuration:

.. autoclass:: bergson.data.IndexConfig
   :members:
   :undoc-members:

Data Configuration:

.. autoclass:: bergson.data.DataConfig
   :members:
   :undoc-members:

Attention Configuration:

.. autoclass:: bergson.data.AttentionConfig
   :members:
   :undoc-members:

**Example:**

.. code-block:: bash

   bergson query \
       --query_cfg.query_path /path/to/query \
       --query_cfg.scores_path /path/to/scores \
       --index_cfg.run_path /path/to/index
