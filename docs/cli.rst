Command Line Interface
======================

Bergson provides a command-line interface with two main commands: ``build`` and ``query``.

Usage
-----

.. code-block:: bash

   bergson {build,query,score} [OPTIONS]

Commands
--------

.. code-block:: bash

   bergson build [OPTIONS]

.. autoclass:: bergson.__main__.Build
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson build runs/my-index \
       --model EleutherAI/pythia-14m \
       --dataset NeelNanda/pile-10k \

.. code-block:: bash

   bergson query [OPTIONS]

.. autoclass:: bergson.__main__.Query
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson query \
       --index runs/my-index

.. code-block:: bash

   bergson score [OPTIONS]

.. autoclass:: bergson.__main__.Score
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: bash

   bergson score \
        runs/my-index-metadata \
        --query_path /runs/my-index \
        --scores_path /runs/scores \
        --dataset EleutherAI/SmolLM2-135M-10B


Configuration
^^^^^^^^^^^^^

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

Query Configuration:

.. autoclass:: bergson.data.QueryConfig
   :members:
   :undoc-members:

Score Configuration:

.. autoclass:: bergson.data.ScoreConfig
   :members:
   :undoc-members:
