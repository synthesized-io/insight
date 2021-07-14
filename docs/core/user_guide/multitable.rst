.. _multitable_guide:

===================
Two Table Synthesis
===================

Synthesized supports generating synthetic data from two tables that are related with a primary and foreign key
constraint. With the :class:`~synthesized.complex.TwoTableSynthesizer` class, it is possible to:

1. Keep the referential integrity between two tables, i.e the primary and foreign keys can be joined.
2. Maintain the distribution of foreign key counts in the case of a one-to-many relationship.

It is assumed that each row in the first table is a unique entity, such as a customer, with no duplicates. The
second table must relate to the first through a foreign key that matches the primary key, e.g the transactions of
each customer.

.. important::
    ``TwoTableSynthesizer`` will maintain the correlations between fields *within* each table, but not *across* the two
    tables.

:class:`~synthesized.complex.TwoTableSynthesizer` requires two tables that have a single column which acts as a unique identifier for that row,
i.e a primary key. In addition, the second table must have a single foreign key that references the primary key of the
first table. Currently, :class:`~synthesized.complex.TwoTableSynthesizer` only supports integer type primary and foreign keys.

.. ipython:: python

    import pandas as pd
    from synthesized import MetaExtractor
    from synthesized.complex import TwoTableSynthesizer

Load the two related tables. In this case, ``customer_table`` contains a primary key column named ``customer_id``, and
``transaction_table`` contains a primary key column named ``transaction_id``, together with the foreign key
``customer_id``. This references the ``customer_id`` in ``customer_tab``.

.. ipython:: python
    :verbatim:

    df_customer = pd.read_csv('customer_table.csv')
    df_transactions = pd.read_csv('transaction_table.csv')

Extract the :class:`~synthesized.DataFrameMeta` for each table:

.. ipython:: python
    :verbatim:

    df_metas = (MetaExtractor.extract(df_cust), MetaExtractor.extract(df_tran))

Initialise the :class:`~synthesized.complex.TwoTableSynthesizer`. The column names of the primary keys of each table must be specified using the
``keys`` parameter. If the foreign key column name does not match the primary key name, this can be specified with the
``relation`` parameter, e.g by passing a dictionary of the mapping `{'primary_key_name': 'foreign_key_name'}`

.. ipython:: python
    :verbatim:

    synthesizer = TwoTableSynthesizer(df_metas=df_metas, keys=('customer_id', 'transaction_id'))

Train the Synthesizer:

.. ipython:: python
    :verbatim:

    synthesizer.learn(df_train=dfs)

.. note::
    ``fit(...)`` method is an alias of ``learn(...)`` method.

Hence, the training can also be done as follows:

.. ipython:: python
    :verbatim:
    
    synthesizer.fit(df_train=dfs)

Generate 1,000 rows of new data. This will return two tables: the first contains the 1,000 synthetic customers, and
the second contains the synthetic transacations for each customer. Note: the size of the second table cannot be fixed,
as the number of generated rows is sampled from the learned distribution.

.. ipython:: python
    :verbatim:

    df_customer_synthetic, df_transaction_synthetic = synthesizer.synthesize(num_rows=1000)

.. note::
    ``sample(...)`` method is an alias of ``synthesize(...)`` method.

Hence, the data synthesis can also be done as follows:

.. ipython:: python
    :verbatim:
    
    df_customer_synthetic, df_transaction_synthetic = synthesizer.sample(num_rows=1000)

Verify that the referential integrity is maintained by joining the two tables:

.. ipython:: python
    :verbatim:

    df_customer_synthetic.merge(df_transaction_synthetic, on='customer_id', how='left')
