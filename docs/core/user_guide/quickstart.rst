.. _quickstart:

=============
Quickstart
=============

Synthesized can be used to quickly generate realistic high dimensional tabular data. This is achieved by learning a high
dimensional, intelligent model of the original data that preserves the statistical properties, distributions and
correlations between fields.

.. ipython:: python
    :verbatim:

    import pandas as pd
    import synthesized

.. note::
    Synthesized interfaces with the `pandas <https://pandas.pydata.org/>`_ package, as such the data to be synthesized must
    be first loaded into a DataFrame object.

We load some example data from the package

.. ipython:: python
    :verbatim:

    df = synthesized.util.get_example_data()
    df.head()

.. csv-table::
    :header: ,SeriousDlqin2yrs,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio
    :widths: 10, 10, 10, 10, 10, 10

    1,1,0.7661266090000001,45,2,0.8029821290000001
    2,0,0.957151019,40,0,0.121876201
    3,0,0.65818014,38,1,0.085113375
    4,0,0.233809776,30,\-,0.036049682
    5,0,0.9072394,49,1,0.024925695


Extract the ``DataFrameMeta``. This is necessary to learn the schema, which in turn defines the
necessary transformations required to understand the original data and generate new data points.

.. ipython:: python
    :verbatim:

    df_meta = synthesized.MetaExtractor.extract(df)

Initialize the ``HighDimSynthesizer`` object, which will learn an intelligent model of the underlying data.

.. ipython:: python
    :verbatim:

    synth = synthesized.HighDimSynthesizer(df_meta)

Train the model.

.. ipython:: python
    :verbatim:

    synth.learn(df, num_iterations=None)

Synthesize an arbitrary amount of data:

.. ipython:: python
    :verbatim:

    df_synth = synth.synthesize(num_rows=1000)
    df_synth.head()

.. csv-table::
    :header: ,SeriousDlqin2yrs,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio
    :widths: 10, 10, 10, 10, 10, 10

    0,0,0.6002727150917053,79,0,0.28565606474876404
    1,0,0.4615554213523865,56,3,0.24112118780612946
    2,0,0.36208802461624146,58,0,354.8174743652344
    3,1,0.13040462136268616,36,2,0.08531860262155533
    4,0,0.38728469610214233,45,2,0.5294051766395569


Be default, Synthesized imputes any missing values. To generate ``NaN`` values,
use ``produce_nans=True``.

.. ipython:: python
    :verbatim:

    df_synth = synth.synthesize(num_rows=1000, produce_nans=True)

.. csv-table::
    :header: ,SeriousDlqin2yrs,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio
    :widths: 10, 10, 10, 10, 10, 10

    0,0,0.17057423293590546,28,0,0.4905789792537689
    1,0,0.3659568130970001,62,\-,1.0660463571548462
    2,0,0.6086112260818481,44,1,0.043783850967884064
    3,0,0.4614080488681793,72,0,0.06663402169942856
    4,0,0.2034926861524582,37,\-,0.3162900507450104
