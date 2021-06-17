.. _singletable_guide:

=======================
Single Table Synthesis
=======================

The Synthesized SDK contains the ``synthesized.HighDimSynthesizer`` object which is designed to synthesize data from
single tables easily and accurately, see the :ref:`quickstart<quickstart>` for an introduction. On this page we go through
that same process in more detail.

The ``HighDimSynthesizer`` uses advanced generative modelling techniques to produce synthetic data that closely resembles
the original. Whilst the ``HighDimSynthesizer`` works across a variety of use cases, there are some restrictions that
are worth bearing in mind to get the most out of the Synthesizer.

.. important::
    Synthesized assumes that there is no temporal or conditional dependencies the between rows; each row in the table is
    assumed to be independent and identically distributed.

This means that time-series data or data where rows are dependent between each other will not be correctly synthesized,
those dependancies can be lost.

We load the dataset in, compute the ``DataFrameMeta`` and instantiate the ``HighDimSynthesizer``

.. ipython:: python
    :verbatim:

    import pandas as pd
    import synthesized

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

.. ipython:: python
    :verbatim:

    df_meta = synthesized.MetaExtractor.extract(df, annotations=...)
    synthesizer = synthesized.HighDimSynthesizer(df_meta, type_overrides=...)

This builds a blank generative model of the data ready for the learning process.

.. note::
    At this stage both :ref:`annotations<annotation_guide>` and type overrides can be passed to the ``MetaExtractor`` and
    ``HighDimSynthesizer`` respectively. They will alter how the Synthesizer treats certain columns to improve synthesis
    to cater to different use cases.


Training
------------

Now the data can be learnt

.. ipython:: python
    :verbatim:

    synthesizer.learn(df, num_iterations=None)

Depending on the size of the dataset this process could take a few minutes to complete, here the ``HighDimSynthesizer``
will learn patterns present in the data so that it can generate them later. When we set `num_iterations=None` this let's
the HighDimSynthesizer model choose when to end training.

.. note::
    Whilst the ``HighDimSynthesizer`` can use a GPU to improve training time, we mostly encourage CPU training for now.
    As the dataset is loaded into memory as a pandas dataframe, read-write speed should not be a limiting factor for
    training time. Instead, the memory of the system might need to be tracked to ensure it is not used up and the
    operating system starts to swap.

The ``num_iterations`` argument can be set to a specific value in order to constrain the number of
learning steps of the Synthesizer. This can be particularly useful for testing any pipelines containing the
``HighDimSynthesizer`` before trying to Synthesize data properly.

However, if a large value is provided to ``num_iterations`` the Synthesizer may decide to end training
early regardless, so increasing training time is not possible in this way. It is possible to force the Synthesizer to
train for longer by calling ``.learn`` additional times. The Synthesizer has been designed to learn the dataset in a
single call so this should not be necessary in most cases.



Synthesis
------------

Finally, the Synthesizer can be used to generate data

.. ipython:: python
    :verbatim:

    df_synth = synthesizer.synthesize(num_rows=1000)
    df_synth.head()

.. csv-table::
    :header: ,SeriousDlqin2yrs,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio
    :widths: 10, 10, 10, 10, 10, 10

    0,0,0.6002727150917053,79,0,0.28565606474876404
    1,0,0.4615554213523865,56,3,0.24112118780612946
    2,0,0.36208802461624146,58,0,354.8174743652344
    3,1,0.13040462136268616,36,2,0.08531860262155533
    4,0,0.38728469610214233,45,2,0.5294051766395569



this will generate a dataframe with the required number of rows. This process should be very quick in comparison to
training time. Optionally, the Synthesizer can be forced to generate missing values in a pattern that is common with
the input dataset.

.. ipython:: python
    :verbatim:

    df_synth = synthesizer.synthesize(num_rows=1000, produce_nans=True)
    df_synth.head()

.. csv-table::
    :header: ,SeriousDlqin2yrs,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio
    :widths: 10, 10, 10, 10, 10, 10

    0,0,0.17057423293590546,28,0,0.4905789792537689
    1,0,0.3659568130970001,62,\-,1.0660463571548462
    2,0,0.6086112260818481,44,1,0.043783850967884064
    3,0,0.4614080488681793,72,0,0.06663402169942856
    4,0,0.2034926861524582,37,\-,0.3162900507450104

In this dataset the ``HighDimSynthesizer`` doesn't recognise that the structure of the email addresses can be deduced
from the name of the person. To see how to configure the synthesizer to do this, read the
:ref:`annotation guide<annotation_guide>`. Additional rules or constraints on the data can also be specified with the
``.synthesize_from_rules`` method as detailed in the :ref:`rules guide<rules_guide>`


Saving and Loading Models
--------------------------

To save models use the ``synthesizer.export_model`` method to save as a binary file.

.. ipython:: python
    :verbatim:

    with open("example.synth", "wb") as out_f:
        synthesizer.export_model(fh)

to import this model into a new HighDimSynthesizer instance, use the static method
``HighDimSynthesizer.import_model``

.. ipython:: python
    :verbatim:

    with open("example.synth", "rb") as in_f:
        synthesizer2 = synthesized.HighDimSynthesizer.import_model(in_f)

    synthesizer2.synthesize(num_rows=5)

.. csv-table::
    :header: ,SeriousDlqin2yrs,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio
    :widths: 10, 10, 10, 10, 10, 10

    0,0,0.18753696978092194,53,1,0.29868805408477783
    1,1,0.2405071258544922,49,3,0.24129432439804077
    2,0,0.15856477618217468,56,0,0.5956577658653259
    3,1,0.5415436625480652,37,3,0.8815135359764099
    4,1,0.18602889776229858,52,2,0.429883420467376
