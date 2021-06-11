.. _rules_guide:

=====
Rules
=====

``synthesized.common.rules`` allows the user to constrain the synthetic dataset, ensuring it confirms to pre-defined
business logic, or a custom scenario.

.. ipython:: python
    :verbatim:

    df_synth = synth.synthesize_from_rules(num_rows=1000, association_rules=..., generic_rules=..., expression_rules=...)

Associations
------------

Often, datasets may contain two columns with an important well-defined relationship. For example:

.. csv-table:: cars-original
   :header: "Make", "Model", "Total"
   :widths: 20, 20, 10

   "Ford", "Fiesta", 372013
   "BMW", "M3", 10342
   "BMW", "X5", 39753
   "Volkswagen", "Polo", 87421
   "Ferrari", "California", 632

In the above dataset, "Make" has a one-to-many association with "Model". In other words, certain categories in "Model"
only appear with certain categories in "Make". The ``HighDimSynthesizer`` captures highly detailed dataset-wide information,
but as it also attempts to generalize specific row-level information, a case such as "Polo" always appearing with
"Volkswagen" isn't strictly followed.  A possible output of the synthesizer could be:

.. csv-table:: cars-synthetic
   :header: "Make", "Model", "Total"
   :widths: 20, 20, 10

   "BMW", "X6", 36382
   "Ford", "Fiesta", 401877
   "BMW", "Polo", 67862

In this example, the ``HighDimSynthesizer`` has generated a row with a "BMW Polo", which is an unrealistic combination. If
capturing strict column associations such as this is important, the synthesizer can be configured to do so by defining an
``Association`` rule

.. ipython:: python
    :verbatim:

    rule = Association.detect_association(df, df_meta, associations=["car_manufacturer", "car_model"])

In addition, sometimes empty values are correlated, e.g: if one column specifies the number of children in a family we
would expect that the names of these children to be empty if they don't exist

.. ipython:: python
    :verbatim:

    rule = Association.detect_association(df, df_meta, associations=["NumberOfChildren"], nan_associations=["Child1Name", "Child2Name", ...])

The association class contains a class method ``detect_association`` that automatically detects these rules betweens the columns,
if some category of a column never appears with another then it can force the Synthesizer to never output those values together.
However, if a specific rule is required that isn't present in the data the Association can be intialised on its own.

.. ipython:: python
    :verbatim:

    rule = Association(binding_mask=binding_mask, associations=..., nan_association=...)

Here the ``binding mask`` specifies the possible outputs of the Synthesizer, this isn't currently user-friendly to construct due to its lack of use-case.

There are some constraints on what rules you can define, the Synthesizer only allows a column to appear in one association
and a column cannot appear in both the ``association`` and ``nan_association`` argument.
Some of these constraints may be possible to change in the future.


Expressions
-----------

When it is known apriori that a field in a dataset is related to others through a mathematical transformation, this can
be enforced with an ``Expression`` rule. This takes a string expression that can be parsed by `pandas.eval <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.eval.html>`__.

.. ipython:: python
    :verbatim:

    rule = Expression(name="total", expr="a+b+c")

Generic
-------

A ``GenericRule`` is a special type of rule that can be enforced by conditional sampling of ``HighDimSynthesizer``

.. warning::
    As these rules are enforced by iterative conditional sampling, it may not be possible to fully generate the desired
    number of rows if the rules cannot be fulfilled, or represent a very small proportion of the original data. In this
    case, ``HighDimSynthesizer.synthesize_from_rules`` will throw a ``RuntimeError``. Increasing the ``max_iter``
    parameter may avoid this issue.


ValueRange
^^^^^^^^^^

``ValueRange`` can be used to constrain synthesized data to a user-defined range, either to improve the quality of the synthetic data
or to generate custom scenarios. The upper and lower bounds of the range can be numeric, e.g '0 < x < 10.

.. ipython:: python
    :verbatim:

    rule = ValueRange(name="x", low=0, high=10)

or they can also be defined by another field of the dataset, e.g x < y

.. ipython:: python
    :verbatim:

    rule = ValueRange(name="x", high='y')


ValueEquals
^^^^^^^^^^^

``ValueEquals`` enforces the field of a dataset to be strictly equal to a specified value, either numeric or categorical.

.. ipython:: python
    :verbatim:

    rule = ValueEquals(name="x", value='A')

ValueIsIn
^^^^^^^^^

``ValueIsIn`` is similar to ``ValueEquals``, but specifies a list of allowed values.

.. ipython:: python
    :verbatim:

    rule = ValueEquals(name="x", values=['A', 'B'])

CaseWhenThen
^^^^^^^^^^^^

``CaseWhenThen`` can be used to impose a conditional structure between fields of a dataset. For example, the business
logic of `when age < 18 then income = 0` can be enforced with

.. ipython:: python
    :verbatim:

    rule = CaseWhenThen(when=ValueRange("age", high=18), then=ValueEquals("income", value=0))

The ``when`` and ``then`` parameters are specified by a ``GenericRule``.
