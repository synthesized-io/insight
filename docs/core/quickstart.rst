Getting started guide
=====================

Metadata inference
------------------

.. code-block:: python

    >>> import synthesized
    >>> df = pd.read_csv("data/credit.csv", index_col=0)
    >>> df_meta = synthesized.MetaExtractor.extract(df)
    >>> df_meta.children
    [<Ring[i8]: IntegerBool(name=SeriousDlqin2yrs)>,
    <Ring[f8]: Float(name=RevolvingUtilizationOfUnsecuredLines)>,
    <Scale[i8]: Integer(name=age)>,
    <Scale[i8]: Integer(name=NumberOfTime30-59DaysPastDueNotWorse)>,
    <Ring[f8]: Float(name=DebtRatio)>,
    <Scale[i8]: Integer(name=MonthlyIncome)>,
    <Scale[i8]: Integer(name=NumberOfOpenCreditLinesAndLoans)>,
    <Scale[i8]: Integer(name=NumberOfTimes90DaysLate)>,
    <Scale[i8]: Integer(name=NumberRealEstateLoansOrLines)>,
    <Scale[i8]: Integer(name=NumberOfTime60-89DaysPastDueNotWorse)>,
    <Scale[i8]: Integer(name=NumberOfDependents)>]


Tabular synthesis
-----------------

.. code-block:: python

    >>> synth = synthesizer.HighDimSynthesizer(df_meta)

.. code-block:: python

    >>> synth.learn(df, num_iterations=None)

.. code-block:: python

    >>> df_synth = synth.synthesize(num_rows=1000)

Annotations
-----------

Person
^^^^^^

Address
^^^^^^^

Bank
^^^^

Rules
-----

:ref:`synthesized.common.rules` allows the user to constrain the synthetic dataset, ensuring it confirms to pre-defined
business logic, or a custom scenario.

.. code-block:: python

    >>> df_synth = synth.synthesize_from_rules(num_rows=1000, association_rules=..., generic_rules=..., expression_rules=...)

Associations
^^^^^^^^^^^^
Associations enforce strict categorical relationships in the generated data. For example, given a dataset with fields
`car_manufacturer` and `car_model`, there is a strict association between each model and manufacturer (i.e no model appears
in more than one manufactuter). This relationship may not be learned perfectly by the synthesizer (after all, we can
only learn a probabilistic approximation), and in this case the strict constrain can be enforced by defining an
``Association`` rule

.. code-block:: python

    >>> rule = Association.detect_association(df, df_meta, associations=["car_manufacturer", "car_model"])

In addition, sometimes empty values are correlated, e.g: if one column specifies the number of children in a family we
would expect that the names of these children to be empty if they don't exist

.. code-block:: python

    >>> rule = Association.detect_association(df, df_meta, associations=["NumberOfChildren"], nan_associations=["Child1Name", "Child2Name", ...])

The association class contains a class method ``detect_association`` that automatically detects these rules betweens the columns,
if some category of a column never appears with another then it can force the Synthesizer to never output those values together.
However, if a specific rule is required that isn't present in the data the Association can be intialised on its own.

.. code-block:: python

    >> rule = Association(binding_mask=binding_mask, associations=..., nan_association=...)

Here the binding mask specifies the possible outputs of the Synthesizer, this isn't currently user-friendly to construct due to its lack of use-case.

There are some constraints on what rules you can define, the Synthesizer only allows a column to appear in one association
and a column cannot appear in both the ``association`` and ``nan_association`` argument.
Some of these constraints may be possible to change in the future.


Expressions
^^^^^^^^^^^
When it is known apriori that a field in a dataset is related to others through a mathematical transformation, this can
be enforced with an ``Expression`` rule. This takes a string expression that can be parsed by `pandas.eval <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.eval.html>`__.

.. code-block:: python

    >>> rule = Expression(name="total", expr="a+b+c")

Generic
^^^^^^^
A ``GenericRule`` is a special type of rule that can be enforced by conditional sampling of ``HighDimSynthesizer``

ValueRange
**********
``ValueRange`` can be used to constrain synthesized data to a user-defined range, either to improve the quality of the synthetic data
or to generate custom scenarios. The upper and lower bounds of the range can be numeric, e.g '0 < x < 10.

.. code-block:: python

    >>> rule = ValueRange(name="x", low=0, high=10)

or they can also be defined by another field of the dataset, e.g x < y

.. code-block:: python

    >>> rule = ValueRange(name="x", high='y')


ValueEquals
***********
``ValueEquals`` enforces the field of a dataset to be strictly equal to a specified value, either numeric or categorical.

.. code-block:: python

    >>> rule = ValueEquals(name="x", value='A')

ValueIsIn
*********
``ValueIsIn`` is similar to ``ValueEquals``, but specifies a list of allowed values.

.. code-block:: python

    >>> rule = ValueEquals(name="x", values=['A', 'B'])

CaseWhenThen
************
``CaseWhenThen`` can be used to impose a conditional structure between fields of a dataset. For example, the business
logic of `when age < 18 then income = 0` can be enforced with

.. code-block:: python

    >>> rule = CaseWhenThen(when=ValueRange("age", high=18), then=ValueEquals("income", value=0))

The ``when`` and ``then`` parameters are specified by a ``GenericRule``.
