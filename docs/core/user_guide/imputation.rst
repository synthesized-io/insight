.. _imputation_guide:

===============
Data Imputation
===============

Given a dataset, and a ``HighDimSynthesizer`` trained on that dataset, ``synthesized.DataImputer``
is able to generate data for certain values. This is specially usefull when the dataset contains missing values or
outliers, although it can be used to impute any value.

For example, given a dataset with the following structure:

.. csv-table:: credit.csv
   :header: "Age", "MonthlyIncome", "Delinquent"
   :widths: 10, 10, 10

   23, 1500, ``True``
   58, 3800, *NaN*
   *NaN*, 2600, ``False``
   47, *NaN*, ``True``
   72, 3600, ``False``

A ``DataImputer`` object can get the information such as marginal and joint probability distributions provided by
the ``HighDimSynthesizer`` and fill missing values with realistic new samples:

.. csv-table:: credit-imputed.csv
   :header: "Age", "MonthlyIncome", "Delinquent"
   :widths: 10, 10, 10

   23, 1500, ``True``
   58, 3800, ``False``
   36, 2600, ``False``
   47, 3100, ``True``
   72, 3600, ``False``

.. important::
   The output dataframe will still contain original data for non-missing values, the ``DataImputer`` will only generate
   *Synthesized* data for missing values.

Imputing Missing Values
^^^^^^^^^^^^^^^^^^^^^^^

.. ipython:: python

    from synthesized import DataImputer

Data Imputation is achieved using the ``DataImputer`` class. This requires a ``HighDimSynthesizer`` instance
that has already been learned on the desired dataset.


.. ipython:: python
    :verbatim:

    data_imputer = DataImputer(synthesizer) # synthesizer is an HighDimSynthesizer instance

Once the ``DataImputer`` has been initialized, the user can impute missing values to a given ``df: pd.DataFrame``
with the following command:

.. ipython:: python
    :verbatim:

    df_nans_imputed = data_imputer.impute_nans(df, inplace=False)

The ``DataImputer`` will find all ``NaN``s in ``df``, fill them with new values, and return new data frame
``df_nans_imputed`` without missing values.

.. note::
    With the ``inplace`` argument, the user can control whether the given data-frame is modified or a copy of it is
    created, modified, and returned. After running ``data_imputer.impute_nans(df, inplace=True)``, ``df`` will not contain
    missing values.

    It is recommended to use ``inplace=True`` for big datasets in order to optimize memory usage.

Imputing Outliers
^^^^^^^^^^^^^^^^^

Outliers in data can heavily decrease model performance if not treaded carefuly, as many loss functions (e.g. MSE) are
highly impacted by heavy tailed distributions. For these situations, the ``DataImputer`` can reduce the number of
outliers by automatically detecting and imputing them with the following command:

.. ipython:: python
    :verbatim:

    df_outliers_imputed = data_imputer.impute_outliers(df, outliers_percentile=0.05, inplace=False)

The output dataframe ``df_outliers_imputed`` will have the top 2.5% and bottom 2.5% values for each continuous column
filled by the corresponding values as learned from the ``HighDimSynthesizer``.

.. note::
    For each column, the ``DataImputer`` will use a percentile-based approach to detect outliers. If some other
    approach is needed, it is recommended to create a boolean mask and use ``impute_mask()`` as described below.

Imputing a Mask
^^^^^^^^^^^^^^^

If the user needs to replace any other value (e.g. wrong values, anomalies...), he can do so by providing
a boolean mask dataframe, with the same size and columns as the original dataframe, where all ``True`` values will be
computed from the ``HighDimSynthesizer`` and ``False`` values will be returned as they are.

.. ipython:: python
    :verbatim:

    df_imputed = data_imputer.impute_mask(df, mask=df_mask, inplace=False)

For example given the *credit-anomaly.csv* below,

.. csv-table:: credit-anomaly.csv
   :header: "Age", "MonthlyIncome", "Delinquent"
   :widths: 10, 10, 10

   23, 1500, ``True``
   58, 921817402182, ``False``
   36, 2600, ``False``
   9816, 3600, ``True``

the user can to impute values for detected anomalies (``MonthlyIncome=921817402182`` and ``age=9816``)
by creating the following mask and passing it to the data imputer:

.. ipython:: python
    :verbatim:

    df = pd.read_csv("credit-anomaly.csv")
    df_mask = pd.DataFrame({
        "Age": [False, False, False, True],
        "MonthlyIncome": [False, True, False, False],
        "Delinquent": [False, False, False, False]
    })
    df_imputed = data_imputer.impute_mask(df, mask=df_mask, inplace=False)
