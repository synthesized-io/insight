.. _privacy_masks_guide:


=============
Privacy Masks
=============

Synthesized provides a variety of masks to anonymize parts of data for privacy purposes. The privacy masks
replace the most identifying fields within data record with an artificial pseudonym.

Synthesized enables data masking through the following transformers:
    * :class:`~synthesized.privacy.masking.NullTransformer`
    * :class:`~synthesized.privacy.masking.PartialTransformer`
    * :class:`~synthesized.privacy.masking.RandomTransformer`
    * :class:`~synthesized.privacy.masking.RoundingTransformer`
    * :class:`~synthesized.privacy.masking.SwappingTransformer`

NullTransformer
~~~~~~~~~~~~~~~

:class:`~synthesized.privacy.masking.NullTransformer` masks the data by nulling out a given column.

The following example illustrates it:

.. ipython:: python

    import pandas as pd
    from synthesized.privacy.masking import NullTransformer

    df = pd.DataFrame({'card_no': ['490 508 10L', 'ff4sff4', 'jdj DFj 34', '123POFjd33', '2334 fgg4 223', 'djdjjf 83838jd83', '123 453']})
    transformer = NullTransformer(name='card_no')
    transformer.fit(df)
    df_transformed = transformer.transform(df.copy())
    df_transformed.head()


PartialTransformer
~~~~~~~~~~~~~~~~~~

:class:`~synthesized.privacy.masking.PartialTransformer` performs data masking by masking out the first 75% (or N%) of each sample for the given column.
Arg ``masking_proportion`` determines what percentage of each sample will be masked.

The following example illustrates it:

.. ipython:: python

    from synthesized.privacy.masking import PartialTransformer

    df = pd.DataFrame({'account_num': ['49050810L', 'ff4sff4', 'jdjjdjDFj34', '123POFjd33', 'djB88ndjK93', '2234dr',
                      'DER44', '2334 fgg4 223', 'djdjjf 83838jd83', 'djjdjd093k']})

    transformer = PartialTransformer(name='account_num', masking_proportion=0.8)
    transformer.fit(df)
    df_transformed = transformer.transform(df.copy())
    df_transformed.head()


RandomTransformer
~~~~~~~~~~~~~~~~~

:class:`~synthesized.privacy.masking.RandomTransformer` masks a column by replacing the column values with a random string with slight format consistency.
Arg ``str_length`` determines what length of the random string will be generated.

.. note::
    Depending on whether the column values contain upper case character, lower case character and/or
    numeric character, the random values generated will or will not contain these.

.. ipython:: python

    from synthesized.privacy.masking import RandomTransformer

    df = pd.DataFrame({'Id': ['49050810L', 'D44J322K', 'FK53MDK3', '9FNF43MD', 'SJ42KDK4']})
    transformer = RandomTransformer(name='Id', str_length=7)
    transformer.fit(df)
    df_transformed = transformer.transform(df.copy())
    df_transformed.head()


Since the 'Id' column values have numeric character and upper case characters, hence, the transformed
column values will have numeric character and upper case characters.


RoundingTransformer
~~~~~~~~~~~~~~~~~~~

:class:`~synthesized.privacy.masking.RoundingTransformer` masks a numerical column by binning the values to N bins.
Arg ``n_bins`` determines the number of bins to bin the value range of the column, the default value is 20.

The following example illustrates it:

.. ipython:: python

    from synthesized.privacy.masking import RoundingTransformer

    df = pd.DataFrame({'age': np.random.randint(1, 97, size=(5000,))})
    transformer = RoundingTransformer(name='age', n_bins=10)
    transformer.fit(df)
    df_transformed = transformer.transform(df.copy())
    df_transformed.head()


SwappingTransformer
~~~~~~~~~~~~~~~~~~~

:class:`~synthesized.privacy.masking.SwappingTransformer` masks by shuffling the categories around in a given categorical column.
Boolean arg ``uniform`` determines if the categories should be distributed uniformly or if the
existing proportion of categories in the column should be maintained.

The following example shows it:

.. ipython:: python

    from synthesized.privacy.masking import SwappingTransformer

    df = pd.DataFrame({'wday': np.random.choice(['mon', 'tues', 'wed', 'thur', 'fri', 'sat', 'sun'],
                      size=100)})
    transformer = SwappingTransformer(name='wday', uniform=True) # for uniform=True, the weekdays will be distributed uniformly in the transformed column
    transformer.fit(df)
    df_transformed = transformer.transform(df.copy())
    df_transformed.head()


MaskingTransformerFactory
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~synthesized.privacy.masking.MaskingTransformerFactory` can be used to transform the same or multiple columns of a dataframe using the above
data masking transformers.

The following example illustrates it:

.. ipython:: python

    from faker import Faker
    from synthesized.privacy.masking import MaskingTransformerFactory

    fkr = Faker()
    df = pd.DataFrame({'Username': [fkr.user_name() for _ in range(1000)],
                        'Name': [fkr.name() for _ in range(1000)],
                        'Password': [fkr.password() for _ in range(1000)],
                        'CreditCardNo': [fkr.credit_card_number() for _ in range(1000)],
                        'Age': [fkr.pyint(min_value=10, max_value=78) for _ in range(1000)],
                        'MonthlyIncome': [fkr.pyint(min_value=1000, max_value=10000) for _ in range(1000)]})

    df.head()

Next, create a config dictionary with the key as the column name to which the transformation is to be applied
and the value is the name of the transformation to be applied. Arguments to the transformer can be provided using
'|' operator.

The config dictionary is passed in the call of method ``create_transformers`` of the ``MaskingTransformerFactory``
object. This method returns a ``DataFrameTransformer`` which can then be used to fit and transform the dataset.

.. ipython:: python

    config = dict(Age='rounding',
                  MonthlyIncome='rounding|3',
                  Username='partial_masking|0.25',
                  CreditCardNo='partial_masking',
                  Name='random',
                  Password='null')

    mt_factory = MaskingTransformerFactory()
    dfm_transformer = mt_factory.create_transformers(config)
    dfm_transformer.fit(df)
    masked_df = dfm_transformer.transform(df)
    masked_df.head()
