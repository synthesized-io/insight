.. _annotation_guide:


=================
Entity Annotation
=================

.. warning::
    Configuring an entity annotation is necessary to generate realistic fake PII such as customer names and addresses.
    Synthesized does not currently automatically recognise fields that contain PII, and therefore the default
    behaviour will be to generate the original data from such fields.

Tabular datasets often contain fields that when combined can describe a specific entity, such as a unique person or
postal address. For example, consider the dataset below that contains customer PII

.. csv-table:: customer-purchases
   :header: "title", "first_name", "last_name", "gender", "email", "amount"
   :widths: 20, 20, 20, 20, 20, 10

   "Mr", "John", "Doe", "Male", "john.doe@gmail.com", 101.2
   "Mrs", "Jane", "Smith", "Female", "jane.smith@gmail.com", 28.2
   "Dr", "Albert", "Taylor", "Male", "albert.taylor@aol.com", 98.1
   "Ms", "Alice", "Smart", "Female", "alice.smart@hotmail.com", 150.3

The combination of ('title', 'first_name', 'last_name', 'gender', 'email') describes a unique person in this data, and
there is a strict relationship between these attributes. E.g When "title" is "Mrs" or "Ms" then "first_name" will most
likely contain a name given to females.

When it is important to maintain this correct description of an entity in the generated synthetic data, the dataset
must be *annotated* to link the appropriate fields together. When annotated, synthesized will learn to generate
realsistic entities as a whole, rather than independently generating individual attributes.

Currently, synthesized can handle person, address, bank and generic formatted string entities.

Person
------
Generating synthetic PII for individuals in a dataset can be achieved by defining a :class:`~synthesized.metadata.value.Person` annotation.

.. ipython:: python

   from synthesized.metadata.value import Person
   from synthesized.config import PersonLabels

The columns of a dataset that relate to the attributes of a person are specifed using :class:`~synthesized.config.PersonLabels`. This is used
to define the :class:`~synthesized.metadata.value.Person` values that synthesized can then generate.

.. ipython:: python

   person = Person(
        name='person',
        labels=PersonLabels(
            gender_label='gender',
            title_label='title',
            firstname_label='first_name',
            lastname_label='last_name',
            email_label='email'
        )
    )


.. ipython:: python
   :verbatim:

   df_meta = MetaExtractor.extract(df=data, annotations=[person])


.. note::

   It is possible to define multiple Person annotations if a dataset contains PII columns for more than one person.
   These must be created as separate Person objects with unique names, and then passed to the list of annotations,
   e.g ``MetaExtractor.extract(df=..., annotations=[person_1, person_2])``


.. ipython:: python
   :verbatim:

   synthesizer = HighDimSynthesizer(df_meta=df_meta)
   synthesizer.learn(...)
   df_synthesized = synthesizer.synthesize(num_rows=...)
 
PersonModel
^^^^^^^^^^^
``PersonModel`` encapsulates the attributes of a person. When paired with a :class:`~synthesized.metadata.value.Person` Meta, 
they are able to understand and learn about the attributes that define a person and then generate data from their learned understanding. It captures gender using `Gender` model internally. 
It can be used to create the following attributes:

- ``gender`` `(orig.)`
- ``title`` `(orig.)`
- ``first_name``
- ``last_name``
- ``email``
- ``username``
- ``password``
- ``home/work/mobile_number``

Attributes marked with `'orig.'` have values that correspond to the original dataset. The rest are intelligently
generated based on the hidden model for the hidden attribute, `_gender` = {"F", "M", "NB", "A"}.

There are 3 special configuration cases for this model that should be considered:
    1. The attribute ``gender`` is present: In this case, the hidden model for `_gender` is based directly on the ``gender`` attribute. All values in the ``gender`` attribute should correspond to "F", "M", "U" or <NA>. In other words, there should be no ambiguous values in the collection "A".
    2. No ``gender`` present but ``title`` is present: The hidden model for `_gender` can be based on the available titles. As this is not a direct correspondence, not all values will correspond to a single collection. In other words, there MAY be some ambiguous values in the collection "A".
    3. Neither ``gender`` nor ``title`` is present: The hidden model for gender cannot be fitted to the data and so the `_gender` attribute is assumed to be evenly distributed amongst the genders specified in the config.

.. note::
    ``PersonModel`` can be provided `PersonModelConfig` during initialization. 'person_locale' is a member variable
    of the `PersonModelConfig` class which can be set to specify the locality of the people.
    
    | E.g. person_locale = 'ru_RU' will refer to people belonging to Russia
    | This can be quite useful to synthesize details of people belonging to a particular locality.

.. ipython:: python

    import pandas as pd
    from synthesized.metadata.factory import MetaExtractor
    from synthesized.config import PersonModelConfig, PersonLabels
    from synthesized.metadata.value import Person
    from synthesized.model.models import PersonModel

    meta = Person('person', labels=PersonLabels(title_label='title', gender_label='gender', name_label='name',
                                   firstname_label='firstname', lastname_label='lastname'))
    person_model_config = PersonModelConfig()
    person_model_config.person_locale='zh_CN'
    model = PersonModel(meta=meta, config=person_model_config)
    df = pd.DataFrame({'gender': np.random.choice(['m', 'f', 'u'], size=100), 'title': np.random.choice(['mr', 'mr.', 'mx', 'miss', 'Mrs'], size=100)})
    df[[c for c in model.params.values() if c not in df.columns]] = 'test'

    model.meta.revert_df_from_children(df)
    model.fit(df)
    model.sample(3)

Address
-------

Similarly, an :class:`~synthesized.metadata.value.Address` annotation allows Synthesized to generate fake address details. Currently, only
UK addresses can be generated.

.. ipython:: python

    from synthesized.metadata.value import Address
    from synthesized.config import AddressLabels

The columns of a dataset that relate to the attributes of an address are specifed using :class:`~synthesized.config.AddressLabels`.

.. ipython:: python

    address = Address(
         name='address',
         labels=AddressLabels(
             postcode_label='postcode',
             county_label='county',
             city_label='city',
             district_label='district',
             street_label='street_name',
             house_number_label='house_number'
         )
     )

.. ipython:: python
    :verbatim:

    df_meta = MetaExtractor.extract(df=data, annotations=[address])

AddressModel
^^^^^^^^^^^^
``AddressModel`` models addresses. It uses :class:`~synthesized.metadata.value.Address` meta, which represents 
columns with different address labels such as city, house_number, postcode, full_address, etc., to captures all the information 
needed to recreate similar synthetic data.

`AddressModelConfig` can also be provided as a part of the initialization. `AddressModelConfig` contains information
such as if an address file is provided or if the postcodes need to be learned for address synthesis.

.. note::
    ``AddressModel`` uses ``PostcodeModel`` to learn and synthesize the addresses. If the address file is provided then the 
    addresses corresponding to the learned postcodes are sampled from that. In case the address file is not provided
    then the `Faker <https://faker.readthedocs.io/en/master/>`_ is used to generate addresses.

.. tip::
   ``AddressModel`` class has a member variable 'postcode_level' which provides the flexibility to use partial or the full
   postcode for fitting and sampling.

    | E.g. for postcode "EC2A 2DP":
    | postcode_level=0 will signify "EC"
    | postcode_level=1 will signify "EC2A"
    | postcode_level=2 will signify "EC2A 2DP"

Without address file
####################
.. ipython:: python

    from synthesized.metadata.value import Address
    from synthesized.config import AddressLabels, AddressModelConfig
    from synthesized.model.models import AddressModel 

    config = AddressModelConfig(addresses_file=None, learn_postcodes=False)
    df = pd.DataFrame({
        'postcode': ["" for _ in range(10)],
        'street': ["" for _ in range(10)],
        'full_address': ["" for _ in range(10)],
        'city': ["" for _ in range(10)]
    })

    annotations = [Address(name='Address', nan_freq=0.3,
                   labels=AddressLabels(postcode_label='postcode', city_label='city',
                                        street_label='street', full_address_label='full_address'))]
    meta = MetaExtractor.extract(df, annotations=annotations)
    model = AddressModel(meta['Address'], config=config)
    model.meta.revert_df_from_children(df)
    model.fit(df)
    model.sample(n=3)

With address file
#################
.. ipython:: python

    from faker import Faker

    address_file_path = 'data/addresses.jsonl.gz'
    config = AddressModelConfig(addresses_file=address_file_path, learn_postcodes=True)
    fkr = Faker('en_GB')
    df = pd.DataFrame({
        'postcode': [fkr.postcode() for _ in range(10)],
        'street': [fkr.street_name() for _ in range(10)],
        'full_address': [fkr.address() for _ in range(10)],
        'city': [fkr.city() for _ in range(10)]
    })

    annotations = [Address(name='Address', nan_freq=0.3,
                   labels=AddressLabels(postcode_label='postcode', city_label='city',
                                        street_label='street', full_address_label='full_address'))]
    meta = MetaExtractor.extract(df, annotations=annotations)
    model = AddressModel(meta['Address'], config=config)
    model.meta.revert_df_from_children(df)
    model.fit(df)
    model.sample(n=3)

Bank
----

Defining a :class:`~synthesized.metadata.value.Bank` annotation allows Synthesized to generate fake bank account numbers and sort codes. Currently,
Synthesized can only generate 8-digit account numbers and 6-digit sort codes.

.. ipython:: python

    from synthesized.metadata.value import Bank
    from synthesized.config import BankLabels

The columns of a dataset that relate to the bank account attributes are specifed using :class:`~synthesized.config.BankLabels`.

.. ipython:: python

    bank = Bank(
         name='bank',
         labels=BankLabels(
            sort_code_label='sort_code',
            account_label='account_number'
         )
     )


.. _formattedstrings_guide:

FormattedString
----------------

A :class:`~synthesized.metadata.value.FormattedString` annotation can be used to generate synthetic data that conforms to a given regular expression,
e.g social security numbers, or customer account numbers that have a specific format.

.. ipython:: python

    from synthesized.metadata.value.categorical import FormattedString

The :class:`~synthesized.metadata.value.FormattedString` is defined by passing the respective column name, and a regex pattern

.. ipython:: python

    regex = "^(?!666|000|9\\d{2})\\d{3}-(?!00)\\d{2}-(?!0{4})\\d{4}$";
    social_security = FormattedString(
                        name="social_security_number",
                        pattern=regex)

.. ipython:: python
    :verbatim:

    df_meta = MetaExtractor.extract(df=data, annotations=[social_security])

FormattedStringModel
^^^^^^^^^^^^^^^^^^^^
``FormattedStringModel`` models a column with a specific regex pattern using :class:`~synthesized.metadata.value.FormattedString` annotation.

.. ipython:: python

    import pandas as pd
    from synthesized.metadata.factory import MetaExtractor
    from synthesized.metadata.value import FormattedString
    from synthesized.model.models import FormattedStringModel

    df = pd.DataFrame({'fstr_col': ['SJ-3921', 'LE-0826', 'PQ-0871']})
    df_meta = MetaExtractor.extract(df=df, annotations=[FormattedString(name='fstr_col',
                                                                        pattern='[A-Z]{2}-[0-9]{4}')])
    formatted_str_model = FormattedStringModel(df_meta['fstr_col'])
    formatted_str_model.fit(df)
    formatted_str_model.sample(3)