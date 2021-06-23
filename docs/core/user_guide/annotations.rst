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
