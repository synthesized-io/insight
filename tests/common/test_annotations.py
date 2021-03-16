import random
import string

import numpy as np
import pandas as pd
import pytest

from synthesized import HighDimSynthesizer, MetaExtractor
from synthesized.config import AddressLabels, BankLabels, PersonLabels
from synthesized.metadata_new.value import Address, Bank, FormattedString, Person


def test_synthesis_w_annotations():
    n = 1000
    df_original = pd.DataFrame({
        'x': np.random.normal(size=n),
        'y': np.random.choice(['a', 'b', 'c'], size=n),
        'sample': [''.join([random.choice(string.ascii_letters) for _ in range(10)]) for _ in range(n)],
        'sample2': [''.join([random.choice(string.ascii_letters.upper()) for _ in range(5)]) for _ in range(n)],
        'bic': [''.join([random.choice(string.ascii_letters) for _ in range(4)]) for _ in range(n)],
        'sort_code': [''.join([random.choice(string.ascii_letters) for _ in range(6)]) for _ in range(n)],
        'account': [''.join([random.choice(string.digits) for _ in range(6)]) for _ in range(n)],
        'bic2': [''.join([random.choice(string.ascii_letters) for _ in range(4)]) for _ in range(n)],
        'sort_code2': [''.join([random.choice(string.ascii_letters) for _ in range(6)]) for _ in range(n)],
        'account2': [''.join([random.choice(string.digits) for _ in range(6)]) for _ in range(n)],
        'gender': np.random.choice(['m', 'f', 'u'], size=n),
        'gender2': np.random.choice(['m', 'f'], size=n),
        'title': np.random.choice(['mr', 'mr.', 'mx', 'miss', 'Mrs'], size=n),
        'name': ['test_name'] * n,
        'email': ['test_name@email.com'] * n,
        'postcode': np.random.choice(['NW5 2JN', 'RG1 0GN', 'YO31 1MR', 'BD1 0WN'], size=n),
        'street': [''.join([random.choice(string.ascii_letters) for _ in range(10)]) for _ in range(n)],
        'building_number': [''.join([random.choice(string.digits) for _ in range(3)]) for _ in range(n)],
        'postcode2': np.random.choice(['NW5 2JN', 'RG1 0GN', 'YO31 1MR', 'BD1 0WN'], size=n),
        'street2': [''.join([random.choice(string.ascii_letters) for _ in range(10)]) for _ in range(n)],
    })
    annotations = [
        FormattedString(name='sample', pattern='[A-Za-z]{10}'),
        FormattedString(name='sample2', pattern='[A-Za-z]{5}'),
        Bank(name='bank', labels=BankLabels(bic_label='bic', sort_code_label='sort_code', account_label='account')),
        Bank(name='bank2', labels=BankLabels(bic_label='bic2', sort_code_label='sort_code2', account_label='account2')),
        Person(name='person', labels=PersonLabels(gender_label='gender', title_label='title',
                                                  firstname_label='name', email_label='email')),
        Person(name='person2', labels=PersonLabels(gender_label='gender2')),
        Address(name='address', labels=AddressLabels(postcode_label='postcode', street_label='street',
                                                     house_number_label='building_number')),
        Address(name='address2', labels=AddressLabels(postcode_label='postcode2', street_label='street2')),
    ]

    df_meta = MetaExtractor.extract(df=df_original, annotations=annotations)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)
    df_synthesized = synthesizer.synthesize(n)
    assert df_synthesized.shape == df_original.shape


@pytest.mark.slow
def test_annotations_all():
    data = pd.read_csv('data/annotations_nd.csv')

    person = Person(name='person', labels=PersonLabels(
        gender_label=None,
        title_label='Title (Tab selection)',
        firstname_label='First Name',
        lastname_label='Last Name',
        email_label='Email address',
        mobile_number_label='Mobile No.',
        username_label='username',
        password_label='password'
    ))

    bank = Bank(name='bank', labels=BankLabels(
        sort_code_label='Sort code',
        account_label='Bank Account Number'
    ))

    address = Address(name='address', labels=AddressLabels(
        postcode_label='POSTCODE',
        city_label='POSTTOWN',
        district_label='DISTRICT',
        street_label='STREET',
        flat_label='FLAT',
        house_name_label='HOUSENAME',
        full_address_label='Full Address',
    ))

    pa_address = Address(
        name='pa_address',
        labels=AddressLabels(
            postcode_label='PA_POSTCODE',
            county_label='PA_COUNTY',
            city_label='PA_POSTTOWN',
            street_label='PA_STREET',
            flat_label='PA_FLAT',
            house_name_label='PA_HOUSENAME',
    ))

    county = FormattedString(
        name='COUNTY',
        pattern='[A-Za-z]{5,10}'
    )

    pa_district = FormattedString(
        name='PA_DISTRICT',
        pattern='[A-Z][a-z]{8,15}'
    )

    annotations = [person, bank, address, county, pa_district]

    df_meta = MetaExtractor.extract(
        df=data,
        annotations=annotations
    )

    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(df_train=data, num_iterations=10)
    df_synthesized = synthesizer.synthesize(num_rows=len(data))

    # TODO: fix encode for annotations (ML-254)
    # _ = synthesizer.encode(data)
    # _ = synthesizer.encode_deterministic(data)

    assert df_synthesized.shape == data.shape

    # TODO: fix import export for annotations (ML-255)
    # Ensure that import-export works
    # f = BytesIO()
    # synthesizer.export_model(f)

    # f.seek(0)
    # synthesizer2 = HighDimSynthesizer.import_model(f)
    # df_synthesized2 = synthesizer.synthesize(num_rows=len(data))

    # assert df_synthesized2.shape == data.shapep
