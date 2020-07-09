import pandas as pd

from synthesized import HighDimSynthesizer, MetaExtractor
from synthesized.config import AddressParams, BankParams, PersonParams, AddressMetaConfig


def test_annotations():
    data = pd.read_csv('data/annotations_nd.csv')
    person_params = PersonParams(
        gender_label='Title (Tab selection)',
        title_label='Title (Tab selection)',
        firstname_label='First Name',
        lastname_label='Last Name',
        email_label='Email address',
        mobile_number_label='Mobile No.',
    )

    bank_params = BankParams(
        sort_code_label='Sort code',
        account_label='Bank Account Number',
    )

    address_params = AddressParams(
        postcode_label=['POSTCODE', 'PA_POSTCODE'],
        county_label=[None, 'PA_COUNTY'],
        city_label=['POSTTOWN', 'PA_POSTTOWN'],
        district_label=['DISTRICT', None],
        street_label=['STREET', 'PA_STREET'],
        house_number_label=None,
        flat_label=['FLAT', 'PA_FLAT'],
        house_name_label=['HOUSENAME', 'PA_HOUSENAME']
    )

    df_meta = MetaExtractor.extract(
        df=data,
        address_params=address_params,
        bank_params=bank_params,
        person_params=person_params
    )

    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(df_train=data, num_iterations=10)
        df_synthesized = synthesizer.synthesize(num_rows=len(data))

    assert df_synthesized.shape == data.shape


def test_addresses_from_file():
    data = pd.read_csv('data/annotations_nd.csv')

    address_meta_config = AddressMetaConfig(addresses_file='')

    address_params = AddressParams(
        postcode_label='POSTCODE',
        county_label='COUNTY',
        city_label='POSTTOWN',
        district_label='DISTRICT',
        street_label='STREET',
        house_number_label='HOUSENUMBER',
        flat_label='FLAT',
        house_name_label='HOUSENAME'
    )

    df_meta = MetaExtractor.extract(
        df=data,
        address_params=address_params
    )

    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(df_train=data, num_iterations=10)
        df_synthesized = synthesizer.synthesize(num_rows=len(data))

    assert df_synthesized.shape == data.shape

