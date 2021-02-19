from io import BytesIO

import pandas as pd
import pytest

from synthesized import HighDimSynthesizer, MetaExtractor
from synthesized.config import AddressParams, BankLabels, FormattedStringParams, MetaExtractorConfig, PersonLabels
from synthesized.metadata import DataFrameMeta


@pytest.mark.slow
@pytest.mark.skip(reason="Currently in development")
def test_annotations_all():
    data = pd.read_csv('data/annotations_nd.csv')
    person_params = PersonLabels(
        gender_label=None,
        title_label='Title (Tab selection)',
        firstname_label='First Name',
        lastname_label='Last Name',
        email_label='Email address',
        mobile_number_label='Mobile No.',
        username_label='username',
        password_label='password',
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
        house_name_label=['HOUSENAME', 'PA_HOUSENAME'],
        full_address_label=['Full Address', None],
    )

    formatted_string_params = FormattedStringParams(
        formatted_string_label=['COUNTY', 'PA_DISTRICT']
    )

    df_meta = MetaExtractor.extract(
        df=data,
        address_params=address_params,
        bank_params=bank_params,
        person_params=person_params,
        formatted_string_params=formatted_string_params,
        config=MetaExtractorConfig(label_to_regex={'COUNTY': '[a-z]{5,10}', 'PA_DISTRICT': '[a-z]{5,10}'})
    )

    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(df_train=data, num_iterations=10)
        df_synthesized = synthesizer.synthesize(num_rows=len(data))
        _ = synthesizer.encode(data)
        _ = synthesizer.encode_deterministic(data)

    assert df_synthesized.shape == data.shape

    # Ensure that import-export works
    f = BytesIO()
    synthesizer.export_model(f)

    f.seek(0)
    synthesizer2 = HighDimSynthesizer.import_model(f)
    df_synthesized2 = synthesizer.synthesize(num_rows=len(data))

    assert df_synthesized2.shape == data.shape


@pytest.mark.slow
@pytest.mark.skip(reason="Currently in development")
def test_addresses_from_file():
    data = pd.read_csv('data/annotations_nd.csv')

    meta_extractor_config = MetaExtractorConfig(addresses_file='data/addresses.jsonl.gz')

    address_params = AddressParams(
        postcode_label='POSTCODE',
        county_label='COUNTY',
        city_label='POSTTOWN',
        district_label='DISTRICT',
        street_label='STREET',
        house_number_label='HOUSENUMBER',
        flat_label='FLAT',
        house_name_label='HOUSENAME',
        full_address_label='Full Address',
    )

    df_meta = MetaExtractor.extract(
        df=data,
        config=meta_extractor_config,
        address_params=address_params
    )

    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(df_train=data, num_iterations=10)
        df_synthesized = synthesizer.synthesize(num_rows=len(data))
        _ = synthesizer.encode(data)
        _ = synthesizer.encode_deterministic(data)

    assert df_synthesized.shape == data.shape

    # Use real postcodes but do not learn them
    meta_extractor_config = MetaExtractorConfig(addresses_file='data/addresses.jsonl.gz', learn_postcodes=False)

    df_meta = MetaExtractor.extract(
        df=data,
        config=meta_extractor_config,
        address_params=address_params,
    )

    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(df_train=data, num_iterations=10)
        df_synthesized = synthesizer.synthesize(num_rows=len(data))

    assert df_synthesized.shape == data.shape


@pytest.mark.slow
@pytest.mark.skip(reason="Currently in development")
def test_pre_post_processing():
    df = pd.read_csv('data/annotations_nd.csv')
    person_params = PersonLabels(
        gender_label=None,
        title_label='Title (Tab selection)',
        firstname_label='First Name',
        lastname_label='Last Name',
        email_label='Email address',
        mobile_number_label='Mobile No.',
        username_label='username',
        password_label='password',
    )

    bank_params = BankParams(
        sort_code_label='Sort code',
        account_label='Bank Account Number',
    )

    address_params = AddressParams(
        postcode_label='POSTCODE',
        county_label='COUNTY',
        city_label='POSTTOWN',
        district_label='DISTRICT',
        street_label='STREET',
        house_number_label='HOUSENUMBER',
        flat_label='FLAT',
        house_name_label='HOUSENAME',
        full_address_label='Full Address',
    )

    df_meta = MetaExtractor.extract(
            df=df,
            address_params=address_params,
            bank_params=bank_params,
            person_params=person_params
        )
    df_pre = df_meta.preprocess(df=df)
    df_post = df_meta.postprocess(df=df_pre)
    assert df.shape == df_post.shape

    df_pre = df_meta.preprocess(df=df, max_workers=None)
    df_post = df_meta.postprocess(df=df_pre, max_workers=None)
    assert df.shape == df_post.shape

    # Ensure that get_variables/set_variables works
    variables = df_meta.get_variables()
    df_meta2 = DataFrameMeta.from_dict(variables)
