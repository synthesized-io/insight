import random
from io import BytesIO

import pandas as pd
import pytest

from synthesized import HighDimSynthesizer, MetaExtractor
from synthesized.config import AddressLabels, BankLabels, HighDimConfig, PersonLabels
from synthesized.metadata.value import Address, Bank, FormattedString, Person


@pytest.mark.parametrize(
    "high_dim_config",
    [
        pytest.param(HighDimConfig(), id="default_config"),
        pytest.param(HighDimConfig(addresses_file="data/addresses.jsonl.gz"), id="addresses_file"),
        pytest.param(HighDimConfig(learn_postcodes=True), id="learn_postcodes"),
        pytest.param(HighDimConfig(person_locale='ru_RU'), id="russian"),
    ]
)
@pytest.mark.slow
def test_synthesis_w_annotations(high_dim_config):
    data = pd.read_csv('data/annotations_nd.csv')

    person = Person(
        name='person',
        labels=PersonLabels(
            gender_label=None,
            title_label='Title (Tab selection)',
            firstname_label='First Name',
            lastname_label='Last Name',
            email_label='Email address',
            mobile_number_label='Mobile No.',
            username_label='username',
            password_label='password'
        )
    )
    bank = Bank(
        name='bank',
        labels=BankLabels(
            sort_code_label='Sort code',
            account_label='Bank Account Number',
        )
    )
    address = Address(
        name='address',
        labels=AddressLabels(
            postcode_label='POSTCODE',
            city_label='POSTTOWN',
            district_label='DISTRICT',
            street_label='STREET',
            flat_label='FLAT',
            house_name_label='HOUSENAME',
            full_address_label='Full Address',
        )
    )

    pa_address = Address(
        name='pa_address',
        labels=AddressLabels(
            postcode_label='PA_POSTCODE',
            county_label='PA_COUNTY',
            city_label='PA_POSTTOWN',
            street_label='PA_STREET',
            flat_label='PA_FLAT',
            house_name_label='PA_HOUSENAME',
        )
    )

    county = FormattedString(
        name='COUNTY',
        pattern='[A-Za-z]{5,10}'
    )

    pa_district = FormattedString(
        name='PA_DISTRICT',
        pattern='[A-Z][a-z]{8,15}'
    )

    annotations = [person, bank, address, pa_address, county, pa_district]

    df_meta = MetaExtractor.extract(
        df=data,
        annotations=annotations
    )

    synthesizer = HighDimSynthesizer(df_meta=df_meta, config=high_dim_config)
    synthesizer.learn(df_train=data, num_iterations=10)
    df_synthesized = synthesizer.synthesize(num_rows=len(data))

    _ = synthesizer.encode(data)
    _ = synthesizer.encode_deterministic(data)
    _ = synthesizer.encode_deterministic(data, produce_nans=True)
    assert df_synthesized.shape == data.shape

    f = BytesIO()
    synthesizer.export_model(f)

    f.seek(0)
    synthesizer2 = HighDimSynthesizer.import_model(f)
    df_synthesized2 = synthesizer2.synthesize(num_rows=len(data))

    assert df_synthesized2.shape == data.shape
