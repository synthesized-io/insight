import pandas as pd

from synthesized.config import AddressParams
from synthesized.metadata_new.value import Address

df = pd.DataFrame({
    "street": ["Blah Drive", "Placeholder Avenue", "Test Road"],
    "city": ["Cambridge", "London", "Swansea"],
    "house name": ["", "Housey McHouseface", ""],
    "house number": ["1", "", "42"],
})

collapsed_df = pd.DataFrame({
    "collapsed_address": ["Cambridge|Blah Drive|1|",
                          "London|Placeholder Avenue||Housey McHouseface",
                          "Swansea|Test Road|42|"]
})


def test_collapse_expand_parity():
    params = AddressParams(street_label="street", city_label="city",
                           house_name_label="house name", house_number_label="house number")

    meta = Address(name="address", address_params=params)
    df_new = df.copy()
    meta.collapse(df_new)
    assert (df_new == collapsed_df).all().all()
    meta.expand(df_new)
    assert (df[meta.address_labels] == df_new).all().all()
