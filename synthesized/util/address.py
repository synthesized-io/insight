from typing import Optional, Pattern, Sequence

import pandas as pd


def get_postcode_key_from_df(df: pd.DataFrame, postcode_regex: Pattern[str], postcode_label: Optional[str] = None,
                             full_address_label: Optional[str] = None, postcode_level: int = 0,
                             postcodes: Optional[Sequence[str]] = None) -> pd.Series:

    if postcode_label is not None:
        return df[postcode_label].fillna('nan').apply(get_postcode_key, postcode_regex=postcode_regex,
                                                      postcode_level=postcode_level, postcodes=postcodes)
    elif full_address_label is not None:
        return df[full_address_label].fillna('nan').apply(get_postcode_key_from_address, postcode_regex=postcode_regex,
                                                          postcode_level=postcode_level, postcodes=postcodes)
    else:
        raise ValueError("Can't extract postcode if 'postcode_label' or 'full_address_label' are not defined.")


def get_postcode_key_from_address(address: str, postcode_regex: Pattern[str], postcode_level: int = 0,
                                  postcodes: Optional[Sequence[str]] = None):
    g = postcode_regex.search(address)
    if g is not None:
        return get_postcode_key(g.group(0), postcode_regex=postcode_regex, postcode_level=postcode_level,
                                postcodes=postcodes)
    return 'nan'


def get_postcode_key(
        postcode: str, postcode_regex: Pattern[str], postcode_level: int = 0,
        postcodes: Optional[Sequence[str]] = None
):
    """Returns the component of a postcode given the level and regex unless postcode in postcodes list.

    The postcode_level corresponds to 1 less than the number of regex groups
    included in the postcode.

    Examples:
        >>> postcode = "EC2A 2DP"
        >>> regex = re.compile("([A-Za-z]{1,2})([0-9]+[A-Za-z]?)( *[0-9]+[A-Za-z]{2})")
        >>> get_postcode_key(postcode, regex, postcode_level=0)
        "EC"
        >>> get_postcode_key(postcode, regex, postcode_level=1)
        "EC2A"
        >>> get_postcode_key(postcode, regex, postcode_level=2)
        "EC2A 2DP"

    """
    if postcodes and postcode in postcodes:
        return postcode

    if postcode == 'nan':
        return 'nan'

    match = postcode_regex.match(postcode)
    if match is None:
        return 'nan'

    return ''.join(match.groups()[:postcode_level + 1])
