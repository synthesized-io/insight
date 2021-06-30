import enum
import pathlib
import pkg_resources
import rsa
import os
import datetime
import base64
from typing import Union

KEY_VAR = "SYNTHESIZED_KEY"
KEY_FILEPATH = os.path.expanduser("~/.synthesized/key")
_PUBLIC_KEY = ".pubkey"

_KEY: str = ""
"""licence key"""
_EXPIRY: str = ""
"""expiry date"""
_FEATURES: str = ""
"""feature ids that are activated. See OptionalFeature."""
_SIGNATURE: str = ""
"""signed data"""

PathLike = Union[str, pathlib.Path]


class LicenceError(Exception):
    pass


class OptionalFeature(enum.Enum):
    """Optional features that are enabled/disabled by checking the licence key information.

    If the enumeration value is present in the licence, then this feature will be enabled. Each feature
    can be flagged by calling verify(feature=...), which will return True or False depending on the feature ids
    present in the licence.
    """

    FAIRNESS = 1
    DIFFERENTIAL_PRIVACY = 2


def verify(feature: OptionalFeature = None, verbose: bool = False) -> bool:
    """Verify validity of the synthesized licence.

    Checks expiry date and whether optional features are valid with given licence.

    Args:
        feature: Verify that this feature is enabled.

    Returns
        True if licence is within expiry date and optional features are active, False otherwise.
    """
    if not _EXPIRY:
        _read_licence()
        if verbose:
            print("Copyright (C) Synthesized Ltd. - All Rights Reserved.")
            print(f"Licence: {_KEY}")
            print(f"Expires: {_EXPIRY}")

    data = f"""{_EXPIRY}\n{_FEATURES}"""
    _verify_signature(data, _SIGNATURE, _read_public_key(_PUBLIC_KEY))

    date_is_valid = _verify_date(_EXPIRY)

    if feature is not None:
        feature_is_valid = _verify_features(_FEATURES, feature)
    else:
        feature_is_valid = True

    if date_is_valid and feature_is_valid:
        return True
    else:
        return False


def _read_licence_from_file(path: PathLike) -> str:
    with open(path, "r") as f:
        return f.read()


def _read_licence_from_env(var: str) -> str:
    return os.environ[var]


def _read_licence() -> None:
    global _KEY, _EXPIRY, _FEATURES, _SIGNATURE

    try:
        licence = _read_licence_from_env(KEY_VAR)
    except KeyError:
        try:
            licence = _read_licence_from_file(KEY_FILEPATH)
        except FileNotFoundError:
            raise LicenceError(
                f"""Unable to read licence. Ensure {KEY_VAR} is set as an environment variable or
                                licence file exists at {KEY_FILEPATH}."""
            )

    try:
        _KEY = licence.strip('\n')
        licence = base64.b64decode(licence).decode('utf-8')
        _EXPIRY, _FEATURES, _SIGNATURE = licence.splitlines()
    except ValueError:
        raise LicenceError("Unable to read licence. Please contact team@synthesized.io.")


def _read_public_key(filename: str = ".pubkey") -> rsa.PublicKey:
    key_file = pkg_resources.resource_string("synthesized", filename)
    return rsa.PublicKey.load_pkcs1(key_file)


def _verify_signature(data: str, signature: str, public_key: rsa.PublicKey) -> bool:
    try:
        rsa.verify(data.encode("utf-8"), base64.b64decode(_SIGNATURE), public_key)
        return True
    except rsa.VerificationError:
        raise LicenceError("Unable to verify licence. Please contact team@synthesized.io.")


def _verify_features(feature_ids: str, feature: OptionalFeature) -> bool:
    features = feature_ids.split(" ")
    if len(features) == 1 and features[0] == "*":
        return True
    elif str(feature.value) in features:
        return True
    else:
        return False


def _verify_date(expiry) -> bool:
    if datetime.datetime.now() > datetime.datetime.strptime(expiry, "%Y-%m-%d"):
        raise LicenceError("Licence expired. Please renew to continue using synthesized.")
    else:
        return True
