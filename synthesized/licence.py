import base64
import datetime
import enum
import json
import os
import pathlib
import warnings
from typing import Dict, List, Union

import pkg_resources
import rsa

KEY_VAR = "SYNTHESIZED_KEY"
KEY_FILEPATH = os.path.expanduser("~/.synthesized/key")
_PUBLIC_KEY = ".pubkey"

_KEY: str = ""
"""licence key"""
_SIGNATURE: str = ""
"""signed data"""
_LICENCE_DATA: str = ""
_LICENCE_INFO: Dict = {}
"""licence information"""

EXPIRY_DATE: str = ""
FEATURES: List[str] = []
"""public globals for checking at runtime"""

PathLike = Union[str, pathlib.Path]


class LicenceError(Exception):
    pass


class LicenceWarning(Warning):
    pass


class OptionalFeature(enum.Enum):
    """Optional features that are enabled/disabled by checking the licence key information.

    If the enumeration value is present in the licence, then this feature will be enabled. Each feature
    can be flagged by calling verify(feature=...), which will return True or False depending on the feature ids
    present in the licence.
    """

    FAIRNESS = 1
    DIFFERENTIAL_PRIVACY = 2


def verify(feature: OptionalFeature = None) -> bool:
    """Verify validity of the Synthesized licence.

    Checks expiry date and whether optional features are valid with given licence.

    Args:
        feature: Verify that this feature is enabled.

    Returns
        True if licence is within expiry date and optional features are active, False otherwise.
    """
    if not _LICENCE_INFO:
        _read_licence()

    _verify_signature(_LICENCE_DATA, _SIGNATURE, _read_public_key(_PUBLIC_KEY))

    date_is_valid = _verify_date(_LICENCE_INFO["expiry"])

    if feature is not None:
        feature_is_valid = _verify_features(_LICENCE_INFO["feature_ids"], feature)
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
    global _KEY, _SIGNATURE, _LICENCE_DATA, _LICENCE_INFO
    global EXPIRY_DATE, FEATURES  # public globals for checking at runtime

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
        _SIGNATURE = licence[:88]  # first 88 bytes contain signed data
        _LICENCE_DATA = licence[88:]
        _LICENCE_INFO = json.loads(_LICENCE_DATA)
        EXPIRY_DATE = _LICENCE_INFO["expiry"]
        if _LICENCE_INFO["feature_ids"] == ["*"]:
            FEATURES = [feature.name for feature in OptionalFeature]
        else:
            FEATURES = [feature.name for feature in OptionalFeature if feature.value in _LICENCE_INFO["feature_ids"]]
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


def _verify_features(feature_ids: List[Union[int, str]], feature: OptionalFeature) -> bool:
    if len(feature_ids) == 1 and feature_ids[0] == "*":
        return True
    elif feature.value in feature_ids:
        return True
    else:
        return False


def _verify_date(expiry) -> bool:
    expiry_date = datetime.datetime.strptime(expiry, "%Y-%m-%d")
    if datetime.datetime.now() > expiry_date:
        raise LicenceError("Licence expired. Please renew to continue using synthesized.")
    elif (expiry_date - datetime.datetime.now()).days < 7:
        warning_str = "Synthesized SDK licence expires in less than 7 days, "
        "please renew your licence to continue using the SDK"
        # warnings.warn prints the source line, so we separate out the string to avoid weird warning messages
        warnings.warn(warning_str, LicenceWarning)
    return True
