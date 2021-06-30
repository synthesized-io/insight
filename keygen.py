"""Generate licence key with given expiry date (yyyy-mm-dd) and optional features.

Optional features named must be defined as members in the synthesized.licence.OptionalFeature enum.

Examples:
    Create licence that expires after 2021-12-31. By default, all features are deactivated.

    $ python keygen.py --expirydate 2021-12-31

    Create licence that expires after 10 days, with only fairness feature activated.

    $ python keygen.py --days 10 --features FAIRNESS

    Create licence that expires after 10 days, with all features activated.

    $ python keygen.py --days 10 --features "*"

"""
import base64
import argparse
import datetime
import sys
import pathlib
from typing import List, Optional, Union
from importlib.abc import Loader
from importlib.util import spec_from_file_location, module_from_spec

import rsa


def get_feature_ids(features: Optional[Union[str, List[str]]]) -> Union[List[str], str]:
    if features is None:
        return ''
    if len(features) == 1 and features[0] == "*":
        return "*"
    else:
        ids = []
        for feature in features:
            try:
                ids.append(str(licence.OptionalFeature[feature.upper()].value))
            except KeyError:
                raise ValueError(
                    f"Feature '{feature}' not recognised. Optional features must be valid members defined in synthesized.licence.OptionalFeature. "
                )
        return ids


def generate_key(expiry: str, features: Optional[Union[str, List[str]]] = None) -> str:
    """Generate synthesized licence key.

    Expiry date and optional feature information is signed using a private SHA-256 key. The key
    is then a base64 encoded version of the expiry date, optional features and this signature.

    Args:
        expiry: licence key invalid after this date. Required format is yyyy-mm-dd.
        features: values of synthesized.licence.OptionalFeature enumeration.
    """
    with open("keys/privkey", "rb") as in_f:
        privkey = rsa.PrivateKey.load_pkcs1(in_f.read())

    key_data = f"{expiry}\n{' '.join(get_feature_ids(features))}"

    signed_data = rsa.sign(key_data.encode("utf-8"), privkey, "SHA-256")
    licence_key = key_data + "\n" + base64.b64encode(signed_data).decode("ascii")
    return base64.b64encode(licence_key.encode()).decode()


# need to hack the import to avoid licence check in __init__.py
def import_module_from_path(path: str):
    """Import a module from the given path. Taken from https://stackoverflow.com/a/58423785"""
    module_path = pathlib.Path(path).resolve()
    module_name = module_path.stem
    spec = spec_from_file_location(module_name, module_path)
    if spec is not None:
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        assert isinstance(spec.loader, Loader)
        spec.loader.exec_module(module)
    return module


licence = import_module_from_path("synthesized/licence.py")


def parse_args(args):
    parser = argparse.ArgumentParser("Generate licence key")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--expirydate", type=str, help="licence expiry date: yyyy-mm-dd")
    group.add_argument("--days", type=int, help="number of days licence is valid for.")
    parser.add_argument(
        "--features",
        default=None,
        nargs="+",
        type=str,
        help='names of optional features to activate, which must be members of synthesized.licence.OptionalFeatures. Defaults to "*", meaning all features are enabled.',
    )

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    if args.days is not None:
        expiry = (datetime.datetime.now() + datetime.timedelta(days=args.days)).strftime("%Y-%m-%d")
    else:
        expiry = args.expirydate

    if datetime.datetime.strptime(expiry, "%Y-%m-%d") < datetime.datetime.now():
        raise ValueError("Oops! Looks like the date entered is in the past!")

    print(generate_key(expiry, args.features))


if __name__ == "__main__":
    main(sys.argv[1:])
