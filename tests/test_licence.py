import datetime
import json
import os
import pathlib
import sys

import pytest

sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())

import synthesized.licence
from keygen import generate_key


@pytest.fixture
def valid_licence():
    date = datetime.datetime.today() + datetime.timedelta(days=1)
    return generate_key(date.strftime('%Y-%m-%d'), ['FAIRNESS', 'DIFFERENTIAL_PRIVACY'])


@pytest.fixture
def valid_licence_all_features():
    date = datetime.datetime.today() + datetime.timedelta(days=1)
    return generate_key(date.strftime('%Y-%m-%d'), ['*'])


@pytest.fixture
def valid_licence_one_feature():
    date = datetime.datetime.today() + datetime.timedelta(days=1)
    return generate_key(date.strftime('%Y-%m-%d'), ['FAIRNESS'])


@pytest.fixture
def valid_licence_no_features():
    date = datetime.datetime.today() + datetime.timedelta(days=1)
    return generate_key(date.strftime('%Y-%m-%d'))


@pytest.fixture
def expired_licence():
    date = '1993-01-01'
    return generate_key(date, ['FAIRNESS', 'DIFFERENTIAL_PRIVACY'])


def test_read_licence_from_env(valid_licence):
    synthesized.licence.KEY_FILEPATH = ''
    os.environ[synthesized.licence.KEY_VAR] = valid_licence
    synthesized.licence._read_licence_from_env(synthesized.licence.KEY_VAR)
    synthesized.licence._read_licence()

    date = datetime.datetime.today() + datetime.timedelta(days=1)

    assert synthesized.licence._LICENCE_INFO["expiry"] == date.strftime('%Y-%m-%d')
    assert synthesized.licence._LICENCE_INFO["feature_ids"] == [1, 2]

    del os.environ[synthesized.licence.KEY_VAR]
    with pytest.raises(synthesized.licence.LicenceError):
        synthesized.licence._read_licence()

    os.environ[synthesized.licence.KEY_VAR] = ''
    with pytest.raises(synthesized.licence.LicenceError):
        synthesized.licence._read_licence()


def test_read_licence_from_file(valid_licence, tmpdir):
    with open(f"{tmpdir}/key", "w") as f:
        f.write(valid_licence)

    del os.environ[synthesized.licence.KEY_VAR]
    synthesized.licence.KEY_FILEPATH = f"{tmpdir}/key"

    synthesized.licence._read_licence_from_file(synthesized.licence.KEY_FILEPATH)
    synthesized.licence._read_licence()

    date = datetime.datetime.today() + datetime.timedelta(days=1)

    assert synthesized.licence._LICENCE_INFO["expiry"] == date.strftime('%Y-%m-%d')
    assert synthesized.licence._LICENCE_INFO["feature_ids"] == [1, 2]


def test_verify_expired(expired_licence):
    synthesized.licence.KEY_FILEPATH = ''
    os.environ[synthesized.licence.KEY_VAR] = expired_licence
    synthesized.licence._read_licence()
    with pytest.raises(synthesized.licence.LicenceError):
        synthesized.licence.verify()


def test_verify_colab():

    colab_licence = "QjJsMkdFekw3aHZkSFoxK1pDenRmeXZtZ29yaTFaaGxqRStkKy9SdHZRa1hSSkdLVThoLzk5S0UyWUlJRnBnQk9qUHRrejVkOHZMcVYvbjIyOHZlY3c9PXsiZXhwaXJ5IjogIjIwMjQtMDQtMDgiLCAiZmVhdHVyZV9pZHMiOiBbIioiXSwgImNvbGFiIjogdHJ1ZSwgImVtYWlsIjogInJvYkBzeW50aGVzaXplZC5pbyJ9"
    os.environ[synthesized.licence.KEY_VAR] = colab_licence
    with pytest.raises(synthesized.licence.LicenceError):
        synthesized.licence._read_licence()
        synthesized.licence.verify()


def test_verify_valid(valid_licence):
    os.environ[synthesized.licence.KEY_VAR] = valid_licence
    synthesized.licence._read_licence()
    assert synthesized.licence.verify() is True


def test_verify_date(valid_licence, expired_licence):
    synthesized.licence.KEY_FILEPATH = ''
    os.environ[synthesized.licence.KEY_VAR] = valid_licence
    synthesized.licence._read_licence()
    assert synthesized.licence._verify_date(synthesized.licence._LICENCE_INFO["expiry"]) is True
    with pytest.raises(synthesized.licence.LicenceError):
        os.environ[synthesized.licence.KEY_VAR] = expired_licence
        synthesized.licence._read_licence()
        synthesized.licence._verify_date(synthesized.licence._LICENCE_INFO["expiry"])


def test_verify_features(valid_licence_all_features, valid_licence_one_feature, valid_licence_no_features):
    synthesized.licence.KEY_FILEPATH = ''
    os.environ[synthesized.licence.KEY_VAR] = valid_licence_all_features
    synthesized.licence._read_licence()

    for feature in synthesized.licence.OptionalFeature:
        assert synthesized.licence.verify(feature=feature) is True

    os.environ[synthesized.licence.KEY_VAR] = valid_licence_one_feature
    synthesized.licence._read_licence()
    assert synthesized.licence.verify(feature=synthesized.licence.OptionalFeature.FAIRNESS) is True
    assert synthesized.licence.verify(feature=synthesized.licence.OptionalFeature.DIFFERENTIAL_PRIVACY) is False

    os.environ[synthesized.licence.KEY_VAR] = valid_licence_no_features
    synthesized.licence._read_licence()
    for feature in synthesized.licence.OptionalFeature:
        assert synthesized.licence.verify(feature=feature) is False


def test_verify_signature(valid_licence):
    synthesized.licence.KEY_FILEPATH = ''
    os.environ[synthesized.licence.KEY_VAR] = valid_licence
    synthesized.licence._read_licence()

    data = synthesized.licence._LICENCE_DATA
    signature = synthesized.licence._SIGNATURE
    public_key = synthesized.licence._read_public_key()

    assert synthesized.licence._verify_signature(data, signature, public_key) is True
    with pytest.raises(synthesized.licence.LicenceError):
        data = {"expiry": "2100-01-01", "feature_ids": synthesized.licence._LICENCE_INFO["feature_ids"]}
        synthesized.licence._verify_signature(json.dumps(data), signature, public_key)
