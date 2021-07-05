import base64
import datetime
import json
import pathlib
import sys

import pytest

sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())


from keygen import generate_key, main
from synthesized.licence import OptionalFeature


def test_generate():
    expiry = '2021-01-01'
    features = ['FAIRNESS', 'DIFFERENTIAL_PRIVACY']
    extras = {"info": "1729", "context": "synthesized"}

    key = generate_key(expiry, features, **extras)
    key_decoded = base64.b64decode(key).decode()
    key_data = json.loads(key_decoded[88:])
    assert len(key_data) == 4
    assert expiry == key_data["expiry"]
    assert all(OptionalFeature[feature].value in key_data["feature_ids"] for feature in features) is True
    assert "info" in extras and extras["info"] == "1729"
    assert "context" in extras and extras["context"] == "synthesized"

    features = ['*']
    key = generate_key(expiry, features, **extras)
    key_decoded = base64.b64decode(key).decode()
    key_data = json.loads(key_decoded[88:])
    assert len(key_data) == 4
    assert expiry == key_data["expiry"]
    assert key_data["feature_ids"] == ['*']
    assert "info" in extras and extras["info"] == "1729"
    assert "context" in extras and extras["context"] == "synthesized"

    key = generate_key(expiry)
    key_decoded = base64.b64decode(key).decode()
    key_data = json.loads(key_decoded[88:])
    assert len(key_data) == 2
    assert expiry == key_data["expiry"]
    assert key_data["feature_ids"] == []

    with pytest.raises(ValueError):
        generate_key(expiry, ['THIS_IS_NOT_A_KNOWN_FEATURE'])


def test_main(capfd):
    expiry = '2030-01-01'
    features = ['FAIRNESS', 'DIFFERENTIAL_PRIVACY']
    extras = ["info=1729", "context=synthesized"]

    main(['--expirydate', expiry, '--features', *features, '--extras', *extras])
    out, err = capfd.readouterr()
    key = generate_key(expiry, features, **{"info": "1729", "context": "synthesized"})
    assert out.strip('\n') == key

    with pytest.raises(ValueError):
        main(['--expirydate', '1993-01-01', '--features', *features])

    main(['--days', '10', '--features', '*', '--extras', "info=1729", "context=synthesized"])
    out, err = capfd.readouterr()

    date = datetime.datetime.now()
    key_decoded = base64.b64decode(out).decode()
    key_data = json.loads(key_decoded[88:])
    assert key_data["expiry"] == (date + datetime.timedelta(days=10)).strftime('%Y-%m-%d')
    assert key_data["feature_ids"] == ['*']
    assert key_data["info"] == "1729"
    assert key_data["context"] == "synthesized"

    main(['--days', '10'])
    out, err = capfd.readouterr()
    key_decoded = base64.b64decode(out).decode()
    key_data = json.loads(key_decoded[88:])
    assert key_data["feature_ids"] == []
