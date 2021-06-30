import pytest
import base64
import datetime
import pathlib
import sys
sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())


from synthesized.licence import OptionalFeature
from keygen import generate_key, main


def test_generate():
    expiry = '2021-01-01'
    features = ['FAIRNESS', 'DIFFERENTIAL_PRIVACY']

    key = generate_key(expiry, features)
    key_decoded = base64.b64decode(key).decode().splitlines()
    assert len(key_decoded) == 3
    assert expiry == key_decoded[0]
    assert all(str(OptionalFeature[feature].value) in key_decoded[1].split(' ') for feature in features) is True

    features = ['*']
    key = generate_key(expiry, features)
    key_decoded = base64.b64decode(key).decode().splitlines()
    assert len(key_decoded) == 3
    assert expiry == key_decoded[0]
    assert key_decoded[1] == '*'

    key = generate_key(expiry)
    key_decoded = base64.b64decode(key).decode().splitlines()
    assert len(key_decoded) == 3
    assert expiry == key_decoded[0]
    assert key_decoded[1] == ''

    with pytest.raises(ValueError):
        generate_key(expiry, ['THIS_IS_NOT_A_KNOWN_FEATURE'])


def test_main(capfd):
    expiry = '2030-01-01'
    features = ['FAIRNESS', 'DIFFERENTIAL_PRIVACY']

    main(['--expirydate', expiry, '--features', *features])
    out, err = capfd.readouterr()
    key = generate_key(expiry, features)
    assert out.strip('\n') == key

    with pytest.raises(ValueError):
        main(['--expirydate', '1993-01-01', '--features', *features])

    main(['--days', '10', '--features', '*'])
    out, err = capfd.readouterr()

    date = datetime.datetime.now()
    assert base64.b64decode(out).decode().splitlines()[0] == (date + datetime.timedelta(days=10)).strftime('%Y-%m-%d')
    assert base64.b64decode(out).decode().splitlines()[1] == '*'

    main(['--days', '10'])
    out, err = capfd.readouterr()
    assert base64.b64decode(out).decode().splitlines()[1] == ''
