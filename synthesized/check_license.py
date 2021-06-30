import base64
import os
from datetime import datetime

import pkg_resources
import rsa

in_f = pkg_resources.resource_string("synthesized", ".pubkey")
pubkey = rsa.PublicKey.load_pkcs1(in_f)


def _check_license():
    try:
        key_env = 'SYNTHESIZED_KEY'
        key_path = '~/.synthesized/key'
        key_path = os.path.expanduser(key_path)
        print("Copyright (C) Synthesized Ltd. - All Rights Reserved")
        license_key = os.environ.get(key_env, None)
        if license_key is None and os.path.isfile(key_path):
            with open(key_path, 'r') as f:
                license_key = f.read().strip('\n ')
        if license_key is None:
            print('No license key detected (env variable {} or {})'.format(key_env, key_path))
            return False
        else:
            print('License key: ' + license_key)

        data, license_key = license_key.split('\n')
        rsa.verify(data.encode('utf-8'), base64.b64decode(license_key), pubkey)

        date = datetime.strptime(data.split(' ')[1], "%Y-%m-%d")
        now = datetime.now()
        if now >= date:
            print('License has been expired')
            return False
        else:
            print('Expires at: ' + str(date))
            return True
    except Exception as e:
        print(e)
        return False
