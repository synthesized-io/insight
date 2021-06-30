import base64

import rsa

if __name__ == '__main__':
    key_data = "user@synthesized.io 2021-12-31"
    with open("keys/privkey", "rb") as in_f:
        privkey = rsa.PrivateKey.load_pkcs1(in_f.read())

    signed_data = rsa.sign(key_data.encode('utf-8'), privkey, 'SHA-256')
    license_key = key_data + '\n' + base64.b64encode(signed_data).decode('ascii')

    print(license_key)
