import base64
import sys
import zlib


# 4 bytes-length crc-based hash
def h(x):
    return (zlib.crc32(x) % 2**32).to_bytes(4, 'little')


def digest(msg, key):
    return h(msg + key)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit('usage: <key> <email> <code>')
    key = sys.argv[1]
    email = sys.argv[2]
    code = sys.argv[3]

    if len(code) != 8:
        sys.exit('wrong code length (!= 8)')

    key_bytes = key.encode('utf-8')
    email_bytes = email.encode('utf-8')
    code_bytes = base64.b16decode(code)

    d = digest(email_bytes, key_bytes)

    if d == code_bytes:
        print('valid')
    else:
        print('invalid')
