import base64
import sys
import zlib


# 5 bytes-length crc-based hash
def h(x):
    return (zlib.crc32(x) % 2**32).to_bytes(5, 'big')


# hmac-like message digest
def hmacish(key, msg):
    return h(key + h(key + msg))


def bytes_to_base32_str(b):
    return base64.b32encode(b).decode('utf-8')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('usage: <key> <code>')
    key = sys.argv[1]
    code = sys.argv[2]

    if len(code) != 16:
        sys.exit('wrong code length (!= 16)')

    key_bytes = key.encode('utf-8')
    msg_bytes = base64.b32decode(code[:8])
    orig_sig = base64.b32decode(code[8:])

    sig = hmacish(key_bytes, msg_bytes)

    if orig_sig == sig:
        print('valid')
    else:
        print('invalid')
