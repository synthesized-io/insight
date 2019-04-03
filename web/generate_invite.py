import base64
import os
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
    if len(sys.argv) < 2:
        sys.exit('usage: <key>')
    key = sys.argv[1]

    key_bytes = key.encode('utf-8')
    msg_bytes = os.urandom(5)  # to have no padding in base32

    sig = hmacish(key_bytes, msg_bytes)

    code = bytes_to_base32_str(msg_bytes) + bytes_to_base32_str(sig)
    print(code)
