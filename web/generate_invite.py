import base64
import sys
import zlib


# 4 bytes-length crc-based hash
def h(x):
    return (zlib.crc32(x) % 2**32).to_bytes(4, 'little')


def digest(msg, key):
    return h(msg + key)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('usage: <key> <email>')
    key = sys.argv[1]
    email = sys.argv[2]

    key_bytes = key.encode('utf-8')
    msg_bytes = email.encode('utf-8')

    d = digest(msg_bytes, key_bytes)

    print(base64.b16encode(d).decode('utf-8'))
