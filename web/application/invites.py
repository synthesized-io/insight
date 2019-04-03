import base64
import zlib
import logging

logger = logging.getLogger(__name__)


# 5 bytes-length crc-based hash
def h(x):
    return (zlib.crc32(x) % 2**32).to_bytes(5, 'big')


# hmac-like message digest
def hmacish(key, msg):
    return h(key + h(key + msg))


def bytes_to_base32_str(b):
    return base64.b32encode(b).decode('utf-8')


def check_invite_code(code: str, key: str):
    if len(code) != 16:
        return False
    try:
        key_bytes = key.encode('utf-8')
        msg_bytes = base64.b32decode(code[:8])
        orig_sig = base64.b32decode(code[8:])
        sig = hmacish(key_bytes, msg_bytes)
        return orig_sig == sig
    except Exception as e:
        logger.error(e)
        return False
