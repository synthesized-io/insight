import base64
import zlib
import logging

logger = logging.getLogger(__name__)


# 4 bytes-length crc-based hash
def h(x):
    return (zlib.crc32(x) % 2**32).to_bytes(4, 'big')


def digest(msg, key):
    return h(msg + key)


def check_invite_code(code: str, email: str, key: str):
    if len(code) != 8:
        return False
    try:
        key_bytes = key.encode('utf-8')
        email_bytes = email.encode('utf-8')
        code_bytes = base64.b16decode(code)
        d = digest(email_bytes, key_bytes)
        return d == code_bytes
    except Exception as e:
        logger.error(e)
        return False
