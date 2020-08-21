from typing import BinaryIO
from datetime import datetime
import enum
import gzip
import lzma
import pickle
import zlib


class BinarySerializeException(Exception):
    """Base exception class for binary serializing errors"""


class BinaryDeserializeException(Exception):
    """Base exception class for binary deserializing errors"""


class BinaryType(enum.Enum):
    MODEL = 'model'
    DATASET = 'dataset'


class CompressionType(enum.Enum):
    ZLIB = 'zlib'
    GZIP = 'gzip'
    LZMA = 'lzma'
    NONE = 'none'


class Binary:
    """Initializes an instance of the Binary with passed params.

    Args:
        binary_type: The type of binary being created (Dataset or Model)
        body: The byte representation of Dataset/Model which will be stored in body of binary
        body_compression: The algorithm which was used to compress the body
        title: The title of the Dataset/Model which this binary encodes
        description: The description of the Dataset/Model which this binary encodes
        author: The author of the Dataset/Model which this binary encodes
    """
    def __init__(
            self,
            binary_type: BinaryType = None,
            body: bytes = None,
            body_compression: CompressionType = CompressionType.NONE,
            title: str = None,
            description: str = None,
            author: str = None,
            creation_date: datetime = datetime.now()):
        self.binary_type = binary_type
        self.body_compression = body_compression
        self.body = body
        self.title = title
        self.description = description
        self.author = author
        self.creation_date = creation_date

        if body is not None:
            self._set_body(body)

    def serialize(self, fp: BinaryIO):
        """Pickle body and metadata then store in binary file.

        Args:
            fp: The binary file to write to.
        """
        binary_object = {
            'meta_data': {
                'title': self.title,
                'description': self.description,
                'author': self.author,
                'creation_date': self.creation_date
            },
            'body_type': self.binary_type,
            'body_compression': self.body_compression,
            'body': self.body
        }
        try:
            binary = pickle.dumps(binary_object, protocol=4)
        except Exception as exc:
            raise BinarySerializeException(exc)

        fp.write(binary)

    def deserialize(self, fp: BinaryIO):
        """Populate binary object from binary file.

        Args:
            fp: The binary file to read from.
        """
        binary = fp.read()

        try:
            binary_object = pickle.loads(binary)
        except Exception as exc:
            raise BinaryDeserializeException(exc)

        self.binary_type = binary_object['body_type']
        self._set_body(binary_object['body'])
        self.body_compression = binary_object['body_compression']

        meta_data = binary_object.get('meta_data', None)
        if meta_data:
            self.title = meta_data.get('title', None)
            self.description = meta_data.get('description', None)
            self.author = meta_data.get('author', None)
            self.creation_date = meta_data.get('creation_date', None)

    def get_body(self) -> bytes:
        """Uncompress and retrieve body bytes of binary."""
        assert isinstance(self.body, bytes)
        if self.body_compression == CompressionType.GZIP:
            decompressed_body = gzip.decompress(self.body)
        elif self.body_compression == CompressionType.ZLIB:
            decompressed_body = zlib.decompress(self.body)
        elif self.body_compression == CompressionType.LZMA:
            decompressed_body = lzma.decompress(self.body)
        else:
            decompressed_body = self.body
        return decompressed_body

    def _set_body(self, body: bytes):
        """Compress body of binary, and set in binary.

        Args:
            body: uncompressed byte array to store in binary
        """
        if self.body_compression == CompressionType.GZIP:
            body = gzip.compress(body)
        elif self.body_compression == CompressionType.ZLIB:
            body = zlib.compress(body)
        elif self.body_compression == CompressionType.LZMA:
            body = lzma.compress(body)
        self.body = body

    def get_id(self):
        """Returns a string identifying the binary object."""
        return f"{self.author}_{self.creation_date.strftime('%m-%d-%Y-%H-%M-%S')}"


class ModelBinary(Binary):
    """An instance of a binary for synthesizer models."""
    def __init__(self, *args, **kwargs):
        super().__init__(binary_type=BinaryType.MODEL, *args, **kwargs)


class DatasetBinary(Binary):
    """An instance of a binary for datasets."""
    def __init__(self, *args, **kwargs):
        super().__init__(binary_type=BinaryType.DATASET, *args, **kwargs)
