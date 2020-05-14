from datetime import datetime
import enum
import gzip
import lzma
import os
import pickle
import tempfile
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

    :ivar BinaryType binary_type: The type of binary being created (Dataset or Model)
    :ivar bytes body: The byte representation of model/dataset which will be stored in body of binary
    :ivar CompressionType body_compression: The algorithm which was used to compress the body
    :ivar str title: The title of the Dataset/Model which this binary encodes
    :ivar str description: The description of the Dataset/Model which this binary encodes
    :ivar str author: The author of the Dataset/Model which this binary encodes
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
        self.body = body
        self.body_compression = body_compression
        self.title = title
        self.description = description
        self.author = author
        self.creation_date = creation_date

        self.binary_file = None

    def serialize_to_file(self, file=None):
        """Pickle body and metadata then store in binary file.

        :param str file: (optional) The path of a file to serialize binary into.
        If not provided, a temporary file will be created.
        """
        binary = self.serialize()
        if file is None:
            file = self._create_temp_file(prefix=self.get_id())
        self.binary_file = file
        with open(self.binary_file, 'wb') as f:
            f.write(binary)
        return self.binary_file

    def deserialize_from_file(self, file: str):
        """Populate binary object from binary file.

        :param str file: the path of a binary file to deserialize
        """
        with open(file, 'rb') as f:
            binary = f.read()

        self.deserialize(binary)

    def get_body(self):
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

    def set_body(self, body: bytes):
        """Compress body of binary, and set in binary.

        :param str body: uncompressed byte array to store in binary
        """
        if self.body_compression == CompressionType.GZIP:
            body = gzip.compress(body)
        elif self.body_compression == CompressionType.ZLIB:
            body = zlib.compress(body)
        elif self.body_compression == CompressionType.LZMA:
            body = lzma.compress(body)
        self.body = body

    def delete_binary_file(self):
        """Cleans up temporary file."""
        try:
            os.remove(self.binary_file)
        except OSError:
            # file descriptor already closed, ignore error
            pass

    def serialize(self):
        """Pickle body and metadata."""
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
        return binary

    def deserialize(self, binary: bytes):
        """Un-Pickle body and metadata from bytes."""
        try:
            binary_object = pickle.loads(binary)
        except Exception as exc:
            raise BinaryDeserializeException(exc)

        self.binary_type = binary_object['body_type']
        self.body = binary_object['body']
        self.body_compression = binary_object['body_compression']

        meta_data = binary_object.get('meta_data', None)
        if meta_data:
            self.title = meta_data.get('title', None)
            self.description = meta_data.get('description', None)
            self.author = meta_data.get('author', None)
            self.creation_date = meta_data.get('creation_date', None)

    def get_id(self):
        return f"{self.author}_{self.creation_date.strftime('%m-%d-%Y-%H-%M-%S')}"

    @staticmethod
    def _create_temp_file(**kwargs) -> str:
        _, file = tempfile.mkstemp(**kwargs)
        return file


class ModelBinary(Binary):
    def __init__(self, *args, **kwargs):
        super().__init__(binary_type=BinaryType.MODEL, *args, **kwargs)


class DatasetBinary(Binary):
    def __init__(self, *args, **kwargs):
        super().__init__(binary_type=BinaryType.DATASET, *args, **kwargs)
