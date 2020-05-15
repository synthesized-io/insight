import csv
import os
import pytest
import pickle
from tempfile import mkstemp

from synthesized.common.binary_builder import BinaryType, CompressionType, ModelBinary, DatasetBinary, Binary


def store_file_in_binary_body(binary: Binary, file: str):
    """Convert file to bytes, compress and store in binary object."""
    with open(file, 'rb') as f:
        body = f.read()
    binary.set_body(body)


def extract_file_from_binary_body(binary: Binary, file_extension: str = None):
    """Extract uncompressed body into a file."""
    assert isinstance(binary.body, bytes)
    decompressed_body = binary.get_body()
    file_path = binary._create_temp_file(suffix=file_extension)
    with open(file_path, 'wb') as f:
        f.write(decompressed_body)
    return file_path


def test_model_binary_init():
    model_binary = ModelBinary()
    assert model_binary.binary_type == BinaryType.MODEL


def test_dataset_binary_init():
    dataset_binary = DatasetBinary()
    assert dataset_binary.binary_type == BinaryType.DATASET


@pytest.mark.parametrize(
    "body_compression",
    [
        pytest.param(CompressionType.NONE),
        pytest.param(CompressionType.ZLIB),
        pytest.param(CompressionType.GZIP),
        pytest.param(CompressionType.LZMA)
    ]
)
def test_set_get_body(body_compression):
    body = 'this is a test'.encode()
    dataset_binary = DatasetBinary(body_compression=body_compression)
    dataset_binary.set_body(body)

    body_1 = dataset_binary.get_body()
    assert body == body_1


@pytest.mark.parametrize(
    "body,title,description,author,binary_type",
    [
        pytest.param('test body'.encode(), 'test title', 'test description', 'test author', BinaryType.DATASET),
        pytest.param('test body 1'.encode(), 'test title 1', 'test description 1', 'test author 1', BinaryType.MODEL),
    ]
)
def test_deserialize(body, title, description, author, binary_type):
    binary = Binary(binary_type=binary_type, body=body, title=title, description=description, author=author)
    binary_file = binary.serialize_to_file()
    creation_date = binary.creation_date

    unknown_binary = Binary()
    unknown_binary.deserialize_from_file(binary_file)
    assert unknown_binary.binary_type == binary_type
    assert unknown_binary.body_compression == CompressionType.NONE
    assert unknown_binary.body == body
    assert unknown_binary.title == title
    assert unknown_binary.description == description
    assert unknown_binary.author == author
    assert unknown_binary.creation_date == creation_date

    binary.delete_binary_file()


def test_deserialize_no_metadata():
    body = 'test body'.encode()
    body_type = BinaryType.MODEL
    body_compression = CompressionType.NONE

    binary_obj = {
        'body': body,
        'body_type': body_type,
        'body_compression': body_compression
    }
    binary = pickle.dumps(binary_obj, 4)

    unknown_binary = Binary()
    unknown_binary.deserialize(binary)
    assert unknown_binary.binary_type == body_type
    assert unknown_binary.body_compression == body_compression
    assert unknown_binary.body == body


def test_serialize():
    body = 'model body'.encode()
    title = 'test title'
    description = 'test description'
    author = 'test author'
    model_binary = ModelBinary(body=body, title=title, description=description, author=author)

    model_binary_file = model_binary.serialize_to_file()

    assert os.path.isfile(model_binary_file)

    model_binary.delete_binary_file()

    assert not os.path.isfile(model_binary_file)


@pytest.mark.parametrize(
    "body_compression",
    [
        pytest.param(CompressionType.NONE),
        pytest.param(CompressionType.ZLIB),
        pytest.param(CompressionType.GZIP),
        pytest.param(CompressionType.GZIP)
    ]
)
def test_serialize_and_deserialize_dataset(body_compression):
    title = 'test title'
    description = 'test description'
    author = 'test author'
    dataset_binary = DatasetBinary(title=title, description=description, author=author)

    csv_fd, csv_file = mkstemp(suffix='.csv')
    csv_content = ['1', '2', '3']
    with open(csv_fd, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_content)

    store_file_in_binary_body(dataset_binary, csv_file)

    dataset_binary_file = dataset_binary.serialize_to_file()

    assert os.path.isfile(dataset_binary_file)

    unknown_binary = DatasetBinary()
    unknown_binary.deserialize_from_file(dataset_binary_file)
    extracted_csv_file = extract_file_from_binary_body(unknown_binary, 'csv')
    with open(extracted_csv_file) as f:
        csv_reader = csv.reader(f)
        extracted_csv_content = next(csv_reader)

    assert extracted_csv_content == csv_content

    dataset_binary.delete_binary_file()

    assert not os.path.isfile(dataset_binary_file)

    os.remove(extracted_csv_file)
    os.remove(csv_file)
