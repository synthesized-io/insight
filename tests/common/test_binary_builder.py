import pytest
from io import BytesIO

import numpy as np
import pandas as pd

from synthesized.complex.binary_builder import BinaryType, CompressionType, ModelBinary, DatasetBinary, Binary


@pytest.mark.fast
def test_model_binary_init():
    model_binary = ModelBinary()
    assert model_binary.binary_type == BinaryType.MODEL


@pytest.mark.fast
def test_dataset_binary_init():
    dataset_binary = DatasetBinary()
    assert dataset_binary.binary_type == BinaryType.DATASET


@pytest.mark.fast
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
    dataset_binary = DatasetBinary(body=body, body_compression=body_compression)

    body_1 = dataset_binary.get_body()
    assert body == body_1


@pytest.mark.fast
@pytest.mark.parametrize(
    "body,title,description,author,binary_type",
    [
        pytest.param('test body'.encode(), 'test title', 'test description', 'test author', BinaryType.DATASET),
        pytest.param('test body 1'.encode(), 'test title 1', 'test description 1', 'test author 1', BinaryType.MODEL),
    ]
)
def test_serialize_and_deserialize(body, title, description, author, binary_type):
    binary = Binary(binary_type=binary_type, body=body, title=title, description=description, author=author)
    creation_date = binary.creation_date

    f = BytesIO()
    binary.serialize(f)

    f.seek(0)
    unknown_binary = Binary()
    unknown_binary.deserialize(f)

    assert unknown_binary.binary_type == binary_type
    assert unknown_binary.body_compression == CompressionType.NONE
    assert unknown_binary.body == body
    assert unknown_binary.title == title
    assert unknown_binary.description == description
    assert unknown_binary.author == author
    assert unknown_binary.creation_date == creation_date


@pytest.mark.fast
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

    df = pd.DataFrame({'r': np.random.normal(loc=5000, scale=1000, size=1000)})

    dataset_binary = DatasetBinary(body=df, title=title, description=description, author=author)

    dataset_binary_file = BytesIO()
    dataset_binary.serialize(dataset_binary_file)
    dataset_binary_file.seek(0)

    unknown_binary = DatasetBinary()
    unknown_binary.deserialize(dataset_binary_file)

    assert df.columns == unknown_binary.body.columns
    assert all(df.values) == all(unknown_binary.body.values)
