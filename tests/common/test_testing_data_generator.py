import pytest

from synthesized.complex.testing_data_generator import TestingDataGenerator  # type: ignore


@pytest.mark.skip("Not updated with new metas.")
def test_testing_data_generator():
    generator = TestingDataGenerator.from_yaml(config_file_name="tests/common/demo-config.yaml")
    df_synthesized = generator.synthesize(1000)

    assert df_synthesized.shape == (1000,  37)
