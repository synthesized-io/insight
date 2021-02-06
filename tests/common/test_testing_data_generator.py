from synthesized.complex import TestingDataGenerator


def test_testing_data_generator():
    generator = TestingDataGenerator.from_yaml(config_file_name="tests/common/demo-config.yaml")
    df_synthesized = generator.synthesize(1000)

    assert df_synthesized.shape == (1000,  37)
