import pandas as pd

from synthesized import HighDimSynthesizer, MetaExtractor


def test_synthesize_constant_column():
    test = {}
    for i in range(0, 100):
        test[i] = {
            "name": str("Test" + str(i)),
            "firstname": "Albert",
            "street": "Post" + str(i),
            "houseno": 32 + i,
            "age": 3 + i
        }
    testdf = pd.DataFrame.from_dict(test, orient="index")
    dfmeta = MetaExtractor.extract(df=testdf)
    synthesizer = HighDimSynthesizer(df_meta=dfmeta)
    synthesizer.learn(testdf, num_iterations=30)
    _ = synthesizer.synthesize(num_rows=30, produce_nans=True)


def test_synthesize_no_tf_values():
    test = {}
    for i in range(0, 100):
        test[i] = {
            "name": str("Test" + str(i)),
            "street": "Post" + str(i),
            "houseno": 32 + i,
            "age": 3 + i
        }
    testdf = pd.DataFrame.from_dict(test, orient="index")
    dfmeta = MetaExtractor.extract(df=testdf)
    synthesizer = HighDimSynthesizer(df_meta=dfmeta)
    synthesizer.learn(testdf, num_iterations=30)
    _ = synthesizer.synthesize(num_rows=30, produce_nans=True)
