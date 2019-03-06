from flask.json import JSONEncoder


# By default NaN is serialized as "NaN". We enforce "null" instead.
class JSONCompliantEncoder(JSONEncoder):
    def __init__(self, *args, **kwargs):
        kwargs["ignore_nan"] = True
        super().__init__(*args, **kwargs)
