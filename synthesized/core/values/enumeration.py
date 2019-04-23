from .value import Value


class EnumerationValue(Value):

    def preprocess(self, data):
        data = data.drop(labels=self.name, axis=1)
        return data

    def postprocess(self, data):
        data.loc[:, self.name] = data.index + 1
        return data
