from .basic_synthesizer import BasicSynthesizer
from .transformations import transformation_modules
from .values import value_modules


class IdSynthesizer(BasicSynthesizer):

    def __init__(self, dtypes, id_embedding_size=128, **kwargs):
        self.id_embedding_size = id_embedding_size

        super().__init__(dtypes=dtypes, **kwargs)

        self.modulation = self.add_module(
            module='modulation', modules=transformation_modules, name='modulation',
            input_size=self.encoding_size, condition_size=id_embedding_size
        )

        if self.identifier_value is None:
            raise NotImplementedError

    def get_value(self, name, dtype):
        if name == 'account_id':
            value = self.add_module(
                module='identifier', modules=value_modules, name=name,
                num_identifiers=4500, embedding_size=self.id_embedding_size
            )
            self.identifier_value = value

        else:
            value = super().get_value(name=name, dtype=dtype)

        return value

    def customized_transform(self, x):
        condition = self.identifier_value.input_tensor()
        return self.modulation.transform(x=x, condition=condition)

    def customized_synthesize(self, x):
        identifier, condition = self.identifier_value.random_value(n=self.num_synthesize)
        return self.modulation.transform(x=x, condition=condition)
