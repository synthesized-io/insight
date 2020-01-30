import tensorflow as tf

from ..module import tensorflow_name_scoped


class Functional(tf.Module):

    def __init__(self, values, name=None):
        if name is None:
            name = self.__class__.__name__.lower()
            if name.endswith('functional'):
                name = name[:-10]
        if values != '*':
            name = name + '-' + '-'.join(values)

        super().__init__(name=name)

        self.values = values

    def specification(self):
        spec = super().specification()
        spec.update(values=self.values)
        return spec

    def required_outputs(self):
        return self.values

    @tensorflow_name_scoped
    def loss(self, *samples_args):
        raise NotImplementedError

    def check_distance(self, *samples_args):
        raise NotImplementedError
