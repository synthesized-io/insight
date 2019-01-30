from ..module import Module


class Functional(Module):

    def __init__(self, outputs=None, name=None):
        if name is None:
            name = self.__class__.__name__.lower()
            if name.endswith('functional'):
                name = name[:-10]
        if outputs is not None:
            name = name + '-' + '-'.join(outputs)

        super().__init__(name=name)

        self.outputs = outputs

    def specification(self):
        spec = super().specification()
        spec.update(outputs=self.outputs)
        return spec

    def required_outputs(self):
        return self.outputs

    def tf_loss(self, *samples):
        raise NotImplementedError
