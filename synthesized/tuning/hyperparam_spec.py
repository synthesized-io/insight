from random import choice, random

import simplejson


class HyperparamSpec(object):

    def __init__(self, specification):
        if isinstance(specification, str):
            with open(specification, 'r') as filehandle:
                specification = simplejson.load(fp=filehandle)

        for spec in specification:
            assert spec.get('dimensionality', 1) == 1
            if spec['type'] == 'continuous':
                assert len(spec['domain']) == 2 and spec['domain'][0] < spec['domain'][1]
                assert spec.get('transformations') is None or all(
                    transformation in ('int', 'pow') for transformation in spec['transformations']
                )

            elif spec['type'] == 'discrete':
                assert len(spec['domain']) > 0 and spec['domain'] == sorted(spec['domain'])
                assert spec.get('transformations') is None or all(
                    parameter in spec['transformations'] for parameter in spec['domain'])

            else:
                assert False, 'invalid type ' + spec['type']

        self.specification = specification

    def random(self):
        while True:
            hyperparams = dict()
            for spec in self.specification:

                if spec['type'] == 'continuous':
                    lower, upper = spec['domain']
                    hyperparam = lower + (upper - lower) * random()
                    if spec.get('transformations') is not None:
                        for transformation in spec['transformations']:
                            if transformation == 'pow':
                                hyperparam = 10.0 ** hyperparam
                            elif transformation == 'int':
                                hyperparam = int(hyperparam)

                elif spec['type'] == 'discrete':
                    hyperparam = choice(spec['domain'])
                    if spec.get('transformations') is not None:
                        hyperparam = spec['transformations'][hyperparam]

                else:
                    assert False

                hyperparams[spec['name']] = hyperparam

            yield hyperparams

    def grid(self, num_values):
        if isinstance(num_values, int):
            num_values = {spec['name']: num_values for spec in self.specification}
        elif isinstance(num_values, (tuple, list)):
            assert len(num_values) == len(self.specification)
            num_values = {
                spec['name']: num_value for spec, num_value in zip(self.specification, num_values)
            }

        num_values = dict(num_values)
        name = next(iter(num_values))
        num_value = num_values.pop(name)

        def recursive_grid(specification):
            if len(specification) == 0:
                yield dict()
                return

            spec = specification.pop()
            for n in range(num_value):

                if spec['type'] == 'continuous':
                    lower, upper = spec['domain']
                    parameter = lower + n * (upper - lower) / (num_value - 1)
                    if spec.get('transformations') is not None:
                        for transformation in spec['transformations']:
                            if transformation == 'pow':
                                parameter = 10.0 ** parameter
                            elif transformation == 'int':
                                parameter = int(parameter)

                elif spec['type'] == 'discrete':
                    if n == num_value - 1:
                        parameter = spec['domain'][-1]
                    else:
                        parameter = spec['domain'][n * (len(spec['domain']) // (num_value - 1))]
                    if spec.get('transformations') is not None:
                        parameter = spec['transformations'][parameter]

                for hyperparams in recursive_grid(specification=list(specification)):
                    hyperparams[spec['name']] = parameter
                    yield hyperparams

        for grid in recursive_grid(specification=list(self.specification)):
            yield grid

    def parse(self, hyperparams):
        # hyperparams = np.asarray(hyperparams)
        # assert hyperparams.ndim == 1 or hyperparams.ndim == 2
        # if hyperparams.ndim == 2:
        #     assert hyperparams.shape[0] == 1
        #     hyperparams = hyperparams[0, :]

        for name, spec in self.specification.items():
            parameter = hyperparams[name]

            if spec['type'] == 'continuous':
                assert spec['domain'][0] <= parameter <= spec['domain'][1]
                if spec.get('transformations') is not None:
                    for transformation in spec['transformations']:
                        if transformation == 'pow':
                            parameter = 10.0 ** parameter
                        elif transformation == 'int':
                            parameter = int(parameter)

            elif spec['type'] == 'discrete':
                parameter = int(parameter)
                assert parameter in spec['domain']
                if spec.get('transformations') is not None:
                    parameter = spec['transformations'][parameter]

            else:
                assert False, 'invalid type ' + spec['type']

            hyperparams[spec['name']] = parameter

        return hyperparams
