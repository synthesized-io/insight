__version__ = '1.0.0'


class DocStringDecorator:
    """"""
    def __init__(self, version: str, directive: str):
        self.version = version
        self.directive = directive

    def __call__(self, func):
        docstring = func.__doc__ or ""

        if docstring:
            docstring += "\n"

        docstring += "    {directive}: {version}\n".format(
            directive=self.directive.replace('_', ' ').title(),
            version=self.version
        )

        func.__doc__ = docstring
        setattr(func, f'_{self.directive}', self.version)

        return func


class versionadded(DocStringDecorator):
    def __init__(self, version: str):
        super().__init__(version, directive='version_added')


class versionchanged(DocStringDecorator):
    def __init__(self, version: str):
        super().__init__(version, directive='version_changed')


class deprecated(DocStringDecorator):
    def __init__(self, version: str):
        super().__init__(version, directive='deprecated')
