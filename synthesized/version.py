import textwrap
import wrapt
from deprecated.classic import ClassicAdapter, deprecated as _deprecated

__version__ = '1.0.0'


class DocStringAdapter(ClassicAdapter):
    """"""

    def __init__(self, directive, reason="", version="", action=None, category=DeprecationWarning):
        self.directive = directive
        super().__init__(
            reason=reason, version=version, action=action, category=category
        )

    def __call__(self, wrapped):
        if self.directive == 'version_changed' and self.version == __version__:
            # TODO: Show some alert that the api to this class/function has changed.
            pass

        reason = textwrap.dedent(self.reason).strip()
        reason = '\n'.join(
            textwrap.fill(line, width=70, initial_indent='   ', subsequent_indent='   ') for line in reason.splitlines()
        ).strip()
        docstring = textwrap.dedent(wrapped.__doc__ or "")
        if docstring:
            docstring += "\n\n"

        docstring += "    {directive}: {version}\n".format(
            directive=self.directive.replace('_', ' ').title(),
            version=self.version
        )
        if reason:
            docstring += "        {reason}\n".format(reason=reason)

        wrapped.__doc__ = docstring
        setattr(wrapped, f'_{self.directive}', self.version)
        return super().__call__(wrapped)


def versionadded(version, reason=""):
    adapter = DocStringAdapter('version_added', reason=reason, version=version)

    # noinspection PyUnusedLocal
    @wrapt.decorator(adapter=adapter)
    def wrapper(wrapped, instance, args, kwargs):
        return wrapped(*args, **kwargs)

    return wrapper


def versionchanged(version, reason=""):
    adapter = DocStringAdapter('version_changed', reason=reason, version=version)

    # noinspection PyUnusedLocal
    @wrapt.decorator(adapter=adapter)
    def wrapper(wrapped, instance, args, kwargs):
        return wrapped(*args, **kwargs)

    return wrapper


def deprecated(version, reason=""):
    """
    This decorator can be used to insert a "deprecated" directive in your function/class docstring in order to
    documents the version of the project which deprecates this functionality in your library.

    Args:
        version: Version of your project which deprecates this feature. The version number has the
            format "MAJOR.MINOR.PATCH".
        reason: Reason message which documents the deprecation in your library (can be omitted).

    """
    directive = 'deprecated'
    adapter_cls = DocStringAdapter
    return _deprecated(version=version, reason=reason, directive=directive, adapter_cls=adapter_cls)
