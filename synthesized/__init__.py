from .version import __version__
import warnings

warnings.filterwarnings(action='ignore', message='numpy.dtype size changed')


__all__ = ['__version__']
