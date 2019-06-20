from .generative import Generative
from .vae import VAE
from ..module import register


register(name='vae', module=VAE)


__all__ = ['Generative']
