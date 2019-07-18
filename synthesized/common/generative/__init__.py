from .generative import Generative
from .vae import VAE
from .vae_old import VAEOld
from ..module import register


register(name='vae', module=VAE)
register(name='vae_old', module=VAEOld)

__all__ = ['Generative']
