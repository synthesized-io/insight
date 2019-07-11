from .generative import Generative
from .vae import VAE
from .vae_master import VAEMaster
from ..module import register


register(name='vae', module=VAE)
register(name='vae_master', module=VAEMaster)

__all__ = ['Generative']
