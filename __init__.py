from .dimensional_latent_perlin import NoisyLatentPerlinD

NODE_CLASS_MAPPINGS = {
    "NoisyLatentPerlinD": NoisyLatentPerlinD
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoisyLatentPerlinD": "Noisy Latent Perlin Dimensional"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']