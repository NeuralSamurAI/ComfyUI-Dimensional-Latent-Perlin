"""
NoisyLatentPerlinD for ComfyUI

This implementation is inspired by and based on the work of Extraltodeus:
https://github.com/Extraltodeus/noise_latent_perlinpinpin

Modifications and enhancements have been made to adapt it for broader use cases.
"""
import torch
import math
import numpy as np
import comfy.utils

class NoisyLatentPerlinD:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "width": ("INT", {"default": 1024, "min": 8, "max": 8192, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 8, "max": 8192, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            "detail_level": ("FLOAT", {"default": 0, "min": -1, "max": 1.0, "step": 0.1}),
            "downsample_factor": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            },
            "optional": {
                "latent_image": ("LATENT", ),
                "model": ("MODEL", ),
            }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_noise"
    CATEGORY = "latent/noise"

    def rand_perlin_2d(self, shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        
        grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
        angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
        
        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
        dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
        
        n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

    def rand_perlin_2d_octaves(self, shape, res, octaves=1, persistence=0.5):
        noise = torch.zeros(shape)
        frequency = 1
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * self.rand_perlin_2d(shape, (frequency*res[0], frequency*res[1]))
            frequency *= 2
            amplitude *= persistence
        noise = torch.remainder(torch.abs(noise)*1000000,11)/11
        return noise
    
    def scale_tensor(self, x):
        min_value = x.min()
        max_value = x.max()
        x = (x - min_value) / (max_value - min_value)
        return x

    def generate_noise(self, seed, width, height, batch_size, detail_level, downsample_factor, latent_image=None, model=None):
        torch.manual_seed(seed)
        
        # Determine the number of dimensions (channels)
        if model is not None:
            dimensions = model.get_model_object("latent_format").latent_channels
        elif latent_image is not None:
            dimensions = latent_image["samples"].shape[1]
        else:
            dimensions = 4  # Default to 4 if neither model nor latent_image is provided
        
        noise = torch.zeros((batch_size, dimensions, height // downsample_factor, width // downsample_factor), dtype=torch.float32, device="cpu")
        
        for i in range(batch_size):
            for j in range(dimensions):
                noise_values = self.rand_perlin_2d_octaves((height // downsample_factor, width // downsample_factor), (1,1), 1, 1)
                result = (1+detail_level/10)*torch.erfinv(2 * noise_values - 1) * (2 ** 0.5)
                result = torch.clamp(result,-5,5)
                noise[i, j, :, :] = result
        
        # If a latent_image is provided, ensure the noise matches its shape
        if latent_image is not None:
            noise = self.fix_latent_shape(noise, latent_image["samples"])
        
        return ({"samples": noise},)

    def fix_latent_shape(self, noise, latent_image):
        if noise.shape != latent_image.shape:
            noise = comfy.utils.repeat_to_batch_size(noise, latent_image.shape[1], dim=1)
            noise = noise[:latent_image.shape[0], :, :latent_image.shape[2], :latent_image.shape[3]]
        return noise