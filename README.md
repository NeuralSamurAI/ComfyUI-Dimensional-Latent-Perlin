# Dimensional Latent Perlin for ComfyUI

![Node](https://github.com/NeuralSamurAI/ComfyUI-Dimensional-Latent-Perlin/blob/main/assets/Node.png)

## Description

Dimensional Latent Perlin is a custom node for ComfyUI that generates Perlin noise in the latent space. This node is designed to work seamlessly with various diffusion models and can be used as an alternative or complement to standard random noise generators in image generation pipelines.

## Features

- Generate Perlin noise that matches the latent space of diffusion models
- Adjustable parameters for fine-tuning the noise generation
- Compatible with different model architectures and latent space dimensions
- Can adapt to existing latent images or model specifications
- Supports batch processing

## Installation

1. Clone this repository into your ComfyUI custom nodes directory via ComfyUI Manager or:

```git clone https://github.com/neuralsamurai/ComfyUI-Dimensional-Latent-Perlin.git```

2. Restart ComfyUI or reload custom nodes.

## Usage

After installation, the Dimensional Latent Perlin node will be available in the ComfyUI interface under the "latent/noise" category.

### Input Parameters

- `seed`: Seed for random number generation (for reproducibility)
- `width`: Width of the output image
- `height`: Height of the output image
- `batch_size`: Number of images to generate in a batch
- `detail_level`: Controls the level of detail in the Perlin noise
- `downsample_factor`: Downsampling factor for the latent space

### Optional Inputs

- `latent_image`: An existing latent image to match dimensions
- `model`: A diffusion model to match latent space specifications

### Output

- `LATENT`: A latent representation of the generated Perlin noise

## Examples

Workflow available in the asset directory

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

This node was inspired by and based on the work of Extraltodeus:
- Original Author: [Extraltodeus](https://github.com/Extraltodeus)
- Original Repository: [noise_latent_perlinpinpin](https://github.com/Extraltodeus/noise_latent_perlinpinpin)

## Acknowledgements

- Special thanks to Extraltodeus for the original implementation that inspired this node.
- This node builds upon the concept of controllable noise generation in latent diffusion models.
- Thanks to the Image Diffusion, AI Twitter, ComfyUI & Banodoco communities for their support and inspiration.
