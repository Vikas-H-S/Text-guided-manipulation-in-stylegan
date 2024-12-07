import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F

def styled_convolution(layer, input_tensor, mod_vector, noise_level):
    convolution_layer = layer.conv
    batch_size, channels, height, width = input_tensor.shape
    mod_vector = mod_vector.view(batch_size, 1, channels, 1, 1)
    weights = convolution_layer.scale * convolution_layer.weight * mod_vector

    if convolution_layer.demodulate:
        normalization = torch.rsqrt(weights.pow(2).sum([2, 3, 4]) + 1e-8)
        weights *= normalization.view(batch_size, convolution_layer.out_channel, 1, 1, 1)

    weights = weights.view(
        batch_size * convolution_layer.out_channel, channels, convolution_layer.kernel_size, convolution_layer.kernel_size
    )

    if convolution_layer.upsample:
        input_tensor = input_tensor.view(1, batch_size * channels, height, width)
        weights = weights.view(
            batch_size, convolution_layer.out_channel, channels, convolution_layer.kernel_size, convolution_layer.kernel_size
        )
        weights = weights.transpose(1, 2).reshape(
            batch_size * channels, convolution_layer.out_channel, convolution_layer.kernel_size, convolution_layer.kernel_size
        )
        result = F.conv_transpose2d(input_tensor, weights, padding=0, stride=2, groups=batch_size)
        _, _, height, width = result.shape
        result = result.view(batch_size, convolution_layer.out_channel, height, width)
        result = convolution_layer.blur(result)

    elif convolution_layer.downsample:
        input_tensor = convolution_layer.blur(input_tensor)
        _, _, height, width = input_tensor.shape
        input_tensor = input_tensor.view(1, batch_size * channels, height, width)
        result = F.conv2d(input_tensor, weights, padding=0, stride=2, groups=batch_size)
        _, _, height, width = result.shape
        result = result.view(batch_size, convolution_layer.out_channel, height, width)

    else:
        input_tensor = input_tensor.view(1, batch_size * channels, height, width)
        result = F.conv2d(input_tensor, weights, padding=convolution_layer.padding, groups=batch_size)
        _, _, height, width = result.shape
        result = result.view(batch_size, convolution_layer.out_channel, height, width)
    result = layer.noise(result, noise=noise_level)
    result = layer.activate(result)
    
    return result

def reconstruct_image(generator, styled_vectors, latent_code, noise_vectors, dataset_name):
    generated = generator.input(latent_code)

    generated = styled_convolution(generator.conv1, generated, styled_vectors[0], noise_vectors[0])
    skip_connection = generator.to_rgb1(generated, latent_code[:, 0])

    layer_index, rgb_index = 2, 1
    for conv1, conv2, noise1, noise2, rgb_conversion in zip(
        generator.convs[::2], generator.convs[1::2], noise_vectors[1::2], noise_vectors[2::2], generator.to_rgbs
    ):
        generated = styled_convolution(conv1, generated, styled_vectors[layer_index], noise=noise1)
        generated = styled_convolution(conv2, generated, styled_vectors[layer_index + 1], noise=noise2)
        skip_connection = rgb_conversion(generated, latent_code[:, rgb_index + 2], skip_connection)

        layer_index += 3
        rgb_index += 2

    final_image = skip_connection
    return final_image

def extract_style_space(generator, latent_vector):
    noise_list = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
    styled_vectors = []
    style_labels = []
    styled_vectors.append(generator.conv1.conv.modulation(latent_vector[:, 0]))
    style_labels.append(f"block4/conv1")
    styled_vectors.append(generator.to_rgbs[0].conv.modulation(latent_vector[:, 0]))
    style_labels.append(f"block4/torgb")

    latent_index, block_size = 1, 3
    for conv1, conv2, noise1, noise2, rgb_conversion in zip(
        generator.convs[::2], generator.convs[1::2], noise_list[1::2], noise_list[2::2], generator.to_rgbs
    ):
        resolution = 2 ** block_size
        styled_vectors.append(conv1.conv.modulation(latent_vector[:, latent_index]))
        style_labels.append(f"block{resolution}/conv1")
        styled_vectors.append(conv2.conv.modulation(latent_vector[:, latent_index + 1]))
        style_labels.append(f"block{resolution}/conv2")
        styled_vectors.append(rgb_conversion.conv.modulation(latent_vector[:, latent_index + 2]))
        style_labels.append(f"block{resolution}/torgb")

        latent_index += 2
        block_size += 1
        
    return styled_vectors, style_labels, noise_list

def divide_style_space(style_vector, styled_vectors, style_labels, boundary=None):
    separated_styles = []
    start_idx = 0
    label_idx = 0
    
    for label, vector in zip(style_labels, styled_vectors):
        channel_count = vector.shape[-1]
        if boundary is not None:
            total_channels = boundary.shape[0]
            if "torgb" not in label and label_idx < total_channels:
                separated_styles.append(style_vector[:, start_idx: start_idx + channel_count])
                label_idx += channel_count
            else:
                separated_styles.append(torch.zeros_like(style_vector[:, start_idx: start_idx + channel_count]))
        else:
            separated_styles.append(style_vector[:, start_idx: start_idx + channel_count])
        start_idx += channel_count
    return separated_styles
