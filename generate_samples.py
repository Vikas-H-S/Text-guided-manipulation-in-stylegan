import os
import yaml
import numpy as np
import torch
from models.stylegan2.models import Generator
from tqdm import tqdm
from pathlib import Path
from manipulation_model import GlobalManipulator
import argparse
from matplotlib import pyplot as plt
from torchvision.utils import save_image

def rescale_image(image, new_min, new_max, target_dtype):
    original_min = image.min()
    original_max = image.max()

    scale_factor = (new_max - new_min) / (original_max - original_min)
    offset = new_max - scale_factor * original_max
    rescaled_image = (scale_factor * image + offset).astype(target_dtype)
    return rescaled_image

def process_single_target(model, target_class, direction_type='single'):
    if direction_type == 'single':
        generated_image, direction_vector, *_ = model.modify_image(target=target_class, output_file=None)
    elif direction_type == 'multi':
        generated_image, direction_vector, *_ = model.modify_image(
            target=target_class, 
            output_file=Path(config['style_space_path']) / 'multi2one.pt'
        )
    else:
        raise NotImplementedError("direction_type must be either 'single' or 'multi'")
    return {
        'modified_image': generated_image.detach().cpu(),
        'direction_vector': direction_vector,
    }

def apply_modifications(target_classes, latent_code):
    settings = {
        'generator_model': generator,
        'latent_code': latent_code,
    }
    settings.update(config)

    manipulator = GlobalManipulator(**settings)
    manipulated_images = []

    for target_class in tqdm(target_classes):
        with torch.no_grad():
            single_direction_output = process_single_target(manipulator, target_class, 'single')
            multi_direction_output = process_single_target(manipulator, target_class, 'multi')
            
            images_set = [
                manipulator.original_image.detach().cpu(), 
                single_direction_output['modified_image'], 
                multi_direction_output['modified_image']
            ]
        concatenated_images = torch.cat(images_set, dim=-1).squeeze(0).permute(1, 2, 0)
        manipulated_images.append(np.asarray(concatenated_images))
    
    save_path = os.path.join('./logs', config['dataset'] + '.png')
    num_rows = len(target_classes) // 2
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(6, 4), constrained_layout=True)
    
    for idx, image in enumerate(manipulated_images):
        column_index = idx % 2
        row_index = idx // 2
        image = rescale_image(image, 0, 255, np.uint8)
        axes[row_index, column_index].imshow(image)
        axes[row_index, column_index].set_title(
            'Original | StyleCLIP | Multi2One', fontsize=7, pad=1.5
        )
        axes[row_index, column_index].set_xlabel(target_classes[idx], fontsize=9)
        axes[row_index, column_index].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    plt.savefig(save_path, dpi=1000)
    print(f"Manipulated image saved at {save_path}")
    plt.cla()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration for StyleCLIP Global Manipulation Method')
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="ffhq", 
        choices=['ffhq', 'afhqcat', 'afhqdog', 'car', 'church'], 
        help="Specify the StyleGAN pretrained dataset"
    )
    args = parser.parse_args()

    if not os.path.exists('Pretrained'):
        import gdown
        zip_url = 'https://drive.google.com/uc?export=download&id=1wkCJxUf7YD-ov70DCZaN8GRUoKsKZJOs'
        downloaded_zip = 'multi2one.zip'
        gdown.download(zip_url, downloaded_zip, quiet=False)
        os.system(f'unzip {downloaded_zip}; rm {downloaded_zip}; mv tmp/* .; rm -r tmp')

    with open('config.yaml', "r") as file:
        config = yaml.safe_load(file)
        config = config[f"{args.dataset}_params"]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    generator = Generator(
        size=config['stylegan_size'],
        style_dim=512,
        n_mlp=8,
        channel_multiplier=2,
    )
    generator.load_state_dict(torch.load(config['stylegan_ckpt'], map_location='cpu')['g_ema'])
    generator.eval()
    generator.to(device)
   
    latent_file = f'dictionary/{args.dataset}/latent.pt'
    if os.path.exists(latent_file):
        latent_code = torch.load(latent_file, map_location='cpu').to(device)
    else:
        mean_latent_vector = generator.mean_latent(4096)
        initial_latent = torch.randn(1, 512).cuda()
        with torch.no_grad():
            _, latent_code = generator(
                [initial_latent], 
                return_latents=True, 
                truncation=0.7, 
                truncation_latent=mean_latent_vector
            )
    apply_modifications(config['targets'], latent_code)
