import os
import sys
import numpy as np
np.set_printoptions(suppress=True, threshold=sys.maxsize)
import argparse
from tqdm import tqdm
from glob import glob
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from utils import encode_features
from models.stylegan2.models import Generator

class StyleClipLoss(nn.Module):
    def __init__(self, config, single_style_dir=None):
        super(StyleClipLoss, self).__init__()
        if config.dataset == 'ffhq':
            generator_size = 1024
        elif config.dataset != 'church':
            generator_size = 512
        else:
            generator_size = 256

        gen_model = Generator(
            size=generator_size,
            style_dim=512,
            n_mlp=8,
            channel_multiplier=2,
        )
        gen_model.load_state_dict(torch.load(os.path.join(config.model_path, f"stylegan2/{config.dataset}.pt"), map_location='cpu')['g_ema'])
        gen_model.eval()
        self.generator = gen_model.to(config.device)
        self.latent_vectors = torch.load(config.latent_vectors_path, map_location='cpu').to(config.device)
        self.config = config
        
        self.style_space, self.style_layer_names, self.noise_factors = encode_features(
            generator=self.generator,
            latent=self.latent_vectors
        )
        self.mse_loss = nn.MSELoss()
        self.single_style_dir = single_style_dir
        
    def forward(self, deltas, dictionary, multi_style_dir):
        deltas_adjusted = self.exclude_rgb_layers(deltas, dictionary)
        dictionary_adjusted = self.exclude_rgb_layers(dictionary, dictionary)
        reconstruction_loss = self.mse_loss(torch.matmul(deltas_adjusted, dictionary_adjusted), multi_style_dir)

        alpha_weights = multi_style_dir @ dictionary_adjusted.T
        alpha_weights = alpha_weights / (alpha_weights + 1e-6).abs().max(dim=-1, keepdim=True)[0]
        deltas_normalized = deltas_adjusted / deltas_adjusted.abs().max(dim=-1, keepdim=True)[0]
        alpha_loss = self.mse_loss(deltas_normalized, alpha_weights)

        return reconstruction_loss, alpha_loss
    
    def exclude_rgb_layers(self, input_tensor, reference_tensor):
        filtered_tensor = []
        tensor_index, style_index = 0, 0
        for name, style in zip(self.style_layer_names, self.style_space):
            num_channels = style.shape[-1]
            if reference_tensor is not None:
                total_channels = reference_tensor.shape[0]
                if ("torgb" not in name) and tensor_index < total_channels:
                    filtered_tensor.append(input_tensor[:, style_index: style_index + num_channels])
                    tensor_index += num_channels
            style_index += num_channels
        filtered_tensor = torch.cat(filtered_tensor, dim=-1).to(self.config.device)
        return filtered_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_path', type=str, default='./Pretrained')
    parser.add_argument('--dataset', type=str, default='ffhq')
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--alpha_lambda', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='adadelta')
    parser.add_argument('--learning_rate', type=float, default=3.0)
    parser.add_argument('--alpha_scaling', type=int, default=15)
    parser.add_argument('--use_ours', action='store_true')
    config = parser.parse_args()
    config.latent_vectors_path = f'dictionary/{config.dataset}/latent.pt'
    style_directions = []
    delta_values = []
    channel_numbers = []
    style_files = glob(f'precomputed_pairs/{config.dataset}/clip/alpha{config.alpha_scaling}ch*.npy')
    for file in style_files:
        channels = file.split('/')[-1].split('ch')[-1][:-4]
        if int(channels) in [10, 30, 50]:
            channel_numbers.append(channels)
        else:
            continue
        tensor_data = torch.Tensor(np.load(file))
        style_directions.append(tensor_data)
        delta_tensor = torch.load(f'precomputed_pairs/{config.dataset}/unsup_dir/ch{channels}.pt')
        delta_values.append(delta_tensor)
        assert tensor_data.shape[0] == delta_tensor.shape[0], f'{channels}: mismatch {tensor_data.shape} vs {delta_tensor.shape}'
    style_directions = torch.cat(style_directions).to(config.device)
    delta_values = torch.cat(delta_values).to(config.device)
    feature_space = torch.Tensor(np.load(f'dictionary/{config.dataset}/fs3.npy').astype(np.float32)).to(config.device)
    dictionary_matrix = torch.zeros(feature_space.shape).to(config.device)
    dictionary_matrix.requires_grad_(True)
    loss_module = StyleClipLoss(config, single_style_dir=None)
    optimizer = optim.Adadelta([dictionary_matrix], lr=config.learning_rate)
    progress_bar = tqdm(range(config.num_epochs))
    best_recorded_loss = np.inf
    
    for epoch in progress_bar:
        recon_loss, alpha_loss = loss_module(config.alpha_scaling * delta_values, dictionary_matrix, style_directions)
        total_loss = recon_loss + config.alpha_lambda * alpha_loss
        progress_bar.set_description(desc=f"Reconstruction: {recon_loss:.3e} | Alpha: {alpha_loss:.3e}")
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        if (epoch) % 1000 == 0 and best_recorded_loss > total_loss.item():
            best_recorded_loss = total_loss.item()
            save_path = f'dictionary/{config.dataset}/final_dictionary.pt'
            torch.save(dictionary_matrix.detach().cpu(), save_path)
    print(f"New dictionary saved at {save_path}")
