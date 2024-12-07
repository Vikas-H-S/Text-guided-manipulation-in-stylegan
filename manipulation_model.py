import os
import sys
from pathlib import Path
module_directory = Path(__file__).parent
sys.path.append(str(module_directory.resolve()))
import copy
import pickle
import numpy as np
from PIL import Image
import torch
from utils import feature_encoder, feature_decoder
from criteria.clip_loss import TextImageLoss

class DirectionManipulator(TextImageLoss):
    def __init__(self, **parameters):
        super().__init__()
        allowed_keys = ['device', 'manipulation_strength', 'selection_count', 'generator_model', 'initial_latent',
                        'dataset_name', 'style_data_path']
        for key in parameters.keys():
            if key in allowed_keys:
                setattr(self, key, parameters[key])

        with torch.no_grad():
            self.styles, self.style_labels, self.noise_vars = feature_encoder(self.generator_model, self.initial_latent)
            self.original_image = feature_decoder(
                self.generator_model,
                self.styles,
                self.initial_latent,
                self.noise_vars,
                self.dataset_name
            )

    def convert_tensor_to_clip(self, image_tensor):
        array_img = image_tensor.detach().cpu().numpy()
        array_img = np.transpose(array_img, [0, 2, 3, 1])
        drange = [-1, 1]
        scale_factor = 255 / (drange[1] - drange[0])
        array_img = array_img * scale_factor + (0.5 - drange[0] * scale_factor)
        np.clip(array_img, 0, 255, out=array_img)
        array_img = array_img.astype('uint8')
        pil_image = Image.fromarray(array_img.squeeze(0))
        processed_image = self.preprocess(pil_image).unsqueeze(0)
        return self.encode_image(processed_image.to(self.device))

    def compute_boundary(self, boundary_file=None):
        if boundary_file is not None:
            self.vector_dict = torch.load(boundary_file).to(self.device)
        else:
            self.vector_dict = torch.Tensor(np.load(os.path.join(self.style_data_path, 'direction.npy'))).to(self.device)

        similarity = np.dot(self.vector_dict.detach().cpu().numpy(), self.text_direction.squeeze(0).detach().cpu().numpy())
        if self.selection_count is None:
            direction_scores = copy.copy(similarity)
            mask = np.abs(similarity) < self.threshold
            count_selected = np.sum(~mask)
            direction_scores[mask] = 0
        else:
            count_selected = self.selection_count
            _, selected_indices = torch.topk(torch.Tensor(np.abs(similarity)), k=self.selection_count, dim=0)
            direction_scores = np.zeros_like(similarity)
            for index in selected_indices:
                index = index.detach().cpu()
                direction_scores[index] = similarity[index]

        max_abs_val = np.abs(direction_scores).max()
        direction_scores /= max_abs_val  # Normalize to range [-1, 1]
        return direction_scores, count_selected, selected_indices

    def partition_styles(self, direction_scores):
        style_directions = []
        start_idx = 0
        dataset_path = f"stylespace/{self.dataset_name}/statistics/"

        stats_path = dataset_path + 'style_stats'
        with open(stats_path, "rb") as file:
            _, latent_codes = pickle.load(file)

        mean_std_path = dataset_path + 'style_mean_std'
        with open(mean_std_path, "rb") as file:
            _, standard_deviations = pickle.load(file)

        for idx, label in enumerate(self.style_labels):
            if "torgb" not in label:
                count = self.styles[idx].shape[1]
                end_idx = start_idx + count
                subset = direction_scores[start_idx:end_idx]
                style_directions.append(subset)
                start_idx = end_idx

        adjusted_directions = []
        cumulative_index = 0
        for idx, label in enumerate(self.style_labels):
            if ("torgb" not in label) and len(style_directions[cumulative_index]) != 0:
                adjusted_subset = style_directions[cumulative_index] * standard_deviations[idx]
                adjusted_directions.append(adjusted_subset)
                cumulative_index += 1
            else:
                adjusted_subset = np.zeros(len(latent_codes[idx][0]))
                adjusted_directions.append(adjusted_subset)
        del style_directions
        return adjusted_directions, latent_codes

    def apply_manipulation(self, latent_space, style_directions):
        """
        Manipulate style space using calculated directions.
        """
        alpha_values = [self.manipulation_strength]
        steps = len(alpha_values)
        reshaped_latents = [latent.reshape((1, -1)) for latent in latent_space]
        repeated_latents = [np.tile(latent[:, None], (1, steps, 1)) for latent in reshaped_latents]

        alpha_array = np.array(alpha_values).reshape([steps if dim == 1 else 1 for dim in repeated_latents[0].ndim])

        for idx in range(len(style_directions)):
            repeated_latents[idx] += alpha_array * style_directions[idx]

        final_codes = []
        for idx in range(len(repeated_latents)):
            reshaped_code = repeated_latents[idx].reshape([-1])
            code_tensor = torch.Tensor(reshaped_code).cuda()
            final_codes.append(code_tensor)
        return final_codes

    def generate_image(self, input_target, boundary_file=None, predefined_dir=None):
        if isinstance(input_target, str):
            self.text_direction = self.encode_text(input_target)
        elif input_target.dim() > 2:
            self.text_direction = self.encode_image(input_target)
        else:
            self.text_direction = input_target

        delta_styles, _, _ = self.compute_boundary(boundary_file=boundary_file)
        standardized_deltas, latent_codes = self.partition_styles(delta_styles)
        modified_styles = self.apply_manipulation(latent_codes, standardized_deltas)

        if predefined_dir is not None:
            idx = 0
            predefined_dir *= self.manipulation_strength

            manipulated_latents = []
            for manipulation in modified_styles:
                manipulated_latents.append(predefined_dir[:, idx: idx + manipulation.shape[-1]].unsqueeze(1))
                idx += manipulation.shape[-1]

            modified_styles = []
            for latent, manipulation in zip(latent_codes, manipulated_latents):
                modified_styles.append(torch.Tensor(latent.reshape(1, 1, -1)).to(self.device) + 2 * manipulation)

        output_image = feature_decoder(
            generator=self.generator_model,
            style_space=modified_styles,
            latent=self.initial_latent,
            noise=self.noise_vars,
            dataset=self.dataset_name
        )
        return output_image, delta_styles
