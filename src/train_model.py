# src/train_model.py
import os
import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.utils import save_image


class ImageStyleTransfer_VGG19(nn.Module):
    def __init__(self):
        super(ImageStyleTransfer_VGG19, self).__init__()
        self.chosen_features = {0: 'conv11', 5: 'conv21', 10: 'conv31', 19: 'conv41', 28: 'conv51'}
        self.model = vgg19(weights='DEFAULT').features[:29]

    def forward(self, x):
        feature_maps = dict()
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx in self.chosen_features.keys():
                feature_maps[self.chosen_features[idx]] = x
        return feature_maps


def _get_content_loss(content_feature, generated_feature):
    return torch.mean((generated_feature - content_feature) ** 2)


def _get_style_loss(style_feature, generated_feature):
    _, channel, height, width = generated_feature.shape
    # Ensure features are float for mm
    style_feature_float = style_feature.float()
    generated_feature_float = generated_feature.float()

    style_gram = style_feature_float.view(channel, height*width).mm(
        style_feature_float.view(channel, height*width).t()
    )
    generated_gram = generated_feature_float.view(channel, height*width).mm(
        generated_feature_float.view(channel, height*width).t()
    )
    return torch.mean((generated_gram - style_gram) ** 2)


def train_image(content, style, generated, device, train_config, output_dir, output_img_fmt, 
                content_img_name_original, style_img_name_original, verbose=False):
    model = ImageStyleTransfer_VGG19().to(device).eval()

    num_epochs = train_config.get('num_epochs', 6000)
    lr = train_config.get('learning_rate', 0.001)
    alpha = train_config.get('alpha', 1.0)
    beta = train_config.get('beta', 0.01) # Note: Flask form might send a different default like 100000

    default_layers = {'conv11', 'conv21', 'conv31', 'conv41', 'conv51'}
    capture_content_features_from = train_config.get('capture_content_features_from', default_layers)
    capture_style_features_from = train_config.get('capture_style_features_from', default_layers)
            
    def normalize_feature_set(feature_input, default_set_val): # Renamed default_set to avoid conflict
        if isinstance(feature_input, str):
            return set([item.strip() for item in feature_input.split(',')])
        elif isinstance(feature_input, list):
            return set(feature_input)
        elif isinstance(feature_input, dict):
             return set(feature_input.keys())
        elif isinstance(feature_input, set):
            return feature_input
        else:
            print(f"Warning: Invalid type for feature selection ({type(feature_input)}), using default.")
            return default_set_val

    capture_content_features_from = normalize_feature_set(capture_content_features_from, default_layers)
    capture_style_features_from = normalize_feature_set(capture_style_features_from, default_layers)

    valid_vgg_layers = {'conv11', 'conv21', 'conv31', 'conv41', 'conv51'}
    if not capture_content_features_from.issubset(valid_vgg_layers):
        print(f"ERROR: Invalid layer in 'capture_content_features_from': {capture_content_features_from - valid_vgg_layers}.")
        return 0 # Indicates failure
    if not capture_style_features_from.issubset(valid_vgg_layers):
        print(f"ERROR: Invalid layer in 'capture_style_features_from': {capture_style_features_from - valid_vgg_layers}.")
        return 0 # Indicates failure

    optimizer = torch.optim.Adam([generated], lr=lr)

    simple_intermediate_folder_name = "intermediate_steps"
    simple_intermediate_file_prefix = "step"

    # Initialize verbose_save_intermediate and intermediate_dir
    verbose_save_intermediate = False 
    intermediate_dir = "" 

    if verbose:
        intermediate_dir = os.path.join(output_dir, simple_intermediate_folder_name)
        if not os.path.exists(intermediate_dir):
            try:
                os.makedirs(intermediate_dir)
                print(f"Created intermediate directory: {intermediate_dir}")
                verbose_save_intermediate = True 
            except OSError as e:
                print(f"Error creating intermediate directory {intermediate_dir}: {e}")
                # verbose_save_intermediate remains False
        else: 
            verbose_save_intermediate = True
    
    print(f"Starting training for {num_epochs} epochs. Alpha: {alpha}, Beta: {beta}, LR: {lr}")
    print(f"Content layers: {capture_content_features_from}")
    print(f"Style layers: {capture_style_features_from}")

    for epoch in range(num_epochs):
        content_features = model(content)
        style_features = model(style)
        generated_features = model(generated)
        
        current_content_loss_tensor = torch.tensor(0.0, device=device) # No requires_grad needed here
        current_style_loss_tensor = torch.tensor(0.0, device=device)   # No requires_grad needed here

        for layer_name in generated_features.keys():
            if layer_name in capture_content_features_from:
                current_content_loss_tensor = current_content_loss_tensor + _get_content_loss(content_features[layer_name], generated_features[layer_name])
            if layer_name in capture_style_features_from:
                current_style_loss_tensor = current_style_loss_tensor + _get_style_loss(style_features[layer_name], generated_features[layer_name])
        
        total_loss = alpha * current_content_loss_tensor + beta * current_style_loss_tensor

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if verbose: 
            if (epoch + 1) % 200 == 0:
                # Use .item() for logging scalar loss values
                print(f"\tEpoch {epoch + 1}/{num_epochs}, Content Loss: {alpha*current_content_loss_tensor.item():.4f}, Style Loss: {beta*current_style_loss_tensor.item():.4f}, Total Loss: {total_loss.item():.4f}")
                if verbose_save_intermediate: 
                    intermediate_filename = f'{simple_intermediate_file_prefix}-{epoch + 1}.{output_img_fmt}'
                    try:
                        save_image(generated.detach(), os.path.join(intermediate_dir, intermediate_filename))
                    except Exception as e_save:
                        print(f"Error saving intermediate image {intermediate_filename}: {e_save}")
    
    if verbose: 
        print("\t================================")
        if verbose_save_intermediate and intermediate_dir and os.path.exists(intermediate_dir):
             print(f"\tIntermediate images are saved in directory: '{intermediate_dir}'")
        else:
            print(f"\tIntermediate image saving was disabled or directory was not created/found.")
        print("\t================================")

    return 1 # Indicates success


# Keep train_frame if you plan to use it, otherwise it can be removed if not needed.
def train_frame(content, style, generated, device, output_img_fmt):
    """Update the output image using pre-trained VGG19 model for video transfer."""
    model = ImageStyleTransfer_VGG19().to(device).eval()    # freeze parameters in the model

    # set default value for each configuration
    num_epochs = 2000
    lr = 0.01
    alpha_vf = 50 # Renamed to avoid conflict if this func is called within same scope as train_image one day
    beta_vf = 0.001
    capture_content_features_from_vf = {'conv11', 'conv21', 'conv31', 'conv41', 'conv51'}
    capture_style_features_from_vf = {'conv11', 'conv21', 'conv31', 'conv41', 'conv51'}

    optimizer = torch.optim.Adam([generated], lr=lr)

    for epoch in range(num_epochs):
        content_features = model(content)
        style_features = model(style)
        generated_features = model(generated)

        content_loss_tensor_vf = torch.tensor(0.0, device=device)
        style_loss_tensor_vf = torch.tensor(0.0, device=device)

        for layer_name in generated_features.keys():
            if layer_name in capture_content_features_from_vf:
                content_loss_tensor_vf += _get_content_loss(content_features[layer_name], generated_features[layer_name])
            if layer_name in capture_style_features_from_vf:
                style_loss_tensor_vf += _get_style_loss(style_features[layer_name], generated_features[layer_name])
        
        total_loss = alpha_vf * content_loss_tensor_vf + beta_vf * style_loss_tensor_vf

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return 1