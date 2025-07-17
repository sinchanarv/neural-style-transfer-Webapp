# image_style_transfer.py
import argparse
import os
import sys
import PIL
import yaml # Keep for now, though we might not use the YAML file loading directly from Flask
import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image

from src.process_image import load_image, get_image_name_ext
from src.train_model import train_image # Assuming train_image returns success and details

# MODIFICATION 1: Change the main function to accept a direct train_config dictionary
def image_style_transfer(config, train_config_direct=None): # Added train_config_direct
    """Implements neural style transfer on a content image using a style image, applying provided configuration."""
    # ... (keep existing path handling logic from original image_style_transfer function) ...
    if config.get('image_dir') is not None:
        # This block might not be used if we always provide direct filepaths from Flask
        image_dir = config.get('image_dir')
        content_path = os.path.join(image_dir, config.get('content_filename'))
        style_path = os.path.join(image_dir, config.get('style_filename'))
        output_dir = config.get('output_dir') if config.get('output_dir') is not None else image_dir
    else:
        # This block WILL be used from Flask
        output_dir = config.get('output_dir')
        content_path = config.get('content_filepath') # Make sure this key matches what Flask sends
        style_path = config.get('style_filepath') # Make sure this key matches what Flask sends

    verbose = not config.get('quiet', False) # Default quiet to False

    if verbose:
        print("Loading content and style images...")
    
    try:
        content_img = Image.open(content_path)
    except FileNotFoundError:
        msg = f"ERROR: could not find such file: '{content_path}'."
        print(msg)
        return {"success": False, "error": msg} # MODIFICATION 2: Return status
    except PIL.UnidentifiedImageError:
        msg = f"ERROR: could not identify image file: '{content_path}'."
        print(msg)
        return {"success": False, "error": msg}

    try:
        style_img = Image.open(style_path)
    except FileNotFoundError:
        msg = f"ERROR: could not find such file: '{style_path}'."
        print(msg)
        return {"success": False, "error": msg}
    except PIL.UnidentifiedImageError:
        msg = f"ERROR: could not identify image file: '{style_path}'."
        print(msg)
        return {"success": False, "error": msg}
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_size = config.get('output_image_size')
    if output_size is not None:
        if len(output_size) > 1: 
            output_size = tuple(output_size)
        else:
            output_size = output_size[0]

    content_tensor = load_image(content_path, device, output_size=output_size)
    # The style image should be resized to match the content_tensor dimensions
    actual_output_dims_for_style = (content_tensor.shape[2], content_tensor.shape[3])
    style_tensor = load_image(style_path, device, output_size=actual_output_dims_for_style)


    if verbose:
        print("Content and style images successfully loaded.")
        print()
        print("Initializing output image...")

    generated_tensor = content_tensor.clone().requires_grad_(True)

    if verbose:
        print("Output image successfully initialized.")
        print()

    # MODIFICATION 3: Use train_config_direct if provided, else load from YAML
    train_config_to_use = {} # This will be passed to train_image
    if train_config_direct is not None:
        if verbose: print("Using direct training configuration parameters.")
        train_config_to_use = train_config_direct
    elif (train_config_path := config.get('train_config_path')) is not None:
        if verbose: print("Loading training configuration file...")
        try:
            with open(train_config_path, 'r') as f:
                train_config_to_use = yaml.safe_load(f)
        except FileNotFoundError:
            msg = f"ERROR: could not find such file: '{train_config_path}'."
            print(msg)
            return {"success": False, "error": msg}
        except yaml.YAMLError:
            msg = f"ERROR: fail to load yaml file: '{train_config_path}'."
            print(msg)
            return {"success": False, "error": msg}
        if verbose: print("Training configuration file successfully loaded.")
    else:
        if verbose: print("No training configuration provided, using defaults in train_image.")
        # train_config_to_use will remain empty, train_image will use its internal defaults
    
    if verbose: print()
    if verbose: print("Training...")
    
    content_img_name, content_img_fmt = get_image_name_ext(content_path)
    style_img_name, _ = get_image_name_ext(style_path)

    output_img_fmt = config.get('output_image_format', 'png') # Default to png
    if output_img_fmt == 'same':
        output_img_fmt = content_img_fmt

    # train model
    # Pass the resolved train_config_to_use
    success_train = train_image(content_tensor, style_tensor, generated_tensor, device, train_config_to_use, output_dir, output_img_fmt, content_img_name, style_img_name, verbose=verbose)

    final_output_filename = f'nst-{content_img_name}-{style_img_name}-final.{output_img_fmt}'
    final_output_path_on_server = os.path.join(output_dir, final_output_filename)

    if success_train:
        save_image(generated_tensor, final_output_path_on_server)
        if verbose:
            print(f"Output image successfully generated as {final_output_path_on_server}.")
        # MODIFICATION 4: Return success and path details
        return {
            "success": True, 
            "output_path": final_output_path_on_server,
            "content_name": content_img_name,
            "style_name": style_img_name
        }
    else:
        # train_image should ideally return more info on failure
        msg = "Training process within train_image failed."
        print(msg)
        return {"success": False, "error": msg}


# Keep the original main() for command-line usage if you still want that
def main():
    """Entry point of the program for command line."""
    parser = argparse.ArgumentParser()
    # ... (keep all your argparse definitions) ...
    parser.add_argument("--image_dir", type=str, help="Path to the directory where content image and style image are stored.")
    parser.add_argument("--content_filename", type=str, default="content.jpg", help="File name of the content image in image_dir. Will use \"content.jpg\" if not provided.")
    parser.add_argument("--style_filename", type=str, default="style.jpg", help="File name of the style image in image_dir. Will use \"style.jpg\" if not provided.")
    
    # Conditional requirement logic needs careful handling for argparse if mixing CLI and programmatic use
    # For Flask, we will always provide content_filepath and style_filepath
    required_if_no_imagedir = "--image_dir" not in sys.argv and not os.environ.get("FLASK_RUNNING") # A way to detect if run by Flask
    
    parser.add_argument("--content_filepath", required=required_if_no_imagedir, type=str, help="Path to the content image if image_dir not provided.")
    parser.add_argument("--style_filepath", required=required_if_no_imagedir, type=str, help="Path to the style image if image_dir not provided.")
    parser.add_argument("--output_dir", required=required_if_no_imagedir, type=str, help="Directory that stores the output image. Will be the same as image_dir if not provided while image_dir provided.")
    
    parser.add_argument("--output_image_size", nargs="+", type=int, help="Size of the output image. Either one integer or two integers separated by space is accepted. Will use the dimensions of content image if not provided.")
    parser.add_minute("--output_image_format", choices=["jpg", "png", "jpeg", "same"], default="jpg", help="Format of the output image. Can be either \"jpg\", \"png\", \"jpeg\", or \"same\". If \"same\", output image will have the same format as the content image. \"jpg\" will be the default format.")
    parser.add_argument("--train_config_path", type=str, help="Path to training configuration file in .yaml format. May include: num_epochs, learning_rate, alpha, beta, capture_content_features_from, capture_style_features_from.")
    parser.add_argument("--quiet", type=bool, default=False, help="True stops showing debugging messages, loss function values during training process, and stops generating intermediate images.")


    args = parser.parse_args()
    config = dict()
    for arg_name in vars(args):
        config[arg_name] = getattr(args, arg_name)
    
    # For CLI, train_config_direct will be None
    result_details = image_style_transfer(config, train_config_direct=None)
    if not result_details["success"]:
        print(f"Script failed: {result_details.get('error', 'Unknown error')}")


if __name__ == '__main__':
    # A way to tell the script it's not run by Flask for argparse 'required' logic
    # This is a bit hacky, better ways exist for complex apps
    os.environ["FLASK_RUNNING"] = "false" 
    main()