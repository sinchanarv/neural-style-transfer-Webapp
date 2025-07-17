import os
from PIL import Image
from torchvision import transforms

def load_image(image_path, device, output_size=None):
    """Loads an image by transforming it into a tensor."""
    img = Image.open(image_path)

    # Convert image to RGB (3 channels) if it has an Alpha channel (RGBA) or is Grayscale (L)
    if img.mode == 'RGBA' or img.mode == 'LA' or img.mode == 'P': # P is for palettized images that might have transparency
        img = img.convert('RGB')
    elif img.mode == 'L': # Grayscale
        img = img.convert('RGB') # Convert grayscale to RGB by duplicating the channel

    output_dim = None
    if output_size is None:
        output_dim = (img.size[1], img.size[0]) # (height, width)
    elif isinstance(output_size, int):
        # If one int is given, make it square, but respect aspect ratio by using it as the shorter side for Resize
        # However, transforms.Resize expects (h, w) or a single int for the shorter side.
        # For simplicity, let's assume if one int is given, it's for a square image or shorter side.
        # The original script logic makes it square if one int.
        output_dim = (output_size, output_size) if img.size[0] == img.size[1] else output_size

    elif isinstance(output_size, tuple):
        if (len(output_size) == 2) and isinstance(output_size[0], int) and isinstance(output_size[1], int):
            output_dim = output_size # (height, width)
    else:
        raise ValueError("ERROR: output_size must be an integer or a 2-tuple of (height, width) if provided.")

    # Ensure output_dim is correctly set if it was a single int and not None
    if isinstance(output_dim, int) or output_dim is None: # if still single int or None (original size)
        # transforms.Resize with a single int resizes the smaller edge to this number.
        # If None, it will use original size.
        # Let's be explicit based on the original logic's intent:
        if output_dim is None: # Original size
             final_resize_param = (img.size[1], img.size[0])
        elif isinstance(output_dim, int): # Square output intended by style_transfer.py logic
             final_resize_param = (output_dim, output_dim)
        else: # Should be tuple here
             final_resize_param = output_dim
    else: # output_dim is already a tuple
        final_resize_param = output_dim


    torch_loader = transforms.Compose(
        [
            transforms.Resize(final_resize_param), # Use the determined resize parameter
            transforms.ToTensor()
        ]
    )
    
    img_tensor = torch_loader(img).unsqueeze(0)
    return img_tensor.to(device)


def get_image_name_ext(img_path):
    """Get name and extension of the image file from its path."""
    return os.path.splitext(os.path.basename(img_path))[0], os.path.splitext(os.path.basename(img_path))[1][1:]