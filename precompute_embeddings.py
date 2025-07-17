import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json # To save the list of image paths

def get_embedding(image_path, model, preprocess_transform, device_to_use):
    """Generates an embedding for a single image."""
    try:
        img = Image.open(image_path).convert('RGB') # Ensure 3 channels (RGB)
        img_t = preprocess_transform(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device_to_use) # Create a mini-batch

        with torch.no_grad(): # We don't need to calculate gradients
            embedding = model(batch_t)
        
        # Flatten the embedding and convert to a NumPy array
        return embedding.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    print("Starting to precompute style embeddings...")

    # --- Configuration ---
    style_image_folder_path = 'static/style_library/' # Path to your curated style images
    output_embeddings_file = 'static/style_embeddings.npy'
    output_paths_file = 'static/style_image_paths.json'
    allowed_extensions = ('.png', '.jpg', '.jpeg')

    # --- Setup Model and Preprocessing ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the final fully connected layer (the classifier) to get feature embeddings
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.to(device)
    model.eval() # Set the model to evaluation mode

    # Define the same preprocessing transformations used for ImageNet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Process Style Images ---
    all_embeddings = []
    all_image_relative_paths = []

    if not os.path.exists(style_image_folder_path):
        print(f"ERROR: Style image folder not found at '{style_image_folder_path}'")
        return

    print(f"Processing images from: {style_image_folder_path}")
    for filename in os.listdir(style_image_folder_path):
        if filename.lower().endswith(allowed_extensions):
            full_image_path = os.path.join(style_image_folder_path, filename)
            print(f"  Processing: {filename}")
            
            embedding = get_embedding(full_image_path, model, preprocess, device)
            
            if embedding is not None:
                all_embeddings.append(embedding)
                # Store path relative to 'static' folder for web serving
                relative_path = os.path.join('style_library', filename).replace('\\', '/')
                all_image_relative_paths.append(relative_path)
            else:
                print(f"    Skipping {filename} due to processing error.")
    
    if not all_embeddings:
        print("No embeddings were generated. Check your style library folder and image files.")
        return

    # --- Save Embeddings and Paths ---
    np_embeddings = np.array(all_embeddings)
    np.save(output_embeddings_file, np_embeddings)
    print(f"Saved {len(np_embeddings)} embeddings to {output_embeddings_file}")

    with open(output_paths_file, 'w') as f:
        json.dump(all_image_relative_paths, f, indent=4)
    print(f"Saved {len(all_image_relative_paths)} image paths to {output_paths_file}")

    print("Precomputation of style embeddings complete.")

if __name__ == '__main__':
    main()