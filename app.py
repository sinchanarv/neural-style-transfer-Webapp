# app.py
import os
import uuid
import json
import traceback # For detailed error logging

import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # For comparing embeddings

from flask import Flask, render_template, request, jsonify # send_from_directory is not explicitly used due to auto static handling
from werkzeug.utils import secure_filename 

# --- Import your existing style transfer code's main function ---
from image_style_transfer import image_style_transfer as run_style_transfer_script

app = Flask(__name__)

# --- Configuration for Flask App ---
UPLOAD_FOLDER = 'uploads' # For temporary user uploads
RESULT_FOLDER = 'static/results' # Base folder for final and intermediate results (web accessible)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER # e.g., static/results
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload size

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True) # Creates 'static/results'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- AI Suggestion: Load Model and Precomputed Data (runs once at app start) ---
print("AI Suggestion: Initializing model and loading style library...")
SUGGESTION_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
suggestion_feature_model = None
suggestion_embedding_preprocess = None
style_embeddings_library = None
style_image_paths_library = None

try:
    suggestion_feature_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the final classification layer to get features
    suggestion_feature_model = torch.nn.Sequential(*(list(suggestion_feature_model.children())[:-1]))
    suggestion_feature_model.to(SUGGESTION_DEVICE)
    suggestion_feature_model.eval() # Set to evaluation mode

    suggestion_embedding_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    STYLE_EMBEDDINGS_PATH = 'static/style_embeddings.npy'
    STYLE_IMAGE_PATHS_PATH = 'static/style_image_paths.json'
    
    if os.path.exists(STYLE_EMBEDDINGS_PATH) and os.path.exists(STYLE_IMAGE_PATHS_PATH):
        style_embeddings_library = np.load(STYLE_EMBEDDINGS_PATH)
        with open(STYLE_IMAGE_PATHS_PATH, 'r') as f:
            style_image_paths_library = json.load(f)
        print(f"AI Suggestion: Style library with {len(style_image_paths_library)} styles loaded successfully from static folder.")
    else:
        print(f"AI Suggestion: WARNING - Embedding file '{STYLE_EMBEDDINGS_PATH}' or paths file '{STYLE_IMAGE_PATHS_PATH}' not found.")
        # Keep them as None so the /suggest_styles route can report unavailability
except Exception as e:
    print(f"AI Suggestion: CRITICAL WARNING - Could not load style library or suggestion model: {e}")
    traceback.print_exc()
    # style_embeddings_library and style_image_paths_library will remain None
# --- End AI Suggestion Setup ---


def get_image_embedding_for_suggestion(image_pil, model, preprocess_transform, device_to_use):
    """Generates an embedding for a single PIL image for suggestions."""
    try:
        img_rgb = image_pil.convert('RGB') # Ensure image is RGB
        img_t = preprocess_transform(img_rgb)
        batch_t = torch.unsqueeze(img_t, 0).to(device_to_use)
        with torch.no_grad():
            embedding = model(batch_t)
        return embedding.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error in get_image_embedding_for_suggestion: {e}")
        traceback.print_exc()
        return None

# --- Main Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stylize', methods=['POST'])
def stylize_image_route():
    print("Received /stylize request")
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return jsonify({'error': 'Missing content or style image file parts in request'}), 400

    content_file = request.files['content_image']
    style_file = request.files['style_image']

    if content_file.filename == '' or style_file.filename == '':
        return jsonify({'error': 'No file selected for content or style image'}), 400

    if not (allowed_file(content_file.filename) and allowed_file(style_file.filename)):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg'}), 400

    # Secure filenames and save uploaded images temporarily
    # Using UUIDs for temporary server-side files to avoid name clashes
    unique_session_id = str(uuid.uuid4()) # For this particular stylization request
    content_filename_ext = secure_filename(content_file.filename).rsplit('.', 1)[1].lower()
    style_filename_ext = secure_filename(style_file.filename).rsplit('.', 1)[1].lower()
    
    temp_content_filename = f"{unique_session_id}_content.{content_filename_ext}"
    temp_style_filename = f"{unique_session_id}_style.{style_filename_ext}"

    content_path_server = os.path.join(app.config['UPLOAD_FOLDER'], temp_content_filename)
    style_path_server = os.path.join(app.config['UPLOAD_FOLDER'], temp_style_filename)

    content_file.save(content_path_server)
    style_file.save(style_path_server)
    print(f"Saved content for processing to: {content_path_server}")
    print(f"Saved style for processing to: {style_path_server}")

    try:
        output_size_str = request.form.get('output_size', '512')
        if 'x' in output_size_str:
            h, w = map(int, output_size_str.split('x'))
            output_image_size_param = [h, w]
        else:
            output_image_size_param = [int(output_size_str)]

        num_epochs_param = int(request.form.get('num_epochs', 3000))
        learning_rate_param = float(request.form.get('learning_rate', 0.002))
        alpha_param = float(request.form.get('alpha', 1.0))
        beta_param = float(request.form.get('beta', 100000.0))
        output_image_format_param = "png" # Fixed to PNG as per requirement
    except ValueError:
        return jsonify({'error': 'Invalid parameter format (e.g., size, epochs must be numbers)'}), 400

    # Define output directory for this specific request's results
    # e.g., static/results/unique_session_id/
    script_output_dir = os.path.join(app.config['RESULT_FOLDER'], unique_session_id)
    os.makedirs(script_output_dir, exist_ok=True)
    
    config_for_script = {
        'content_filepath': content_path_server, # Path to temp uploaded content
        'style_filepath': style_path_server,     # Path to temp uploaded style
        'output_dir': script_output_dir,         # Where style_transfer_script saves its output
        'output_image_size': output_image_size_param,
        'output_image_format': output_image_format_param,
        'train_config_path': None, # We pass params directly
        'quiet': False             # For verbose output from the script
    }

    train_config_params = {
        'num_epochs': num_epochs_param,
        'learning_rate': learning_rate_param,
        'alpha': alpha_param,
        'beta': beta_param
        # 'capture_content_features_from' and 'capture_style_features_from' will use defaults in train_model.py
    }
    print(f"Config for style transfer script: {config_for_script}")
    print(f"Direct training parameters: {train_config_params}")

    try:
        # This call should now run the image_style_transfer.py's main logic
        final_image_details = run_style_transfer_script(config_for_script, train_config_direct=train_config_params)

        if final_image_details and final_image_details.get("success"):
            # server_path_to_final_image is relative to project root, e.g., "static/results/unique_id/image.png"
            server_path_to_final_image = os.path.normpath(final_image_details["output_path"])
            
            static_folder_name = "static"
            url_path_prefix_to_strip = static_folder_name + os.sep 

            # Create URL relative to the 'static' folder for the client
            if server_path_to_final_image.startswith(url_path_prefix_to_strip):
                final_image_url_for_client = server_path_to_final_image[len(url_path_prefix_to_strip):].replace('\\', '/')
            else:
                final_image_url_for_client = server_path_to_final_image.replace('\\', '/') 
                print(f"Warning: final_image_server_path '{server_path_to_final_image}' did not start with '{url_path_prefix_to_strip}'. URL for client might be incorrect: {final_image_url_for_client}")

            intermediate_image_urls_for_client = []
            # script_output_dir is 'static/results/unique_id'
            actual_intermediate_dir_on_server = os.path.join(script_output_dir, "intermediate_steps")

            if os.path.exists(actual_intermediate_dir_on_server):
                files_in_intermediate = os.listdir(actual_intermediate_dir_on_server)
                sorted_files = sorted(
                    [f for f in files_in_intermediate if f.startswith("step-") and f.endswith(f".{output_image_format_param}")],
                    key=lambda x: int(x.split('-')[1].split('.')[0]) # Sort by epoch number
                )
                for fname in sorted_files:
                    full_intermediate_path_on_server = os.path.normpath(os.path.join(actual_intermediate_dir_on_server, fname))
                    if full_intermediate_path_on_server.startswith(url_path_prefix_to_strip):
                        url_path = full_intermediate_path_on_server[len(url_path_prefix_to_strip):].replace('\\', '/')
                    else:
                        url_path = full_intermediate_path_on_server.replace('\\','/')
                        print(f"Warning: intermediate_path '{full_intermediate_path_on_server}' did not start with '{url_path_prefix_to_strip}'. URL for client might be incorrect: {url_path}")
                    intermediate_image_urls_for_client.append(url_path)
            
            print(f"Final image URL (sent to client): {final_image_url_for_client}")
            print(f"Intermediate URLs (sent to client): {intermediate_image_urls_for_client}")
            
            return jsonify({
                'message': 'Stylization successful!',
                'result_image_url': final_image_url_for_client,
                'intermediate_image_urls': intermediate_image_urls_for_client
            }), 200
        else:
            error_msg = final_image_details.get("error", "Stylization script reported failure.") if final_image_details else "Stylization script did not return expected details."
            return jsonify({'error': error_msg}), 500

    except Exception as e:
        print(f"Exception during stylization process call: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'An internal server error occurred during stylization. Check server logs.'}), 500
    finally:
        # Clean up uploaded temporary files (content_path_server, style_path_server)
        if os.path.exists(content_path_server):
            try: os.remove(content_path_server)
            except Exception as e_rem: print(f"Error removing temp content file {content_path_server}: {e_rem}")
        if os.path.exists(style_path_server):
            try: os.remove(style_path_server)
            except Exception as e_rem: print(f"Error removing temp style file {style_path_server}: {e_rem}")

@app.route('/suggest_styles', methods=['POST'])
def suggest_styles_route():
    print("Received /suggest_styles request")
    if 'content_image' not in request.files:
        return jsonify({'error': 'No content_image file part in request'}), 400
    
    content_file = request.files['content_image']
    if content_file.filename == '':
        return jsonify({'error': 'No file selected for content_image'}), 400

    if not allowed_file(content_file.filename):
        return jsonify({'error': 'Invalid file type for content image. Allowed: png, jpg, jpeg'}), 400

    if style_embeddings_library is None or style_image_paths_library is None:
        # This check ensures that the precomputed data loaded at app start is available
        print("AI Suggestion: Style library not available for suggestions.")
        return jsonify({'error': 'Style suggestion library not available on server.'}), 500

    try:
        content_pil_image = Image.open(content_file.stream)
        
        content_embedding = get_image_embedding_for_suggestion(
            content_pil_image, 
            suggestion_feature_model, 
            suggestion_embedding_preprocess, 
            SUGGESTION_DEVICE
        )

        if content_embedding is None:
            return jsonify({'error': 'Could not generate embedding for the uploaded content image.'}), 500

        similarities = cosine_similarity(content_embedding.reshape(1, -1), style_embeddings_library)
        
        num_suggestions = 3
        sorted_indices = np.argsort(similarities[0])[::-1]
        top_n_indices = sorted_indices[:num_suggestions]

        # style_image_paths_library contains paths like "style_library/image.jpg"
        # These are already relative to the 'static' folder and use forward slashes.
        suggested_style_urls = [style_image_paths_library[i] for i in top_n_indices]
        
        print(f"AI Suggested style paths (sent to client): {suggested_style_urls}")
        return jsonify({'suggested_styles': suggested_style_urls}), 200

    except Exception as e:
        print(f"Error in /suggest_styles endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'An internal error occurred while suggesting styles. Check server logs.'}), 500

if __name__ == '__main__':
    # Ensure 'templates' and 'static' folders exist for Flask to find them
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
        
    app.run(debug=True, host='0.0.0.0', port=5000)