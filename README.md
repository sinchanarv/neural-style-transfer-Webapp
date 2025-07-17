# Interactive Neural Style Transfer Web App

This project is a web application that implements Neural Style Transfer, allowing users to blend the content of one image with the style of another. It features an interactive UI and an AI-powered system to suggest suitable style images.

## Features
- User-friendly web interface built with Flask, HTML, CSS, and JavaScript.
- Core style transfer algorithm using PyTorch and VGG19.
- AI-powered style suggestions based on ResNet50 embeddings and cosine similarity.
- Control over stylization parameters like image size, epochs, and loss weights.
- Displays final and intermediate results with a download option.

## Setup and Installation

**Prerequisites:**
- Python 3.9+
- An NVIDIA GPU with CUDA is highly recommended for reasonable performance.

**Installation Steps:**
1. Clone the repository:
-  git clone https://github.com/YourUsername/your-repository-name.git
-  cd your-repository-name

2. Create and activate a Python virtual environment:
- python -m venv venv
- On Windows - .\venv\Scripts\activate
- On macOS/Linux - source venv/bin/activate

3. Install the required packages:
- pip install -r requirements.txt

4. **(AI Suggestion Setup)** Place your desired style images into the `static/style_library/` folder. Then, run the precomputation script: python precompute_embeddings.py
This will generate `style_embeddings.npy` and `style_image_paths.json` in the `static/` folder.

## How to Run
1. Make sure your virtual environment is active.
2. Start the Flask server: python app.py
3. Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Project Structure
- `app.py`: The main Flask application.
- `image_style_transfer.py`: Core orchestration script.
- `src/`: Contains the NST model and image processing logic.
- `templates/`: HTML templates.
- `static/`: CSS, JS, and the style library/results.
- `precompute_embeddings.py`: Script to generate embeddings for the AI suggestion feature.
- `requirements.txt`: List of Python dependencies.