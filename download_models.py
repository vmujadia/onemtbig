import os
import subprocess
import requests
import zipfile
from tqdm import tqdm

# Define URLs and paths
MODEL_URL = "https://vandanresearch.sgp1.digitaloceanspaces.com/bhashaverse-models/machine-translation/onemtbig/iiith-onemtbig.zip"
MODEL_DIR = "models"
MODEL_ZIP = "iiith-onemtbig.zip"

def download_model():
    """Downloads the model zip file with progress bar."""
    if not os.path.exists(MODEL_ZIP):
        print(f"Downloading model from {MODEL_URL}...")

        # Send a request to get the file size
        response = requests.get(MODEL_URL, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        # Progress bar for download
        with open(MODEL_ZIP, "wb") as file, tqdm(
            desc="Downloading Model",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

        print("Model downloaded successfully.")
    else:
        print("Model zip already exists. Skipping download.")

def extract_model():
    """Extracts the model zip file into the Triton model directory."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Extracting model files...")
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    print(f"Model extracted to {MODEL_DIR}")

def start_triton():
    """Starts the Triton Inference Server with the extracted model."""
    print("Starting Triton Inference Server...")
    subprocess.run([
        "tritonserver",
        "--model-repository=" + MODEL_DIR
    ])

if __name__ == "__main__":
    download_model()
    extract_model()
    #start_triton()
