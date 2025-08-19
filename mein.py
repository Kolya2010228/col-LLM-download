# mein.py
import os
import sys
import argparse
from pathlib import Path

# Check if running in Google Colab
IN_COLAB = "google.colab" in sys.modules

def install_dependencies():
    """Install required packages if running in Colab"""
    if not IN_COLAB:
        return
        
    print("Installing required packages in Colab...")
    !pip install -q huggingface_hub transformers accelerate

def download_model(model_id, dest_dir, revision=None, token=None, from_hf_cache=False):
    """Download model using huggingface_hub"""
    if IN_COLAB:
        install_dependencies()
    
    from huggingface_hub import snapshot_download
    
    print(f"Downloading model {model_id} to {dest_dir}...")
    
    kwargs = {
        "local_dir": dest_dir,
        "local_dir_use_symlinks": False,
        "resume_download": True,
        "ignore_patterns": ["*.bin", "*.h5", "*.ot", "*.msgpack"],
    }
    
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token
    if from_hf_cache:
        kwargs["local_files_only"] = True
    
    snapshot_download(model_id, **kwargs)
    print("Download completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face models")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-12b-it",
                       help="Model ID from Hugging Face Hub")
    parser.add_argument("--dest", type=str, default="models",
                       help="Destination directory (default: models/)")
    parser.add_argument("--revision", type=str, default=None,
                       help="Model revision (branch/tag/commit hash)")
    parser.add_argument("--token", type=str, default=None,
                       help="Hugging Face access token (or set HF_TOKEN env var)")
    parser.add_argument("--from_hf_cache", action="store_true",
                       help="Use local cache if available")
    
    args = parser.parse_args()
    
    # Use environment variable if token not provided
    token = args.token or os.getenv("HF_TOKEN")
    
    # Create destination directory if it doesn't exist
    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    download_model(
        model_id=args.model_id,
        dest_dir=str(dest_dir),
        revision=args.revision,
        token=token,
        from_hf_cache=args.from_hf_cache
    )

if __name__ == "__main__":
    main()