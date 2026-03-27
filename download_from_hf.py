#!/usr/bin/env python3
import argparse
import os
from huggingface_hub import snapshot_download

MODELS = {
    "llama": [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ],
    "qwen": [
        #"Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        #"Qwen/Qwen3-4B",
        #"Qwen/Qwen3-8B",
        #"Qwen/Qwen3-14B",
        #"Qwen/Qwen3-32B",
    ],
}

CACHE_DIR = 'hf_models'

def download(models):
    for name in models:
        print(f"Downloading {name}...")
        # Create a clean folder name like 'hf_models/Qwen3-8B'
        target_dir = os.path.join(CACHE_DIR, name.split('/')[-1])
        
        try:
            # Replaced cache_dir with local_dir to get a flat file structure
            snapshot_download(
                repo_id=name, 
                local_dir=target_dir, 
                resume_download=True
            )
            print(f"  done. Saved to {target_dir}")
        except Exception as e:
            print(f"  failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=[*MODELS.keys(), "all"])
    args = parser.parse_args()

    if args.group == "all":
        targets = [m for group in MODELS.values() for m in group]
    else:
        targets = MODELS[args.group]

    download(targets)