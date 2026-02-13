import json
import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_coco_subset(json_path: str, output_dir: str, num_images: int = 500):
    print(f"🔍 Parsing LLaVA JSON: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Extract unique image paths
    image_paths = []
    for entry in data:
        if "image" in entry:
            image_paths.append(entry["image"])
        if len(set(image_paths)) >= num_images:
            break
    
    unique_images = list(set(image_paths))[:num_images]
    print(f"📦 Found {len(unique_images)} unique images for subset download.")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Base URL for COCO 2014 train images
    base_url = "http://images.cocodataset.org/train2014/"
    
    success_count = 0
    for img_rel_path in tqdm(unique_images, desc="Downloading images"):
        filename = os.path.basename(img_rel_path)
        img_url = base_url + filename
        target_file = output_path / filename
        
        if target_file.exists() and target_file.stat().st_size > 0:
            success_count += 1
            continue
            
        # Retry logic
        for attempt in range(3):
            try:
                response = requests.get(img_url, stream=True, timeout=15)
                if response.status_code == 200:
                    with open(target_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    success_count += 1
                    break
                elif response.status_code == 404:
                    print(f"⚠️ Image not found (404): {filename}")
                    break
                else:
                    print(f"⚠️ HTTP {response.status_code} on attempt {attempt+1} for {filename}")
            except Exception as e:
                print(f"❌ Attempt {attempt+1} failed for {filename}: {e}")
            
    print(f"✅ Subset download complete: {success_count}/{len(unique_images)} images saved to {output_dir}")

if __name__ == "__main__":
    download_coco_subset(
        json_path="data/llava/llava_instruct_150k.json",
        output_dir="data/llava/train2014",
        num_images=500
    )
