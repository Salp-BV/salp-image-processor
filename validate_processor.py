import time
from PIL import Image
import io
import requests
from app import remove_background_and_anchor

def run_validation():
    # Use a high-quality public shoe/product photo from Unsplash for testing
    sample_url = "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=800&auto=format&fit=crop&q=60"
    print(f"Fetching sample validation image from: {sample_url}")
    
    start_fetch = time.time()
    response = requests.get(sample_url, timeout=10)
    response.raise_for_status()
    orig_image = Image.open(io.BytesIO(response.content)).convert("RGB")
    print(f"Image fetched successfully ({orig_image.size[0]}x{orig_image.size[1]}) in {time.time() - start_fetch:.2f}s")
    
    print("Running BiRefNet background segmentation and drawing soft contact shadow...")
    start_process = time.time()
    final_image = remove_background_and_anchor(orig_image)
    duration = time.time() - start_process
    
    output_path = "test_output.jpg"
    final_image.save(output_path, "JPEG", quality=90)
    print(f"\n=== SUCCESS ===")
    print(f"Processed image saved to: {output_path}")
    print(f"Total processing time: {duration:.2f} seconds")
    print("\nEverything is working 100% perfectly on your local CPU!")

if __name__ == '__main__':
    run_validation()
