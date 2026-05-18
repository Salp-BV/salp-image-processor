import os
import io
import time
import requests
from PIL import Image
from app import remove_background_and_anchor

def run_scenarios():
    print("=== STARTING MULTI-SCENARIO IMAGE PIPELINE TESTING ===")
    
    # 1. Download baseline control sneaker image
    control_url = "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=800&auto=format&fit=crop&q=60"
    print(f"Downloading baseline control shoe photo from Unsplash...")
    response = requests.get(control_url, timeout=10)
    response.raise_for_status()
    base_img = Image.open(io.BytesIO(response.content)).convert("RGB")
    w, h = base_img.size
    print(f"Baseline shoe loaded successfully: {w}x{h}")
    
    # Create the test directories if they do not exist
    os.makedirs("tests_output", exist_ok=True)
    
    # ----------------------------------------------------
    # SCENARIO 1: Control Baseline Sneaker (Standard Center)
    # ----------------------------------------------------
    print("\n--- Running Scenario 1: Control Baseline Sneaker ---")
    start = time.time()
    out1 = remove_background_and_anchor(base_img)
    print(f"Scenario 1 completed in {time.time() - start:.2f}s")
    base_img.save("tests_output/scenario1_before.jpg", "JPEG", quality=90)
    out1.save("tests_output/scenario1_after.jpg", "JPEG", quality=90)
    
    # ----------------------------------------------------
    # SCENARIO 2: Programmatic Off-Center Product (Pushed to the far right/edge)
    # ----------------------------------------------------
    print("\n--- Generating & Running Scenario 2: Off-Center Product ---")
    # Take the original sneaker image and push it to the right of a wider canvas
    off_center_canvas = Image.new("RGB", (w + 400, h), (220, 100, 80)) # Colored backdrop
    # Paste shoe at the far right edge
    off_center_canvas.paste(base_img, (400, 0))
    off_center_canvas.save("tests_output/scenario2_before.jpg", "JPEG", quality=90)
    
    start = time.time()
    out2 = remove_background_and_anchor(off_center_canvas)
    print(f"Scenario 2 completed in {time.time() - start:.2f}s")
    out2.save("tests_output/scenario2_after.jpg", "JPEG", quality=90)

    # ----------------------------------------------------
    # SCENARIO 3: Programmatic Tiny Product (Occupies only 20% of canvas)
    # ----------------------------------------------------
    print("\n--- Generating & Running Scenario 3: Tiny Product (20% Size) ---")
    # Take the original sneaker and shrink it down drastically
    tiny_w, tiny_h = int(w * 0.25), int(h * 0.25)
    tiny_shoe = base_img.resize((tiny_w, tiny_h), Image.Resampling.LANCZOS)
    
    # Paste this tiny shoe on a large canvas (off-center to make it harder)
    tiny_canvas = Image.new("RGB", (w, h), (100, 180, 200)) # Light blue backdrop
    tiny_canvas.paste(tiny_shoe, (50, 50))
    tiny_canvas.save("tests_output/scenario3_before.jpg", "JPEG", quality=90)
    
    start = time.time()
    out3 = remove_background_and_anchor(tiny_canvas)
    print(f"Scenario 3 completed in {time.time() - start:.2f}s")
    out3.save("tests_output/scenario3_after.jpg", "JPEG", quality=90)

    print("\n=== ALL SCENARIOS PROCESSED SUCCESSFULLY ===")
    print("Files saved locally in tests_output/ folder.")

if __name__ == "__main__":
    run_scenarios()
