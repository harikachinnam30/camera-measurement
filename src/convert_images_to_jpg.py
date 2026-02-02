import os
from PIL import Image

input_dir = r"data\calibration_images"
output_dir = r"data\calibration_images_jpg"

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    path = os.path.join(input_dir, fname)
    if not os.path.isfile(path):
        continue

    ext = os.path.splitext(fname)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".avif"]:
        continue

    try:
        img = Image.open(path).convert("RGB")
        out_name = os.path.splitext(fname)[0] + ".jpg"
        out_path = os.path.join(output_dir, out_name)
        img.save(out_path, "JPEG", quality=95)
        print("Saved:", out_path)
    except Exception as e:
        print("Failed:", fname, e)

print("Done. Converted images are in:", output_dir)
