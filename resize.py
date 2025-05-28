import cv2
import os

def resize_images_keep_aspect(input_folder, output_folder, max_width, max_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Could not read {image_file}, skipping.")
            continue

        h, w = image.shape[:2]

        # Compute scale factor to fit within max dimensions
        scale = min(max_width / w, max_height / h)

        # Resize while maintaining aspect ratio
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, resized_image)
        print(f"Resized (kept ratio) and saved: {image_file} — {new_w}x{new_h}")

# === Usage ===
input_folder = 'zoo'
output_folder = 'zoo_resized'
max_width = 1920
max_height = 1080

resize_images_keep_aspect(input_folder, output_folder, max_width, max_height)
