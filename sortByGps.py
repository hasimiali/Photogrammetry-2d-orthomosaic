import os
import exifread
import shutil
from collections import defaultdict

def get_gps_from_image(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
    try:
        lat_ref = tags['GPS GPSLatitudeRef'].values
        lat = tags['GPS GPSLatitude'].values
        lon_ref = tags['GPS GPSLongitudeRef'].values
        lon = tags['GPS GPSLongitude'].values

        def to_deg(val):
            return float(val[0].num)/val[0].den + float(val[1].num)/val[1].den/60 + float(val[2].num)/val[2].den/3600

        latitude = to_deg(lat)
        if lat_ref != 'N': latitude = -latitude

        longitude = to_deg(lon)
        if lon_ref != 'E': longitude = -longitude

        return latitude, longitude
    except:
        return None, None

def sort_images_by_gps_zigzag(image_paths, tolerance=0.0002):
    rows = defaultdict(list)

    for path in image_paths:
        lat, lon = get_gps_from_image(path)
        if lat is not None and lon is not None:
            row_key = round(lat / tolerance)
            rows[row_key].append((path, lat, lon))
        else:
            print(f"⚠️ No GPS for: {path}")

    # Sort rows from top to bottom (lat descending)
    sorted_row_keys = sorted(rows.keys(), reverse=True)
    sorted_paths = []

    for idx, row_key in enumerate(sorted_row_keys):
        row = rows[row_key]
        # Sort longitudes left-to-right
        row.sort(key=lambda x: x[2])
        if idx % 2 == 1:
            row.reverse()  # Zigzag: reverse every other row
        sorted_paths.extend([x[0] for x in row])

    return sorted_paths



def copy_sorted_images_to_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    sorted_paths = sort_images_by_gps_zigzag(image_paths)

    for idx, src_path in enumerate(sorted_paths):
        ext = os.path.splitext(src_path)[-1].lower()
        dst_filename = f"{idx:03d}{ext}"
        dst_path = os.path.join(output_folder, dst_filename)
        shutil.copy2(src_path, dst_path)
        print(f"✅ {os.path.basename(src_path)} → {dst_filename}")

    print(f"\n✅ Done! Sorted images saved to: {output_folder}")

# === USAGE ===
input_folder = 'lapanganBola'             # Folder with original images
output_folder = 'sorted_lapanganBola'     # New sorted folder

copy_sorted_images_to_folder(input_folder, output_folder)
