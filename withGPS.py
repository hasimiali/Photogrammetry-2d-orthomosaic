import os
import cv2
import numpy as np
from PIL import Image
import piexif
from tqdm import tqdm

# === GPS HANDLING ===
def dms_to_decimal(dms, ref):
    degrees, minutes, seconds = [x[0] / x[1] for x in dms]
    sign = -1 if ref in ['S', 'W'] else 1
    return sign * (degrees + minutes / 60 + seconds / 3600)

def extract_gps_from_images(folder_path):
    gps_coords = []
    filenames = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ])
    for filename in filenames:
        path = os.path.join(folder_path, filename)
        img = Image.open(path)
        exif_data = img.info.get('exif', b'')
        if not exif_data:
            gps_coords.append(None)
            continue
        exif_dict = piexif.load(exif_data)
        gps = exif_dict.get('GPS', {})
        if 1 in gps and 2 in gps and 3 in gps and 4 in gps:
            lat = dms_to_decimal(gps[2], gps[1].decode())
            lon = dms_to_decimal(gps[4], gps[3].decode())
            gps_coords.append((lat, lon))
        else:
            gps_coords.append(None)
    return gps_coords

def gps_to_xy(gps_coords, scale=1.0):
    base_lat, base_lon = gps_coords[0]
    coords_xy = []
    for lat, lon in gps_coords:
        dx = (lon - base_lon) * np.cos(np.radians(base_lat)) * 111320
        dy = (lat - base_lat) * 110540
        coords_xy.append((dx * scale, dy * scale))
    return coords_xy

# === IMAGE STITCHING ===
def compute_homographies(images, gps_offsets=None):
    sift = cv2.SIFT_create()
    homographies = [np.eye(3)]

    print("Computing homographies...")
    for i in tqdm(range(len(images) - 1), desc="Matching images"):
        img1_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(images[i + 1], cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            print(f"Skipping pair {i}-{i + 1}, descriptors missing.")
            homographies.append(homographies[-1].copy())
            continue

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 10:
            print(f"Skipping pair {i}-{i + 1}, not enough good matches.")
            homographies.append(homographies[-1].copy())
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

        if H is not None:
            H_global = homographies[-1] @ H
            if gps_offsets:
                dx = gps_offsets[i + 1][0] - gps_offsets[i][0]
                dy = gps_offsets[i + 1][1] - gps_offsets[i][1]
                gps_translation = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
                H_global = H_global @ gps_translation
            homographies.append(H_global)
        else:
            print(f"Skipping pair {i}-{i + 1}, homography failed.")
            homographies.append(homographies[-1].copy())

    return homographies

def get_canvas_size(images, homographies):
    corners = []
    for img, H in zip(images, homographies):
        h, w = img.shape[:2]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, H)
        corners.append(warped_pts)

    all_pts = np.concatenate(corners, axis=0)
    [xmin, ymin] = np.int32(all_pts.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_pts.max(axis=0).ravel())
    return xmin, ymin, xmax, ymax

def stitch_images(images, homographies):
    print("Stitching final image...")
    xmin, ymin, xmax, ymax = get_canvas_size(images, homographies)
    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    result_shape = (xmax - xmin, ymax - ymin)

    result = np.zeros((result_shape[1], result_shape[0], 3), dtype=np.uint8)

    for img, H in tqdm(zip(images, homographies), desc="Warping and blending", total=len(images)):
        warped = cv2.warpPerspective(img, translation @ H, result_shape)

        mask = (warped > 0).astype(np.uint8) * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(mask)

        roi = cv2.bitwise_and(result, result, mask=mask_inv)
        warped_fg = cv2.bitwise_and(warped, warped, mask=mask)

        result = cv2.add(roi, warped_fg)

    return result

# === MAIN FUNCTION ===
def stitch_from_folder(folder_path):
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(image_files) < 2:
        print("Need at least two images to stitch.")
        return None

    images = [cv2.imread(f) for f in image_files]
    gps_coords = extract_gps_from_images(folder_path)

    if None in gps_coords or len(gps_coords) != len(images):
        print("Incomplete or missing GPS data. Proceeding without GPS guidance.")
        gps_offsets = None
    else:
        gps_offsets = gps_to_xy(gps_coords, scale=1.0)

    homographies = compute_homographies(images, gps_offsets)
    stitched = stitch_images(images, homographies)
    return stitched

# === USAGE ===
if __name__ == "__main__":
    folder_path = "gabungan_update"  # Replace with your folder
    result = stitch_from_folder(folder_path)

    if result is not None:
        cv2.imwrite("stitched_gps_output.jpg", result)
        cv2.imshow("Stitched Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
