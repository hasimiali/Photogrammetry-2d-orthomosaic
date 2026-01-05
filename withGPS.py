import cv2
import numpy as np
import os
from tqdm import tqdm
import exifread

def get_gps_from_image(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
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
    
def sort_images_by_gps(image_paths, tolerance=0.0005):
    coords = []
    for path in image_paths:
        lat, lon = get_gps_from_image(path)
        if lat is not None and lon is not None:
            coords.append((path, lat, lon))
        else:
            print(f"⚠️ GPS data not found for {path}")

    # Sort first by latitude (descending), then longitude (ascending)
    coords.sort(key=lambda x: (-round(x[1]/tolerance), x[2]))

    sorted_paths = [p for p, _, _ in coords]
    return sorted_paths



def compute_homographies(images):
    sift = cv2.SIFT_create()
    homographies = [np.eye(3)]

    print("Computing homographies...")
    for i in tqdm(range(len(images) - 1), desc="Matching images"):
        img1_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(images[i+1], cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            print(f"Skipping pair {i}-{i+1}, descriptors missing or too few.")
            homographies.append(homographies[-1].copy())
            continue


        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 20:
            print(f"Skipping pair {i}-{i+1}, not enough good matches.")
            homographies.append(homographies[-1].copy())
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

        if H is not None:
            H_global = homographies[-1] @ H
            homographies.append(H_global)
        else:
            print(f"Skipping pair {i}-{i+1}, homography failed.")
            homographies.append(homographies[-1].copy())

    return homographies

def get_canvas_size(images, homographies, max_width=15000, max_height=15000):
    corners = []
    for img, H in zip(images, homographies):
        h, w = img.shape[:2]
        pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, H)
        corners.append(warped_pts)

    all_pts = np.concatenate(corners, axis=0)
    [xmin, ymin] = np.int32(all_pts.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_pts.max(axis=0).ravel())

    width = xmax - xmin
    height = ymax - ymin

    if width > max_width or height > max_height:
        raise MemoryError(f"🛑 Canvas terlalu besar: {width}x{height} px. Homography mungkin error.")

    return xmin, ymin, xmax, ymax


def stitch_images(images, homographies):
    print("Stitching final image...")
    xmin, ymin, xmax, ymax = get_canvas_size(images, homographies)
    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    result_shape = (xmax - xmin, ymax - ymin)

    result = np.zeros((result_shape[1], result_shape[0], 3), dtype=np.uint8)

    for img, H in tqdm(zip(images, homographies), desc="Warping and blending", total=len(images)):
        warped = cv2.warpPerspective(img, translation @ H, result_shape)

        # Create mask and blend
        mask = (warped > 0).astype(np.uint8) * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(mask)

        roi = cv2.bitwise_and(result, result, mask=mask_inv)
        warped_fg = cv2.bitwise_and(warped, warped, mask=mask)

        result = cv2.add(roi, warped_fg)

    return result

def stitch_from_folder(folder_path):
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(image_files) < 2:
        print("Need at least two images to stitch.")
        return None

    # Sort by GPS
    sorted_files = sort_images_by_gps(image_files)
    images = [cv2.imread(f) for f in sorted_files]

    homographies = compute_homographies(images)
    stitched = stitch_images(images, homographies)
    return stitched

# === USAGE ===
folder_path = 'lapanganBola'  # Ganti sesuai folder
result = stitch_from_folder(folder_path)

if result is not None:
    cv2.imwrite('stitched_two_pass_output.jpg', result)
    cv2.imshow('Stitched Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()