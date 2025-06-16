import cv2
import numpy as np
import os
import psutil
import csv
from tqdm import tqdm

def compute_homographies(images):
    sift = cv2.SIFT_create()
    homographies = [np.eye(3)]

    print("Computing homographies...")
    for i in tqdm(range(len(images) - 1), desc="Matching images"):
        img1_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(images[i + 1], cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            print(f"Skipping pair {i}-{i + 1}, descriptors missing or too few.")
            homographies.append(homographies[-1].copy())
            continue

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 20:
            print(f"Skipping pair {i}-{i + 1}, not enough good matches.")
            homographies.append(homographies[-1].copy())
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

        if H is not None:
            H_global = homographies[-1] @ H
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

def stitch_images(images, homographies, log_path):
    print("Stitching final image...")
    xmin, ymin, xmax, ymax = get_canvas_size(images, homographies)
    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    result_shape = (xmax - xmin, ymax - ymin)

    result = np.zeros((result_shape[1], result_shape[0], 3), dtype=np.uint8)

    with open(log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Total Images', 'Total Dataset Image Size (pixels)', 'Memory (MB)'])

        total_image_pixels = 0

        for i, (img, H) in enumerate(tqdm(zip(images, homographies), desc="Warping and blending", total=len(images))):
            h, w = img.shape[:2]
            total_image_pixels += h * w  # Accumulate total image size

            warped = cv2.warpPerspective(img, translation @ H, result_shape)

            mask = (warped > 0).astype(np.uint8) * 255
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_inv = cv2.bitwise_not(mask)

            roi = cv2.bitwise_and(result, result, mask=mask_inv)
            warped_fg = cv2.bitwise_and(warped, warped, mask=mask)

            result = cv2.add(roi, warped_fg)

            # Logging
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            writer.writerow([i + 1, total_image_pixels, round(mem_mb, 2)])

    return result


def stitch_from_folder(folder_path, log_path='stitch_log.csv'):
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(image_files) < 2:
        print("Need at least two images to stitch.")
        return None

    images = [cv2.imread(f) for f in image_files]
    homographies = compute_homographies(images)
    stitched = stitch_images(images, homographies, log_path)
    return stitched

# === USAGE ===
folder_path = 'dataset2'  # Ganti sesuai folder
log_file = 'stitch_log.csv'
result = stitch_from_folder(folder_path, log_file)

if result is not None:
    cv2.imwrite('stitched_two_pass_output.jpg', result)
    cv2.imshow('Stitched Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
