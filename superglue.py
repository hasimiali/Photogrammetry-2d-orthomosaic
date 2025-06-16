import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

# Import modul dari repositori SuperGlue

from SuperGluePretrainedNetwork.models.matching import Matching

# Konfigurasi perangkat
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Konfigurasi Matching
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}

# Inisialisasi Matching
matching = Matching(config).eval().to(device)

def load_images_from_folder(folder):
    image_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    images = [cv2.imread(p) for p in image_paths]
    return images, image_paths

def compute_homographies(images):
    homographies = [np.eye(3)]
    for i in tqdm(range(len(images) - 1), desc="Computing homographies"):
        img0 = images[i]
        img1 = images[i + 1]

        # Konversi ke grayscale dan normalisasi
        img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img0_gray = img0_gray.astype('float32') / 255.0
        img1_gray = img1_gray.astype('float32') / 255.0

        # Persiapkan input untuk Matching
        data = {
            'image0': torch.from_numpy(img0_gray)[None, None].to(device),
            'image1': torch.from_numpy(img1_gray)[None, None].to(device)
        }

        # Jalankan Matching
        with torch.no_grad():
            pred = matching(data)

        # Ekstrak keypoints dan matches
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()

        # Filter matches yang valid
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        if len(mkpts0) < 4:
            print(f"Not enough matches between image {i} and {i+1}")
            homographies.append(homographies[-1])
            continue

        # Hitung homografi menggunakan RANSAC
        H, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
        if H is None:
            print(f"Homography computation failed between image {i} and {i+1}")
            homographies.append(homographies[-1])
            continue

        # Akumulasi homografi
        H_global = homographies[-1] @ H
        homographies.append(H_global)

    return homographies

def warp_and_blend(images, homographies):
    # Hitung ukuran kanvas
    corners = []
    for img, H in zip(images, homographies):
        h, w = img.shape[:2]
        pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype='float32').reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, H)
        corners.append(warped_pts)

    all_corners = np.concatenate(corners, axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Hitung translasi
    translation = np.array([[1, 0, -xmin],
                            [0, 1, -ymin],
                            [0, 0, 1]])

    # Ukuran kanvas hasil
    result_shape = (xmax - xmin, ymax - ymin)
    result = np.zeros((result_shape[1], result_shape[0], 3), dtype=np.uint8)

    for img, H in zip(images, homographies):
        warped = cv2.warpPerspective(img, translation @ H, result_shape)
        mask = (warped > 0).astype(np.uint8) * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(mask)

        roi = cv2.bitwise_and(result, result, mask=mask_inv)
        warped_fg = cv2.bitwise_and(warped, warped, mask=mask)

        result = cv2.add(roi, warped_fg)

    return result

def main():
    folder_path = 'gabungan'  # Ganti dengan path folder gambar Anda
    images, image_paths = load_images_from_folder(folder_path)
    if len(images) < 2:
        print("Need at least two images to stitch.")
        return

    homographies = compute_homographies(images)
    stitched_image = warp_and_blend(images, homographies)

    # Simpan dan tampilkan hasil
    output_path = 'stitched_superglue_output.jpg'
    cv2.imwrite(output_path, stitched_image)
    cv2.imshow('Stitched Image', stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
