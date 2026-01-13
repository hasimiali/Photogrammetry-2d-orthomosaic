import cv2
import numpy as np
import os
import csv

# ================= UTIL =================

def compute_homography_metrics(img1, img2, ratio_thresh=0.75):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ================= SIFT =================
    sift = cv2.SIFT_create(100000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return None

    # ================= FLANN =================
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # ================= Lowe Ratio Test =================
    good = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    if len(good) < 10:
        return None

    # ================= HOMOGRAPHY =================
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    inliers = mask.ravel().astype(bool)
    inlier_count = np.sum(inliers)
    inlier_ratio = inlier_count / len(good)

    # ================= REPROJECTION ERROR =================
    src_inliers = src_pts[inliers]
    dst_inliers = dst_pts[inliers]
    projected_pts = cv2.perspectiveTransform(src_inliers, H)

    reprojection_errors = np.linalg.norm(
        dst_inliers.reshape(-1, 2) - projected_pts.reshape(-1, 2),
        axis=1
    )

    avg_reprojection_error = np.mean(reprojection_errors)

    return {
        "good_matches": len(good),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "reproj_error": avg_reprojection_error
    }


# ================= MAIN EVALUATION =================

def evaluate_folder_homography(
    image_folder,
    output_csv="homography/rata2/avg_metrics_sift_flann.csv"
):
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    if len(image_files) < 2:
        print("Minimal butuh 2 gambar.")
        return

    total_good = 0
    total_inliers = 0
    total_inlier_ratio = 0
    total_reproj_error = 0
    valid_pairs = 0

    # === Sequential comparison (1-2, 2-3, ...) ===
    for i in range(len(image_files) - 1):
        img1 = cv2.imread(os.path.join(image_folder, image_files[i]))
        img2 = cv2.imread(os.path.join(image_folder, image_files[i + 1]))

        if img1 is None or img2 is None:
            continue

        metrics = compute_homography_metrics(img1, img2)
        if metrics is None:
            print(f"{image_files[i]} → {image_files[i+1]}: FAILED")
            continue

        valid_pairs += 1
        total_good += metrics["good_matches"]
        total_inliers += metrics["inliers"]
        total_inlier_ratio += metrics["inlier_ratio"]
        total_reproj_error += metrics["reproj_error"]

        print(
            f"{image_files[i]} → {image_files[i+1]} | "
            f"Good: {metrics['good_matches']}, "
            f"Inliers: {metrics['inliers']}, "
            f"Ratio: {metrics['inlier_ratio']:.3f}, "
            f"Reproj(px): {metrics['reproj_error']:.2f}"
        )

    if valid_pairs == 0:
        print("Tidak ada pasangan valid.")
        return

    avg_good = total_good / valid_pairs
    avg_inliers = total_inliers / valid_pairs
    avg_ratio = total_inlier_ratio / valid_pairs
    avg_reproj = total_reproj_error / valid_pairs

    print("\n================ AVERAGE RESULTS ================")
    print(f"Total Pairs                 : {valid_pairs}")
    print(f"Avg Good Matches            : {avg_good:.2f}")
    print(f"Avg Inliers                 : {avg_inliers:.2f}")
    print(f"Avg Inlier Ratio            : {avg_ratio:.3f}")
    print(f"Avg Reprojection Error (px) : {avg_reproj:.2f}")
    print("=================================================")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Avg Good Matches",
            "Avg Inliers",
            "Avg Inlier Ratio",
            "Avg Reprojection Error (px)"
        ])
        writer.writerow([
            round(avg_good, 2),
            round(avg_inliers, 2),
            round(avg_ratio, 3),
            round(avg_reproj, 2)
        ])


# ================= USAGE =================

evaluate_folder_homography(
    image_folder="dataset2",
    output_csv="homography/rata2/avg_metrics_sift_flann.csv"
)
