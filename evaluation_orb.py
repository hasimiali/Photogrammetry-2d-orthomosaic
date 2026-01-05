import cv2
import numpy as np
import os
import csv
import psutil
import time

# ================= UTIL =================

def auto_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y+h, x:x+w]
    return image

def blend_images(base_img, overlay_img, x, y):
    h, w = overlay_img.shape[:2]
    roi = base_img[y:y+h, x:x+w]

    gray = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)

    base_img[y:y+h, x:x+w] = cv2.add(bg, fg)
    return base_img

# ================= EVALUATION =================

def evaluate_pair_orb(img1, img2):
    start = time.time()

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=40000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 10:
        return None, None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None, None

    inliers = int(mask.sum())
    projected = cv2.perspectiveTransform(src, H)
    reproj_error = np.mean(np.linalg.norm(projected - dst, axis=2))
    elapsed = time.time() - start

    stats = {
        "kp1": len(kp1),
        "kp2": len(kp2),
        "total_matches": len(matches),
        "good_matches": len(good),
        "inliers": inliers,
        "reproj_error": round(reproj_error, 2),
        "time": round(elapsed, 2)
    }

    return H, stats

def stitch_pair(base, img, H):
    h1, w1 = base.shape[:2]
    h2, w2 = img.shape[:2]

    corners = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(corners, H)

    all_corners = np.concatenate((warped,
        np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)))

    xmin, ymin = np.int32(all_corners.min(axis=0).ravel())
    xmax, ymax = np.int32(all_corners.max(axis=0).ravel())

    T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])
    result = cv2.warpPerspective(base, T @ H, (xmax-xmin, ymax-ymin))
    return blend_images(result, img, -xmin, -ymin)

# ================= PIPELINE =================

def run_photogrammetry_orb(folder):
    files = sorted([os.path.join(folder,f) for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg','.png','.jpeg'))])
    images = [cv2.imread(f) for f in files]

    total_images = len(images)
    print(f"Total images to process: {total_images}")

    stitched = images[0]
    process = psutil.Process(os.getpid())

    kp1_list, kp2_list = [], []

    with open("evaluation_orb.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Pair",
            "KP Img1",
            "KP Img2",
            "Total Matches",
            "Good Matches",
            "Inliers",
            "Reprojection Error",
            "Time (s)",
            "Memory (MB)"
        ])

        for i in range(1, total_images):
            progress = (i / (total_images - 1)) * 100
            print(f"Processing image {i+1}/{total_images} ({progress:.1f}%)")

            H, stats = evaluate_pair_orb(images[i-1], images[i])
            if H is None:
                print("  ⚠ Pair skipped (insufficient matches)")
                continue

            stitched = stitch_pair(stitched, images[i], H)

            mem = process.memory_info().rss / (1024*1024)
            writer.writerow([
                i,
                stats["kp1"],
                stats["kp2"],
                stats["total_matches"],
                stats["good_matches"],
                stats["inliers"],
                stats["reproj_error"],
                stats["time"],
                round(mem,2)
            ])

            # simpan untuk statistik keypoint
            kp1_list.append(stats["kp1"])
            kp2_list.append(stats["kp2"])

            if i % 5 == 0:
                stitched = auto_crop(stitched)
                stitched = cv2.resize(
                    stitched,
                    (stitched.shape[1]//2, stitched.shape[0]//2),
                    interpolation=cv2.INTER_AREA
                )

    stitched = auto_crop(stitched)

    # ================= KEYPOINT STATISTICS =================
    def summarize_keypoints(kp_list, name):
        kp_array = np.array(kp_list)
        print(f"\n{name} Statistics:")
        print(f"  Min: {kp_array.min()}")
        print(f"  Mean: {kp_array.mean():.2f}")
        print(f"  Median: {np.median(kp_array)}")
        print(f"  Max: {kp_array.max()}")

    summarize_keypoints(kp1_list, "KP Img1")
    summarize_keypoints(kp2_list, "KP Img2")

    print(f"\nPhotogrammetry ORB completed: {total_images} images processed.")
    return stitched

# ================= RUN =================

result = run_photogrammetry_orb("dataset2")
cv2.imwrite("stitched_orb_final.jpg", result)
cv2.imshow("ORB Stitching Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
