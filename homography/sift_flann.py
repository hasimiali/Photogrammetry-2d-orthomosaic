import cv2
import numpy as np
import os
import csv
import psutil

# ================= UTIL =================

def auto_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return image[y:y+h, x:x+w]
    return image


def blend_images(base_img, overlay_img, x, y):
    h, w = overlay_img.shape[:2]
    roi = base_img[y:y+h, x:x+w]

    gray_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    base_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    overlay_fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)

    dst = cv2.add(base_bg, overlay_fg)
    base_img[y:y+h, x:x+w] = dst
    return base_img


# ================= STITCH TWO IMAGES =================

def stitch_two_images_flann(
    img1,
    img2,
    output_dir="homography/sift_flann",
    log_path="homography/sift_flann_log.csv"
):
    os.makedirs(output_dir, exist_ok=True)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("Descriptor not found.")
        return None

    # ================= FLANN =================
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # ================= RATIO TEST =================
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        print("Not enough good matches.")
        return None

    # ================= HOMOGRAPHY =================
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography failed.")
        return None

    inliers = mask.ravel().astype(bool)

    # ================= METRICS =================
    total_matches = len(matches)
    good_matches = len(good)
    inlier_count = np.sum(inliers)
    inlier_ratio = inlier_count / good_matches

    src_inliers = src_pts[inliers]
    dst_inliers = dst_pts[inliers]
    projected_pts = cv2.perspectiveTransform(src_inliers, H)

    reprojection_errors = np.linalg.norm(
        dst_inliers.reshape(-1, 2) - projected_pts.reshape(-1, 2),
        axis=1
    )
    avg_reprojection_error = np.mean(reprojection_errors)

    # ================= PRINT =================
    print(f"Total Matches              : {total_matches}")
    print(f"Good Matches               : {good_matches}")
    print(f"Avg Inliers                : {inlier_count}")
    print(f"Inlier Ratio               : {inlier_ratio:.3f}")
    print(f"Avg Reprojection Error(px) : {avg_reprojection_error:.2f}")

    # ================= IMAGE 1: INLIER MATCHES =================
    inlier_matches = [good[i] for i in range(len(good)) if inliers[i]]

    img_inliers = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(
        os.path.join(output_dir, "inlier_matches.jpg"),
        img_inliers
    )

    # ================= IMAGE 2: HOMOGRAPHY RESULT =================
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    all_corners = np.concatenate((
        warped_corners,
        np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
    ), axis=0)

    xmin, ymin = np.int32(all_corners.min(axis=0).ravel())
    xmax, ymax = np.int32(all_corners.max(axis=0).ravel())

    translation = [-xmin, -ymin]
    trans_mat = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

    result = cv2.warpPerspective(
        img1,
        trans_mat @ H,
        (xmax - xmin, ymax - ymin)
    )

    result = blend_images(result, img2, translation[0], translation[1])
    result = auto_crop(result)

    cv2.imwrite(
        os.path.join(output_dir, "stitched_result.jpg"),
        result
    )

    # ================= CSV LOG =================
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    total_pixels = img1.shape[0]*img1.shape[1] + img2.shape[0]*img2.shape[1]

    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Total Matches",
            "Good Matches",
            "Avg Inliers",
            "Inlier Ratio",
            "Avg Reprojection Error (px)",
            "Total Pixels",
            "Memory (MB)"
        ])
        writer.writerow([
            total_matches,
            good_matches,
            inlier_count,
            round(inlier_ratio, 3),
            round(avg_reprojection_error, 2),
            total_pixels,
            round(mem_mb, 2)
        ])

    return result


# ================= USAGE =================

img1 = cv2.imread("homography/image1.jpg")
img2 = cv2.imread("homography/image2.jpg")

result = stitch_two_images_flann(img1, img2)

if result is not None:
    cv2.imshow("Stitched Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
