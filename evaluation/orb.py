import cv2
import numpy as np
import os
import csv
import time
import psutil

# ===================== UTILITIES =====================

def auto_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
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


def compute_reprojection_error(H, src_pts, dst_pts, mask):
    if H is None or mask is None:
        return np.nan

    src_in = src_pts[mask.ravel() == 1]
    dst_in = dst_pts[mask.ravel() == 1]

    if len(src_in) == 0:
        return np.nan

    proj = cv2.perspectiveTransform(src_in, H)
    err = np.linalg.norm(proj - dst_in, axis=2)
    return np.mean(err)


# ===================== ORB STITCHING =====================

def stitch_pair(img1, img2, pair_idx, csv_writer, debug_matches=False):
    start_time = time.time()

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(100000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    kp1_count = len(kp1)
    kp2_count = len(kp2)

    if des1 is None or des2 is None:
        return img1

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    total_matches = min(len(des1), len(des2))

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    good_count = len(good)

    if good_count < 10:
        return img1

    if debug_matches:
        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2, good, None, flags=2
        )
        cv2.imshow("ORB Matches", match_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, 5.0
    )

    if H is None:
        return img1

    inliers = int(mask.sum())
    inlier_ratio = inliers / good_count
    reproj_error = compute_reprojection_error(
        H, src_pts, dst_pts, mask
    )

    # ---- Warp panorama ----
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.float32(
        [[0,0], [0,h1], [w1,h1], [w1,0]]
    ).reshape(-1,1,2)

    warped_corners = cv2.perspectiveTransform(
        corners_img1, H
    )

    all_corners = np.concatenate((
        warped_corners,
        np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
    ), axis=0)

    xmin, ymin = np.int32(all_corners.min(axis=0).ravel())
    xmax, ymax = np.int32(all_corners.max(axis=0).ravel())

    translation = [-xmin, -ymin]
    T = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

    result = cv2.warpPerspective(
        img1,
        T @ H,
        (xmax - xmin, ymax - ymin)
    )

    result = blend_images(
        result, img2, translation[0], translation[1]
    )

    elapsed_time = round(time.time() - start_time, 2)
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    # ---- CSV LOG ----
    csv_writer.writerow([
        pair_idx,
        kp1_count,
        kp2_count,
        total_matches,
        good_count,
        inliers,
        round(inlier_ratio, 3),
        round(reproj_error, 2),
        elapsed_time,
        round(mem_mb, 2)
    ])

    return result


def stitch_images_from_folder(
    folder_path,
    debug_matches=False,
    log_path="orb_pair_evaluation.csv"
):
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(image_files) < 2:
        print("Need at least two images.")
        return None

    images = [cv2.imread(p) for p in image_files]
    stitched = images[0]

    with open(log_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Pair",
            "KP Img1",
            "KP Img2",
            "Total Matches",
            "Good Matches",
            "Inliers",
            "Inliers Ratio",
            "Reprojection Error",
            "Time (s)",
            "Memory (MB)"
        ])

        for i in range(1, len(images)):
            print(f"Stitching ORB pair {i}")
            stitched = stitch_pair(
                stitched,
                images[i],
                pair_idx=i,
                csv_writer=writer,
                debug_matches=debug_matches
            )

            if i % 5 == 0:
                stitched = auto_crop(stitched)
                stitched = cv2.resize(
                    stitched, None, fx=0.5, fy=0.5,
                    interpolation=cv2.INTER_AREA
                )

    return auto_crop(stitched)


# ===================== MAIN =====================

if __name__ == "__main__":
    folder_path = "dataset2"  # ganti sesuai folder

    result = stitch_images_from_folder(
        folder_path,
        debug_matches=False,
        log_path="evaluation/orb/orb_evaluation.csv"
    )

    if result is not None:
        cv2.imwrite("evaluation/orb/stitched_orb_output.jpg", result)
        cv2.imshow("ORB Stitched Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
