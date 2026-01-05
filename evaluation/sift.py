import cv2
import numpy as np
import os
import psutil
import csv
import time
from skimage.metrics import structural_similarity as ssim

def auto_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return image[y:y + h, x:x + w]
    return image

def blend_images(base_img, overlay_img, x, y):
    h, w = overlay_img.shape[:2]
    roi = base_img[y:y + h, x:x + w]

    gray_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    base_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    overlay_fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)

    dst = cv2.add(base_bg, overlay_fg)
    base_img[y:y + h, x:x + w] = dst
    return base_img

def stitch_pair(img1, img2, debug_matches=False):
    """Return stitched image + evaluation metrics."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return img1, 0, 0, None, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
    good_matches = len(good)

    if good_matches < 4:
        return img1, good_matches, 0, None, 0, 0

    if debug_matches:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
        cv2.imshow("Matches", match_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Matches")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    start_time = time.time()
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    duration = time.time() - start_time

    if H is None or mask is None:
        return img1, good_matches, 0, None, 0, duration

    inlier_ratio = np.sum(mask) / len(mask)

    # Reprojection error
    pts2_est = cv2.perspectiveTransform(src_pts, H)
    reproj_error = np.mean(np.linalg.norm(dst_pts - pts2_est, axis=2))

    # Warp & blend
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img1, H)
    all_corners = np.concatenate(
        (warped_corners,
         np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)),
        axis=0
    )

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())
    translation = [-xmin, -ymin]
    trans_mat = np.array([[1, 0, translation[0]],
                          [0, 1, translation[1]],
                          [0, 0, 1]])

    output_width = xmax - xmin
    output_height = ymax - ymin
    max_dim = 32000
    if output_width > max_dim or output_height > max_dim:
        return img1, good_matches, inlier_ratio, reproj_error, 0, duration

    result = cv2.warpPerspective(img1, trans_mat @ H, (output_width, output_height))
    result = blend_images(result, img2, translation[0], translation[1])

    # Optional SSIM on overlap (quick estimate of seam quality)
    overlap_region = img2
    warp_region = result[translation[1]:translation[1]+h2,
                         translation[0]:translation[0]+w2]
    seam_ssim = ssim(cv2.cvtColor(overlap_region, cv2.COLOR_BGR2GRAY),
                     cv2.cvtColor(warp_region, cv2.COLOR_BGR2GRAY))

    return result, good_matches, inlier_ratio, reproj_error, seam_ssim, duration

def stitch_images_from_folder(folder_path, debug_matches=False, log_path='sift_stitch_eval.csv'):
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if len(image_files) < 2:
        print("Need at least two images to stitch.")
        return None

    images = [cv2.imread(p) for p in image_files]
    stitched = images[0]

    total_pixel_size = images[0].shape[0] * images[0].shape[1]

    with open(log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Step', 'ImageName', 'TotalPixels', 'Memory(MB)',
            'GoodMatches', 'InlierRatio', 'ReprojectionError(px)',
            'SSIM_Overlap', 'ProcessingTime(s)'
        ])

        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        writer.writerow([1, os.path.basename(image_files[0]), total_pixel_size,
                         round(mem_mb, 2), '-', '-', '-', '-', '-'])

        for i in range(1, len(images)):
            name = os.path.basename(image_files[i])
            print(f"Stitching image {i+1}/{len(images)}: {name}")

            stitched, good_matches, inlier_ratio, reproj_error, seam_ssim, duration = \
                stitch_pair(stitched, images[i], debug_matches)

            total_pixel_size += images[i].shape[0] * images[i].shape[1]
            mem_mb = process.memory_info().rss / (1024 * 1024)

            writer.writerow([
                i + 1, name, total_pixel_size, round(mem_mb, 2),
                good_matches, round(inlier_ratio, 3) if inlier_ratio else '-',
                round(reproj_error, 2) if reproj_error else '-',
                round(seam_ssim, 3) if seam_ssim else '-',
                round(duration, 2)
            ])

            # periodic crop/resize to control size
            if i % 5 == 0:
                stitched = auto_crop(stitched)
                scale_percent = 50
                width = int(stitched.shape[1] * scale_percent / 100)
                height = int(stitched.shape[0] * scale_percent / 100)
                stitched = cv2.resize(stitched, (width, height), interpolation=cv2.INTER_AREA)
                print(f"Resized and cropped stitched image to: {width}x{height}")

    stitched = auto_crop(stitched)
    return stitched

# === Usage ===
if __name__ == "__main__":
    folder_path = 'dataset2'  # Change to your image folder
    result = stitch_images_from_folder(folder_path, debug_matches=False)

    if result is not None:
        cv2.imwrite('stitched_sift_output_eval.jpg', result)
        print("Final stitched image saved as stitched_sift_output_eval.jpg")
        cv2.imshow('Stitched Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
