import cv2
import numpy as np
import os
import psutil
import csv

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

def stitch_two_images(img1, img2, debug_matches=False, log_path="sift_two_image_log.csv"):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("Feature descriptors not found.")
        return None

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"Good matches: {len(good)}")

    if len(good) < 10:
        print("Not enough good matches for homography.")
        return None

    if debug_matches:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
        cv2.imshow("Feature Matches", match_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Feature Matches")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography computation failed.")
        return None

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
    trans_mat = np.array([[1, 0, translation[0]],
                          [0, 1, translation[1]],
                          [0, 0, 1]])

    output_width = xmax - xmin
    output_height = ymax - ymin

    max_dim = 32000
    if output_width > max_dim or output_height > max_dim:
        print(f"Output too large: {output_width}x{output_height}")
        return None

    result = cv2.warpPerspective(img1, trans_mat @ H, (output_width, output_height))
    result = blend_images(result, img2, translation[0], translation[1])
    result = auto_crop(result)

    # ===== LOGGING =====
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    total_pixels = img1.shape[0]*img1.shape[1] + img2.shape[0]*img2.shape[1]

    with open(log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Pair', 'Good Matches', 'Total Pixel Size', 'Memory (MB)'])
        writer.writerow(['2 images', len(good), total_pixels, round(mem_mb, 2)])

    return result


# ================= USAGE =================

img1 = cv2.imread("homography/image1.jpg")
img2 = cv2.imread("homography/image2.jpg")

stitched = stitch_two_images(img1, img2, debug_matches=False)

if stitched is not None:
    cv2.imwrite("homography/sift/stitched_two_images_sift.jpg", stitched)
    cv2.imshow("Stitched Result", stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
