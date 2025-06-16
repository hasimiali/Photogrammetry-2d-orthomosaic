import cv2
import numpy as np
import os

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

def stitch_pair(img1_color, img2_color, debug_matches=False):
    # Convert to grayscale for feature detection
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None:
        print("Feature descriptors not found.")
        return img1_color

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"Good matches: {len(good)}")

    if len(good) < 10:
        print("Not enough good matches.")
        return img1_color

    if debug_matches:
        match_img = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good, None, flags=2)
        cv2.imshow("Matches", match_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Matches")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography computation failed.")
        return img1_color

    h1, w1 = img1_color.shape[:2]
    h2, w2 = img2_color.shape[:2]

    corners_img1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img1, H)
    all_corners = np.concatenate((warped_corners, np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    translation = [-xmin, -ymin]
    trans_mat = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    output_width = xmax - xmin
    output_height = ymax - ymin

    max_dim = 32000
    if output_width > max_dim or output_height > max_dim:
        print(f"Skipping stitching due to large size: {output_width}x{output_height}")
        return img1_color

    result = cv2.warpPerspective(img1_color, trans_mat @ H, (output_width, output_height))
    result = blend_images(result, img2_color, translation[0], translation[1])

    return result

def stitch_images_from_folder(folder_path, debug_matches=False):
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(image_files) < 2:
        print("Need at least two images to stitch.")
        return None

    # Read and keep only the first image
    stitched = cv2.imread(image_files[0])
    if stitched is None:
        print(f"Failed to read {image_files[0]}")
        return None

    for i in range(1, len(image_files)):
        print(f"Stitching image {i+1}/{len(image_files)}: {os.path.basename(image_files[i])}")
        img_next = cv2.imread(image_files[i])
        if img_next is None:
            print(f"Failed to read {image_files[i]}")
            continue

        stitched = stitch_pair(stitched, img_next, debug_matches)
        del img_next  # Release memory

        if i % 5 == 0:
            stitched = auto_crop(stitched)

    stitched = auto_crop(stitched)
    return stitched

# === Usage ===
folder_path = 'dataset2'  # Change this to your folder path
result = stitch_images_from_folder(folder_path, debug_matches=False)

if result is not None:
    cv2.imwrite('stitched_sift_output_blended.jpg', result)
    cv2.imshow('Stitched Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
