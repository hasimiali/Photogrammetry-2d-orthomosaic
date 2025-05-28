import cv2
import numpy as np
import os

def auto_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)  # kontur terbesar
        x, y, w, h = cv2.boundingRect(c)
        return image[y:y+h, x:x+w]
    return image

def blend_images(base_img, overlay_img, x, y):
    # Membuat ROI (region of interest) di base_img
    h, w = overlay_img.shape[:2]
    roi = base_img[y:y+h, x:x+w]

    # Buat mask dari overlay_img (non hitam)
    gray_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)

    # Area base_img tanpa overlay
    base_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Area overlay_img yang akan ditaruh
    overlay_fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)

    # Gabungkan
    dst = cv2.add(base_bg, overlay_fg)
    base_img[y:y+h, x:x+w] = dst
    return base_img

def stitch_pair(img1, img2, debug_matches=False):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("Feature descriptors not found.")
        return img1

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"Good matches: {len(good)}")

    if len(good) < 10:
        print("Not enough good matches.")
        return img1

    if debug_matches:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
        cv2.imshow("Matches", match_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Matches")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography computation failed.")
        return img1

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

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
        return img1

    # Warp img1 ke hasil panorama
    result = cv2.warpPerspective(img1, trans_mat @ H, (output_width, output_height))

    # Blend img2 ke result dengan posisi yang sudah ditranslasi
    result = blend_images(result, img2, translation[0], translation[1])

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

    images = [cv2.imread(p) for p in image_files]
    stitched = images[0]

    for i in range(1, len(images)):
        print(f"Stitching image {i+1}/{len(images)}: {os.path.basename(image_files[i])}")
        stitched = stitch_pair(stitched, images[i], debug_matches)

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
folder_path = 'dataset2'  # Ganti sesuai foldermu
result = stitch_images_from_folder(folder_path, debug_matches=False)

if result is not None:
    cv2.imwrite('stitched_orb_output_blended.jpg', result)
    cv2.imshow('Stitched Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
