import cv2
import numpy as np
import os

def auto_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return img[y:y+h, x:x+w]
    return img

def feather_blend(img1, img2, mask):
    # Make feather mask for smooth blending
    kernel = np.ones((31,31),np.uint8)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    dist_transform = cv2.GaussianBlur(dist_transform, (31,31), 0)

    dist_transform_3c = cv2.merge([dist_transform]*3)

    blended = img1 * dist_transform_3c + img2 * (1 - dist_transform_3c)
    return blended.astype(np.uint8)

def stitch_pair_akaze(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(gray1, None)
    kp2, des2 = akaze.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("Feature descriptors not found.")
        return img1

    # FLANN with LSH for binary descriptors
    index_params= dict(algorithm = 6,
                       table_number = 6,
                       key_size = 12,
                       multi_probe_level = 1)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"Good matches: {len(good)}")

    if len(good) < 20:
        print("Not enough good matches.")
        return img1

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

    result = cv2.warpPerspective(img1, trans_mat @ H, (output_width, output_height))
    result[translation[1]:translation[1]+h2, translation[0]:translation[0]+w2] = img2

    return result


def stitch_folder_akaze(folder):
    image_files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                          if f.lower().endswith(('.jpg','.jpeg','.png'))])

    if len(image_files) < 2:
        print("Need at least two images.")
        return None

    images = [cv2.imread(f) for f in image_files]
    pano = images[0]

    for i in range(1, len(images)):
        print(f"Stitching {i+1}/{len(images)}: {os.path.basename(image_files[i])}")
        pano = stitch_pair_akaze(pano, images[i])
        pano = auto_crop(pano)

    return pano

# Usage example
folder = "zoo_resized"  # your folder path
result = stitch_folder_akaze(folder)

if result is not None:
    cv2.imwrite("stitched_akaze_feather.jpg", result)
    cv2.imshow("AKAZE Stitching Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
