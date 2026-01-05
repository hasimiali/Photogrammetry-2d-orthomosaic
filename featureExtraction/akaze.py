import cv2
import os

def draw_keypoints(image, keypoints, color=(0, 0, 0)):
    """Draw black dots on keypoints."""
    img_with_dots = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img_with_dots, (x, y), 4, color, -1)
    return img_with_dots

def extract_and_draw_three_akaze(image_path1, image_path2, image_path3, output_dir="featureExtraction/output_akaze_dots"):
    # Read images
    imgs = [cv2.imread(p) for p in [image_path1, image_path2, image_path3]]
    if any(img is None for img in imgs):
        print("Failed to read one or more images.")
        return

    # Initialize AKAZE detector
    akaze = cv2.AKAZE_create()

    # Detect keypoints and descriptors
    kps_des = [akaze.detectAndCompute(img, None) for img in imgs]
    kps_list, des_list = zip(*kps_des)

    # Print total features
    for i, kp in enumerate(kps_list, 1):
        print(f"Total features in image{i}: {len(kp)}")

    # BFMatcher for AKAZE (Hamming)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Pairwise matches
    pair_indices = [(0,1), (1,2), (0,2)]
    matched_kps = [[] for _ in range(3)]  # Store matched keypoints per image

    for idx1, idx2 in pair_indices:
        matches = bf.match(des_list[idx1], des_list[idx2])
        matches = sorted(matches, key=lambda x: x.distance)
        matched_kps[idx1].extend([kps_list[idx1][m.queryIdx] for m in matches])
        matched_kps[idx2].extend([kps_list[idx2][m.trainIdx] for m in matches])
        print(f"Matches between image{idx1+1} and image{idx2+1}: {len(matches)}")

    # Draw keypoints as black dots
    img_dots = [draw_keypoints(imgs[i], matched_kps[i]) for i in range(3)]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save output images
    for i, path in enumerate([image_path1, image_path2, image_path3]):
        base = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(output_dir, f"dots_{base}.jpg"), img_dots[i])
        print(f"Saved: dots_{base}.jpg in {output_dir}")

# === Example Usage ===
extract_and_draw_three_akaze(
    "featureExtraction/image1.jpg",
    "featureExtraction/image2.jpg",
    "featureExtraction/image3.jpg"
)
