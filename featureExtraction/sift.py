import cv2
import os

def draw_keypoints(image, keypoints, color=(0, 0, 0)):
    """Draw black dots on matched keypoints."""
    img_with_dots = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img_with_dots, (x, y), 4, color, -1)
    return img_with_dots

def extract_and_draw(image_path1, image_path2, output_dir="featureExtraction/output_sift_dots"):
    # Read both images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    if img1 is None or img2 is None:
        print("Failed to read one or both images.")
        return

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Print total number of features (keypoints)
    print(f"Total features in {os.path.basename(image_path1)}: {len(kp1)}")
    print(f"Total features in {os.path.basename(image_path2)}: {len(kp2)}")

    # Use BFMatcher for SIFT (L2 norm)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Get matched keypoints
    matched_kp1 = [kp1[m.queryIdx] for m in matches]
    matched_kp2 = [kp2[m.trainIdx] for m in matches]

    # Draw keypoints as black dots
    img1_dots = draw_keypoints(img1, matched_kp1)
    img2_dots = draw_keypoints(img2, matched_kp2)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save output images
    base1 = os.path.basename(image_path1)
    base2 = os.path.basename(image_path2)

    cv2.imwrite(os.path.join(output_dir, f"dots_{base1}"), img1_dots)
    cv2.imwrite(os.path.join(output_dir, f"dots_{base2}"), img2_dots)

    print(f"Saved: dots_{base1} and dots_{base2} in {output_dir}")

# === Example Usage ===
extract_and_draw("featureExtraction/image1.jpg", "featureExtraction/image2.jpg")
