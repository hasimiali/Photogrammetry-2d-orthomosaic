import cv2
import os

def visualize_matches(image_path1, image_path2, output_dir="featureMatching/sift"):
    # Read images
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

    print(f"Total features in {os.path.basename(image_path1)}: {len(kp1)}")
    print(f"Total features in {os.path.basename(image_path2)}: {len(kp2)}")

    # BFMatcher with L2 norm
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"Number of matches: {len(matches)}")

    # Draw top N matches (e.g., 50)
    top_matches = matches[:50]
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, top_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save the visualization
    base1 = os.path.splitext(os.path.basename(image_path1))[0]
    base2 = os.path.splitext(os.path.basename(image_path2))[0]
    output_path = os.path.join(output_dir, f"matches_{base1}_{base2}.jpg")
    cv2.imwrite(output_path, img_matches)
    print(f"Match visualization saved at: {output_path}")

    # Optional: show the image
    cv2.imshow("Feature Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === Example Usage ===
visualize_matches(
    "featureMatching/image1.jpg",
    "featureMatching/image2.jpg"
)
