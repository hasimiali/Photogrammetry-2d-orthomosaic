import cv2
import os

def average_orb_flann_matching_sequential(
    image_folder,
    ratio_thresh=0.75,
    max_features=100000,
    visualize=False,
    output_dir="featureMatching/orb_flann"
):
    # ================= ORB INITIALIZATION =================
    orb = cv2.ORB_create(nfeatures=max_features)

    # ================= FLANN PARAM (LSH for ORB) =================
    index_params = dict(
        algorithm=6,      # FLANN_INDEX_LSH
        table_number=12,  # biasanya 6–12
        key_size=20,      # biasanya 12–20
        multi_probe_level=2
    )
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Load & sort images
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ])

    if len(image_files) < 2:
        print("Minimal butuh 2 gambar.")
        return

    images = {}
    keypoints = {}
    descriptors = {}

    # ================= FEATURE EXTRACTION =================
    for file in image_files:
        img_path = os.path.join(image_folder, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Gagal membaca {file}")
            continue

        kp, des = orb.detectAndCompute(img, None)

        if des is None or len(kp) == 0:
            print(f"Tidak ada feature di {file}")
            continue

        # FLANN butuh dtype uint8 untuk ORB
        des = des.astype("uint8")

        images[file] = img
        keypoints[file] = kp
        descriptors[file] = des

        print(f"{file}: {len(kp)} features")

    valid_files = list(images.keys())

    if len(valid_files) < 2:
        print("Tidak cukup gambar valid untuk matching.")
        return

    total_raw_matches = 0
    total_good_matches = 0
    pair_count = 0

    os.makedirs(output_dir, exist_ok=True)

    # ================= SEQUENTIAL MATCHING =================
    for i in range(len(valid_files) - 1):
        img1_name = valid_files[i]
        img2_name = valid_files[i + 1]

        des1 = descriptors[img1_name]
        des2 = descriptors[img2_name]

        # KNN Match
        matches = flann.knnMatch(des1, des2, k=2)
        raw_matches = len(matches)

        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        total_raw_matches += raw_matches
        total_good_matches += len(good_matches)
        pair_count += 1

        print(
            f"{img1_name} → {img2_name} | "
            f"Raw: {raw_matches}, Good: {len(good_matches)}"
        )

        # Optional visualization
        if visualize:
            img_match = cv2.drawMatches(
                images[img1_name], keypoints[img1_name],
                images[img2_name], keypoints[img2_name],
                good_matches[:50],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            out_name = f"match_{img1_name}_to_{img2_name}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), img_match)

    # ================= AVERAGE RESULT =================
    avg_raw = total_raw_matches / pair_count
    avg_good = total_good_matches / pair_count

    print("\n==============================")
    print(f"Total pasangan (sequential): {pair_count}")
    print(f"Rata-rata Raw Match        : {avg_raw:.2f}")
    print(f"Rata-rata Good Match       : {avg_good:.2f}")
    print("==============================")

    return avg_raw, avg_good


# === Example Usage ===
average_orb_flann_matching_sequential(
    image_folder="dataset2",
    ratio_thresh=0.75,
    visualize=False
)
