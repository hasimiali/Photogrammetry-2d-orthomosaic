
import os
import subprocess
import cv2
import numpy as np
from glob import glob

def run_colmap(image_dir, colmap_dir):
    os.makedirs(colmap_dir, exist_ok=True)

    # Feature extraction
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", f"{colmap_dir}/database.db",
        "--image_path", image_dir
    ])

    # Feature matching
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", f"{colmap_dir}/database.db"
    ])

    # Sparse reconstruction
    sparse_dir = os.path.join(colmap_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    subprocess.run([
        "colmap", "mapper",
        "--database_path", f"{colmap_dir}/database.db",
        "--image_path", image_dir,
        "--output_path", sparse_dir
    ])

    print("COLMAP sparse reconstruction done.")

def run_openmvs(colmap_sparse_dir, mvs_dir):
    os.makedirs(mvs_dir, exist_ok=True)
    model_path = os.path.join(colmap_sparse_dir, "0")
    scene_mvs = os.path.join(mvs_dir, "scene.mvs")

    # Convert to OpenMVS format
    subprocess.run([
        "colmap", "model_converter",
        "--input_path", model_path,
        "--output_path", scene_mvs,
        "--output_type", "OpenMVS"
    ])

    # Densify
    subprocess.run(["DensifyPointCloud", scene_mvs])
    # Reconstruct mesh
    subprocess.run(["ReconstructMesh", "scene_dense.mvs"])
    # Texture
    subprocess.run(["TextureMesh", "scene_dense_mesh.mvs"])

    print("OpenMVS reconstruction complete.")

def create_naive_orthomosaic(image_folder, output_file="naive_orthomosaic.jpg"):
    images = []
    for filename in sorted(glob(os.path.join(image_folder, '*.jpg'))):
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)

    if not images:
        print("No images found for orthomosaic creation.")
        return

    h, w, _ = images[0].shape
    canvas = np.zeros((h, w * len(images), 3), dtype=np.uint8)

    for i, img in enumerate(images):
        canvas[:, i*w:(i+1)*w] = img

    cv2.imwrite(output_file, canvas)
    print(f"Naive orthomosaic saved to {output_file}")

if __name__ == "__main__":
    image_dir = "dataset2"
    colmap_output = "colmap_output"
    mvs_output = "mvs_output"

    run_colmap(image_dir, colmap_output)
    run_openmvs(os.path.join(colmap_output, "sparse"), mvs_output)
    create_naive_orthomosaic(image_dir)
