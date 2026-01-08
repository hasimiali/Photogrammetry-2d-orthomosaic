import pandas as pd

# Baca file CSV
df = pd.read_csv("evaluation/akaze/100000_akaze_evaluation.csv")

# Pilih kolom keypoint
keypoint_cols = ["KP Img1", "KP Img2"]

# Hitung statistik
stats = pd.DataFrame({
    "Min": df[keypoint_cols].min(),
    "Mean": df[keypoint_cols].mean(),
    "Median": df[keypoint_cols].median(),
    "Max": df[keypoint_cols].max()
})

# Tampilkan hasil
print("Statistik Keypoint:\n")
print(stats)
