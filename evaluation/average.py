import pandas as pd

# Baca file CSV
df = pd.read_csv("evaluation/sift/100000_sift_evaluation.csv")

# Hitung rata-rata untuk semua kolom numerik
mean_values = df.mean(numeric_only=True)

# Tampilkan hasil
print("Rata-rata setiap kolom:\n")
print(mean_values)
