import pandas as pd

# ================= LOAD CSV =================

files = {
    "SIFT": "evaluation_sift.csv",
    "ORB": "evaluation_orb.csv",
    "AKAZE": "evaluation_akaze.csv"
}

results = []

# ================= PROCESS =================

for method, path in files.items():
    df = pd.read_csv(path)

    summary = {
        "Method": method,
        "Avg KP Img1": df["KP Img1"].mean(),
        "Avg KP Img2": df["KP Img2"].mean(),
        "Avg Total Matches": df["Total Matches"].mean(),
        "Avg Good Matches": df["Good Matches"].mean(),
        "Avg Inliers": df["Inliers"].mean(),
        "Avg Reprojection Error": df["Reprojection Error"].mean(),
        "Avg Time (s)": df["Time (s)"].mean(),
        "Avg Memory (MB)": df["Memory (MB)"].mean()
    }

    results.append(summary)

# ================= SAVE RESULT =================

comparison_df = pd.DataFrame(results)
comparison_df.to_csv("comparison_feature_methods.csv", index=False)

print("\n=== COMPARISON SUMMARY ===")
print(comparison_df.round(2))
print("\nSaved as comparison_feature_methods.csv")
