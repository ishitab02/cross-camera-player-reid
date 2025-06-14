import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/matched_players.csv")
plt.hist(df["similarity"], bins=20, color='skyblue')
plt.axvline(0.9, color='red', linestyle='--', label="Threshold")
plt.title("Similarity Score Distribution")
plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Matches")
plt.legend()
plt.show()