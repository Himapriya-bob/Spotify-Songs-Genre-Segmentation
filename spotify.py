import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("spotify dataset.csv")
print("\n✅ Dataset Loaded")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

df = df.drop_duplicates().dropna()
features = [
    "danceability","energy","loudness","speechiness",
    "acousticness","instrumentalness","liveness",
    "valence","tempo","duration_ms","key","mode"
]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n✅ Data Preprocessed")
print("Features used:", features)

plt.figure(figsize=(10, 8))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Audio Features")
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3)
plt.title("PCA Projection of Songs")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

k = 6
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters
unique, counts = np.unique(clusters, return_counts=True)
print("\n✅ Cluster Distribution:")
for c, n in zip(unique, counts):
    print(f"  Cluster {c}: {n} songs")

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", alpha=0.6)
plt.title("KMeans Clusters (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Cluster")
plt.show()

def recommend(song_index, top_n=5):
    similarity = cosine_similarity([X_scaled[song_index]], X_scaled)[0]
    indices = similarity.argsort()[-top_n-1:-1][::-1]
    return df.iloc[indices][["track_name", "artist_name"]]

print("\n✅ Example Recommendations")
for i in range(3):
    song = df.iloc[i]["track_name"]
    artist = df.iloc[i]["artist_name"]
    print(f"\n🎵 Song: {song} by {artist}")
    recs = recommend(i)
    for idx, row in recs.iterrows():
        print(f"   → {row['track_name']} by {row['artist_name']}")
