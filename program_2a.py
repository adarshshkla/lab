import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

print("Loading model...")
model = api.load("glove-wiki-gigaword-50")

# 1. Grab the words from Program 1
words = ["king", "man", "woman", "queen", "prince", "princess", "royal", "throne"]
vectors = []
for w in words:
    vectors.append(model[w])
vectors = np.array(vectors)


# ==========================================
# 2. Visualize using PCA
# ==========================================
print("\nDisplaying PCA Graph...")
pca = PCA(n_components=2)
pca_reduced = pca.fit_transform(vectors)

plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    x, y = pca_reduced[i, 0], pca_reduced[i, 1]
    plt.scatter(x, y, color='red')
    plt.text(x + 0.1, y + 0.1, word)

plt.title("Q1 Visualization (PCA)")
plt.show()


# ==========================================
# 3. Visualize using t-SNE
# ==========================================
print("\nDisplaying t-SNE Graph...")
tsne = TSNE(n_components=2, perplexity=3, random_state=42)
tsne_reduced = tsne.fit_transform(vectors)

plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    x, y = tsne_reduced[i, 0], tsne_reduced[i, 1]
    plt.scatter(x, y, color='blue')
    plt.text(x + 0.1, y + 0.1, word)

plt.title("Q1 Visualization (t-SNE)")
plt.show()
