import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

print("Loading model...")
model = api.load("glove-wiki-gigaword-50")

# --- PART 2B: Domain Words & Synonyms ---
domain_words = ["computer", "software", "hardware", "algorithm", "data", 
                "network", "programming", "machine", "learning", "artificial"]

vectors = []
for w in domain_words:
    vectors.append(model[w])
vectors = np.array(vectors)

print("Displaying t-SNE Graph for Technology Domain...")
# Squash 50 dimensions down to 2
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
reduced = tsne.fit_transform(vectors)

# Draw the Graph
plt.figure(figsize=(8, 6))
for i, word in enumerate(domain_words):
    x, y = reduced[i, 0], reduced[i, 1]
    plt.scatter(x, y, color='blue')
    plt.text(x + 0.1, y + 0.1, word)

plt.title("Technology Domain Embeddings (t-SNE)")
plt.show()

# Generate 5 semantically similar words
print("\n5 words semantically similar to 'computer':")
for word, score in model.most_similar("computer", topn=5):
    print(f"- {word} ({score:.4f})")
