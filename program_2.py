import gensim.downloader as api
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

print("Loading model...")
model = api.load("glove-wiki-gigaword-50")

# 1. Generate 5 similar words
print("Words similar to 'technology':")
for word, score in model.most_similar("technology", topn=5):
    print(f"- {word}")

# 2. Select 10 words from specific domains
words = ["computer", "laptop", "software", "internet", "keyboard", 
         "football", "tennis", "basketball", "cricket", "baseball"]

# 3. Get their vectors 
vectors = []
for w in words:
    vectors.append(model[w])

# 4. Reduce 50 dimensions to 2 using t-SNE
tsne = TSNE(n_components=2, perplexity=3, random_state=42)
two_d_vectors = tsne.fit_transform(np.array(vectors))

# 5. Visualize
for i, word in enumerate(words):
    x = two_d_vectors[i, 0]
    y = two_d_vectors[i, 1]
    plt.scatter(x, y, color='blue')
    plt.text(x + 0.1, y + 0.1, word)

plt.show()
