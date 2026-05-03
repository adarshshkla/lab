from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# 1. Medical Corpus
corpus = [
    "patient diagnosed with diabetes and hypertension",
    "treatment involves antibiotics and regular monitoring",
    "symptoms include fever fatigue and muscle pain",
    "vaccine is effective against several viral infections",
    "doctors recommend physical therapy for recovery"
]

# 2. Preprocess (Tokenize)
processed_corpus = []
for sentence in corpus:
    processed_corpus.append(sentence.lower().split())

# 3. Train Custom Word2Vec
print("Training Word2Vec model...")
model = Word2Vec(sentences=processed_corpus, vector_size=100, window=5, min_count=1, epochs=50)

# 4. Extract Embeddings
words = list(model.wv.index_to_key)
vectors = []
for w in words:
    vectors.append(model.wv[w])
vectors = np.array(vectors)

# 5. t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=3)
reduced = tsne.fit_transform(vectors)

plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    x, y = reduced[i, 0], reduced[i, 1]
    plt.scatter(x, y, color="blue")
    plt.text(x + 0.02, y + 0.02, word)

plt.title("Custom Medical Embeddings (t-SNE)")
plt.show()

# 6. Test Semantics
print("\nWords similar to 'treatment':")
for word, score in model.wv.most_similar("treatment", topn=3):
    print(f"- {word} ({score:.2f})")
