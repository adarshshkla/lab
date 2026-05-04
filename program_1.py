import gensim.downloader as api

print("Loading model...")
model = api.load("glove-wiki-gigaword-50")

try:
    sim = model.similarity("cat", "dog")
    print(f"Similarity (cat vs dog): {sim:.4f}")

    print("\nTop 5 similar to 'happy':")
    for word, score in model.most_similar("happy", topn=5):
        print(f"- {word}: {score:.4f}")

    result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
    print(f"\nKing - Man + Woman = {result[0][0]} (Score: {result[0][1]:.4f})")

except KeyError as e:
    print(f"Error: Word not found - {e}")