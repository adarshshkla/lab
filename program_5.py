import gensim.downloader as api
import random

print("Loading model...")
model = api.load("glove-wiki-gigaword-50")

# 1. Get seed word
seed_word = input("\nEnter a seed word: ")

# 2. Get 5 similar words
try:
    similar_words = []
    for word, score in model.most_similar(seed_word, topn=5):
        similar_words.append(word)
except KeyError:
    print("Word not found. Try another.")
    exit()

# 3. Sentence Templates
templates = [
    f"The {seed_word} was surrounded by {similar_words[0]} and {similar_words[1]}.",
    f"People often associate {seed_word} with {similar_words[2]} and {similar_words[3]}.",
    f"In the land of {seed_word}, {similar_words[4]} was a common sight.",
    f"A story about {seed_word} would be incomplete without {similar_words[1]} and {similar_words[3]}."
]

# 4. Generate Paragraph (Pick 4 random sentences and join them)
paragraph = []
for _ in range(4):
    paragraph.append(random.choice(templates))

print("\nGenerated Paragraph:\n")
print(" ".join(paragraph))
