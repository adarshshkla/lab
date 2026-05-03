import gensim.downloader as api
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore') # Hides annoying warnings during the exam

# 1. Load the Word Embedding Model
print("Loading Word2Vec model...")
word_model = api.load("glove-wiki-gigaword-50")

# 2. Find Similar Words to enrich our prompt
seed_word = "robot"
similar_words_data = word_model.most_similar(seed_word, topn=3)

# Extract just the words into a list
extra_words = []
for word, score in similar_words_data:
    extra_words.append(word)

# Join the list into a single comma-separated string
enriched_keywords = ", ".join(extra_words)

# 3. Create the two prompts
original_prompt = f"Tell me a short story about a {seed_word}."
enriched_prompt = f"Tell me a short story about a {seed_word}. Include: {enriched_keywords}."

print(f"\nOriginal Prompt: {original_prompt}")
print(f"Enriched Prompt: {enriched_prompt}")

# 4. Load Generative AI Model (GPT-2 is small, free, and doesn't need API keys)
print("\nLoading GPT-2 AI Model (This might take a minute)...")
generator = pipeline('text-generation', model='gpt2')

# 5. Generate Responses for comparison
print("\n========== ORIGINAL RESPONSE ==========")
output1 = generator(original_prompt, max_length=40, truncation=True)
print(output1[0]['generated_text'])

print("\n========== ENRICHED RESPONSE ==========")
output2 = generator(enriched_prompt, max_length=40, truncation=True)
print(output2[0]['generated_text'])
