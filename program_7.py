from transformers import pipeline

print("🔍 Loading Summarization Model (BART)...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    text = " ".join(text.split())

    print("\n📝 Original Text:")
    print(text)
    print("\n📌 Generating Summaries using different strategies...")

    summary_1 = summarizer(text, max_length=150, min_length=30, do_sample=False)

    summary_2 = summarizer(text, max_length=150, min_length=30, do_sample=True, temperature=0.9)

    summary_3 = summarizer(text, max_length=150, min_length=30, do_sample=False, num_beams=5)

    summary_4 = summarizer(text, max_length=150, min_length=30, do_sample=True, top_k=50, top_p=0.95)

    print("\n📌 Results:")
    print(f"Default: {summary_1[0]['summary_text']}")
    print(f"\nHigh Randomness: {summary_2[0]['summary_text']}")
    print(f"\nConservative (Beam Search): {summary_3[0]['summary_text']}")
    print(f"\nDiverse Sampling: {summary_4[0]['summary_text']}")
           
long_text = """
Artificial Intelligence (AI) is a rapidly evolving field of computer science focused on creating intelligent machines 
capable of mimicking human cognitive functions such as learning, problem-solving, and decision-making. 
In recent years, AI has significantly impacted various industries, including healthcare, finance, education, 
and entertainment. AI-powered applications, such as chatbots, self-driving cars, and recommendation systems, 
have transformed the way we interact with technology. Machine learning and deep learning, subsets of AI, 
enable systems to learn from data and improve over time without explicit programming. However, AI also poses ethical 
challenges, such as bias in decision-making and concerns over job displacement. As AI technology continues to advance, 
it is crucial to balance innovation with ethical considerations to ensure its responsible development and deployment.
"""

if __name__ == "__main__":
    summarize_text(long_text)