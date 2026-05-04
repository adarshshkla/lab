import getpass
from google.colab import auth
from google.colab import drive
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere

auth.authenticate_user()
drive.mount('/content/drive')

file_path = "/content/drive/My Drive/Teaching.txt"
with open(file_path, "r") as file:
    text_content = file.read()

api_key = getpass.getpass("Enter Cohere API Key: ")
llm = Cohere(cohere_api_key=api_key)

template = """
Analyze this document:
{text_content}

📝 Summary:
📌 Key Takeaways:
📊 Sentiment Analysis (Positive/Negative/Neutral):
"""
prompt = PromptTemplate(input_variables=["text_content"], template=template)

formatted_prompt = prompt.format(text_content=text_content)
response = llm.predict(formatted_prompt)

print(response)