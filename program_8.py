import getpass
from google.colab import auth
from google.colab import drive
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere

# 1. Authenticate and Mount Google Drive
auth.authenticate_user()
drive.mount('/content/drive')

# 2. Load the text file from Google Drive
file_path = "/content/drive/My Drive/Teaching.txt"
with open(file_path, "r") as file:
    text_content = file.read()

# 3. Connect to the Cohere AI
api_key = getpass.getpass("Enter Cohere API Key: ")
llm = Cohere(cohere_api_key=api_key)

# 4. Create the Template for the AI
template = """
Analyze this document:
{text_content}

📝 Summary:
📌 Key Takeaways:
📊 Sentiment Analysis (Positive/Negative/Neutral):
"""
prompt = PromptTemplate(input_variables=["text_content"], template=template)

# 5. Inject the text into the template and get prediction
formatted_prompt = prompt.format(text_content=text_content)
response = llm.predict(formatted_prompt)

# 6. Print the result
print(response)
