import wikipedia
import getpass
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere
from langchain.output_parsers import PydanticOutputParser

# 1. Get User Input and Search Wikipedia
institution_name = input("Enter Institution Name: ")
print(f"Searching Wikipedia for '{institution_name}'...")
wiki_text = wikipedia.summary(institution_name, sentences=10)

# 2. Connect to the Cohere AI
api_key = getpass.getpass("Enter Cohere API Key: ")
llm = Cohere(cohere_api_key=api_key)

# 3. Define Pydantic Schema (The exact structure the manual asks for)
class InstitutionDetails(BaseModel):
    founder: str = Field(description="The founder of the Institution")
    founded_year: str = Field(description="When it was founded")
    branches: str = Field(description="Current branches in the institution")
    employees: str = Field(description="How many employees are working in it")
    summary: str = Field(description="A brief 4-line summary of the institution")

# 4. Create the Output Parser
parser = PydanticOutputParser(pydantic_object=InstitutionDetails)

# 5. Create the Template 
# Notice we inject the parser's instructions directly into the template
template = """
Extract information from the following text.
{format_instructions}

TEXT:
{text}
"""
prompt = PromptTemplate(
    input_variables=["text", "format_instructions"],
    template=template
)

# 6. Format the prompt and generate response
formatted_prompt = prompt.format(
    text=wiki_text, 
    format_instructions=parser.get_format_instructions()
)
raw_response = llm.predict(formatted_prompt)

# 7. Convert the raw AI text into a perfect Python object!
final_output = parser.parse(raw_response)

print("\n🎯 --- EXTRACTED DETAILS --- 🎯")
print(final_output.json(indent=2))
