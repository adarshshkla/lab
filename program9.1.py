import wikipedia, getpass
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere
from langchain.output_parsers import PydanticOutputParser
from IPython.display import display, clear_output
import ipywidgets as widgets

llm = Cohere(cohere_api_key=getpass.getpass("Enter Cohere API Key: "))

class Details(BaseModel):
    founder: str = Field(description="Founder name")
    year: str = Field(description="Founded year")
    branches: str = Field(description="Branches")
    employees: str = Field(description="Number of employees")
    summary: str = Field(description="Short summary")

parser = PydanticOutputParser(pydantic_object=Details)
prompt = PromptTemplate(
    input_variables=["text", "format_instructions"],
    template="Extract details.\n{format_instructions}\nTEXT:\n{text}"
)

def fetch_data(b):
    clear_output(wait=True)
    display(text_box, button)
    if text_box.value:
        wiki_text = wikipedia.summary(text_box.value, sentences=10)
        formatted = prompt.format(text=wiki_text, format_instructions=parser.get_format_instructions())
        result = parser.parse(llm.predict(formatted))
        print("\n🎯 Result:")
        print(result.json(indent=2))

text_box = widgets.Text(description='Institution:')
button = widgets.Button(description='Fetch Details', button_style='info')
button.on_click(fetch_data)

display(text_box, button)