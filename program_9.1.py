import wikipedia, getpass
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere
from langchain.output_parsers import PydanticOutputParser
from IPython.display import display, clear_output
import ipywidgets as widgets

# 1. Connect AI
llm = Cohere(cohere_api_key=getpass.getpass("Enter Cohere API Key: "))

# 2. Pydantic Schema & Parser
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

# 3. UI and Core Logic
def fetch_data(b):
    clear_output(wait=True)
    display(text_box, button) # Keep UI visible
    
    if text_box.value:
        # Search Wiki and run AI
        wiki_text = wikipedia.summary(text_box.value, sentences=10)
        formatted = prompt.format(text=wiki_text, format_instructions=parser.get_format_instructions())
        result = parser.parse(llm.predict(formatted))
        
        # Print the beautiful JSON!
        print("\n🎯 Result:")
        print(result.json(indent=2))

# 4. Build and Display Widgets
text_box = widgets.Text(description='Institution:')
button = widgets.Button(description='Fetch Details', button_style='info')
button.on_click(fetch_data)

display(text_box, button)
