import wikipedia, getpass
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import ipywidgets as widgets
from IPython.display import display, clear_output

# 1. Connect to AI
api_key = getpass.getpass("Enter Cohere API Key: ")
llm = Cohere(cohere_api_key=api_key)

# 2. Download IPC from Wikipedia
print("⚖️ Downloading IPC from Wikipedia...")
try:
    ipc_content = wikipedia.page("Indian Penal Code").text[:5000]
except:
    ipc_content = "IPC Document not found."

# 3. Pydantic Model & Output Parser
class IPCResponse(BaseModel):
    section: str = Field(description="The relevant section number (or N/A)")
    explanation: str = Field(description="The detailed legal explanation")

parser = PydanticOutputParser(pydantic_object=IPCResponse)

# 4. Prompt Template
template = """
You are an IPC legal assistant. Answer the user's question using this content:
{content}
{format_instructions}

Question: {question}
"""
prompt = PromptTemplate(
    input_variables=["content", "question", "format_instructions"], 
    template=template
)

# 5. Core Chatbot Logic
def on_click(b):
    clear_output(wait=True); display(text_box, button)
    if not text_box.value: return
    
    # Format prompt and ask AI
    formatted_prompt = prompt.format(
        content=ipc_content, 
        question=text_box.value,
        format_instructions=parser.get_format_instructions()
    )
    raw_response = llm.predict(formatted_prompt)
    
    # Let the parser magically handle all the formatting!
    final_data = parser.parse(raw_response)
    print("\n🎯 Response:\n", final_data.json(indent=2))

# 6. Build Interactive GUI
text_box = widgets.Text(description='You:', placeholder='Ask about IPC...')
button = widgets.Button(description='Ask', button_style='info')
button.on_click(on_click)

display(text_box, button)
