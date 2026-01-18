import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load env variables
load_dotenv()

# Initialize LLM
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

llm = ChatOpenAI(
    model=model_name,
    temperature=0.7,
    base_url=base_url
)

# --- Define a Prompt Template ---
# Templates allow us to create flexible prompts with variables.
# We use a list of messages (tuples or objects) to define the system and user roles.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("user", "{text}")
])

# --- Create the messages by invoking the template ---
# This does NOT call the LLM yet. It just formats the string.
messages = prompt_template.invoke({
    "input_language": "English",
    "output_language": "Chinese",
    "text": "I love programming."
})

print("--- Formatted Messages ---")
print(messages)

# --- Send to LLM ---
print("\n--- Sending to LLM ---")
response = llm.invoke(messages)
print(response.content)

# --- Using a Chain (Preview) ---
# In modern LangChain, we often pipe the prompt to the model directly.
# chain = prompt_template | llm
# response = chain.invoke({...})
