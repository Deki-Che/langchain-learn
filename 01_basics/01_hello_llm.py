import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Check if API key is present
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not found in environment variables. Please create a .env file.")
    exit(1)

# Initialize the Model
# We read base_url and model from env to support OpenRouter or other providers
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

print(f"--- Configuration ---")
print(f"Model: {model_name}")
print(f"Base URL: {base_url if base_url else 'Default (OpenAI)'}")
print(f"API Key: {api_key[:8]}...****")

llm = ChatOpenAI(
    model=model_name,
    temperature=0.7,
    base_url=base_url
)

# Create a message
# In LangChain, we often just pass string prompts to the model directly for simple cases,
# or use message objects for more structure.
prompt = "Hello, how are you today? 给我讲个冷笑话"

print("--- Sending Request to LLM ---")
response = llm.invoke(prompt)

print("\n--- Response ---")
# The response is an AIMessage object. We can access the content directly.
print(response.content)
