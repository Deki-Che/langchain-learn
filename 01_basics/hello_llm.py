import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPEN_API_KEY")
if not api_key:
    print("ERROR")
    exit(1)
base_url = os.getenv("URL")
model_name = os.getenv("model_name")

llm = ChatOpenAI(
    model=model_name,
    temperature=0.7,
    base_url = base_url
)

prompt = "你好，给我讲个消化"
response = llm.invoke()

print(response.content)