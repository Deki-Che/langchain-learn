import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

base_url = os.getenv("url")
model_name = os.getenv("model_name")

llm = ChatOpenAI(
    model=model_name,
    temperature=0.7,
    base_url=base_url
)

prompt_template2 = ChatPromptTemplate.from_messages([
    ("system", "dddd, {input} to {output}"),
    ("user", "{text}")
])

messages2 = prompt_template2.invoke({
    "input":"e", 
    "output":"chi",
    "text": " dddd"
})

response = llm.invoke(messages2)
print(response.content)