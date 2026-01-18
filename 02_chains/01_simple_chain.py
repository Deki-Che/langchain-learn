import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# 1. Initialize the Model (using the working OpenRouter config)
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")

model = ChatOpenAI(
    model=model_name,
    base_url=base_url,
    temperature=0
)

# 2. Define the Prompt Template
prompt = ChatPromptTemplate.from_template("Tell me a short fact about {topic},use Chinese")

# 3. Define the Output Parser
# StrOutputParser converts the AIMessage object from the model into a simple string.
output_parser = StrOutputParser()

# 4. Create the Chain using LCEL (LangChain Expression Language)
# The '|' symbol (pipe) connects these components together.
# Data flows from left to right:
# Input Dictionary -> Prompt -> Model -> Output Parser -> Final String
chain = prompt | model | output_parser

# 5. Invoke the Chain
topic = "Space Exploration"
print(f"--- Running Chain for topic: {topic} ---")

# We pass the input variables as a dictionary
response = chain.invoke({"topic": topic})

print("\n--- Final Output (String) ---")
print(response)
