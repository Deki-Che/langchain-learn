import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()

# 1. Setup Model
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")
model = ChatOpenAI(model=model_name, base_url=base_url, temperature=0.7)

# 2. Setup Prompt
# The prompt must have a MessagesPlaceholder to hold the history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer questions concisely."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

# 3. Create the Base Chain
chain = prompt | model | StrOutputParser()

# 4. Memory Management
# In a real app, this would be a database. Here we use an in-memory dictionary.
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 5. Wrap the Chain with History Management
# This is the "Modern LangChain" way to handle memory.
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 6. First Interaction
session_id = "user_123"
print(f"--- Session: {session_id} ---")

print("\n> User: My name is Alice.")
res1 = chain_with_history.invoke(
    {"input": "Hi, my name is Alice."},
    config={"configurable": {"session_id": session_id}}
)
print(f"AI: {res1}")

# 7. Second Interaction (Checking Memory)
print("\n> User: What is my name?")
res2 = chain_with_history.invoke(
    {"input": "What is my name?"},
    config={"configurable": {"session_id": session_id}}
)
print(f"AI: {res2}")
