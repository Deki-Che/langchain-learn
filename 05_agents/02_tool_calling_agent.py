import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

print("=== Module 5: Advanced Agent with Structured Tools ===\n")

# 1. Define Tools using the @tool decorator
# This is the modern LangChain 1.0 way

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word. Input should be a single word."""
    return len(word)

@tool
def reverse_string(text: str) -> str:
    """Reverses a string. Input should be any text."""
    return text[::-1]

@tool
def count_vowels(text: str) -> int:
    """Counts the number of vowels in a text. Input should be any text."""
    vowels = "aeiouAEIOU"
    return sum(1 for char in text if char in vowels)

# 2. Collect tools
tools = [get_word_length, reverse_string, count_vowels]

# 3. Setup LLM
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")
llm = ChatOpenAI(model=model_name, base_url=base_url, temperature=0)

# 4. Create Prompt
# For tool-calling agents, we use a simpler prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools to answer questions about text."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 5. Create Agent
# create_tool_calling_agent uses the model's native tool-calling capability
agent = create_tool_calling_agent(llm, tools, prompt)

# 6. Create Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3
)

# 7. Test
print("=== Test: Multi-tool Usage ===")
result = agent_executor.invoke({
    "input": "For the word 'LangChain', tell me its length, reverse it, and count the vowels."
})
print(f"\nFinal Answer: {result['output']}\n")
