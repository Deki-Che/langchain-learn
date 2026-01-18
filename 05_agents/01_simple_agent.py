import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Load environment variables
load_dotenv()

print("=== Module 5: Agents (Simplified) ===\n")
print("Note: This is a simplified agent demonstration using tool binding.\n")

# 1. Define Tools using @tool decorator
@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression. Input should be like '2+2' or '10*5'."""
    try:
        result = eval(expression)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

# 2. Setup LLM with Tool Binding
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")
llm = ChatOpenAI(model=model_name, base_url=base_url, temperature=0)

# Bind tools to the model
# This tells the LLM what tools are available
tools = [calculator, get_word_length]
llm_with_tools = llm.bind_tools(tools)

print("=== How Agents Work ===")
print("1. User asks a question")
print("2. LLM decides if it needs to use a tool")
print("3. If yes, LLM returns a 'tool call' request")
print("4. We execute the tool and send the result back")
print("5. LLM uses the tool result to answer the question\n")

# 3. Simple Agent Loop (Manual)
def run_agent(question: str):
    print(f"\n📝 Question: {question}")
    print("-" * 60)
    
    messages = [HumanMessage(content=question)]
    
    # Step 1: Ask the LLM
    response = llm_with_tools.invoke(messages)
    print(f"🤖 LLM Response Type: {type(response).__name__}")
    
    # Check if LLM wants to use a tool
    if response.tool_calls:
        print(f"🔧 LLM wants to use tools: {[tc['name'] for tc in response.tool_calls]}")
        
        # Add LLM's response to messages
        messages.append(response)
        
        # Step 2: Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            print(f"   → Calling {tool_name} with args: {tool_args}")
            
            # Find and execute the tool
            selected_tool = {t.name: t for t in tools}[tool_name]
            tool_output = selected_tool.invoke(tool_args)
            
            print(f"   ← Tool returned: {tool_output}")
            
            # Add tool result to messages
            messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call['id']
            ))
        
        # Step 3: Ask LLM again with tool results
        final_response = llm_with_tools.invoke(messages)
        print(f"\n✅ Final Answer: {final_response.content}")
    else:
        # LLM answered directly without tools
        print(f"✅ Direct Answer: {response.content}")

# 4. Test Cases
print("\n" + "="*60)
print("TEST 1: Simple Calculation")
print("="*60)
run_agent("What is 15 multiplied by 7?")

print("\n" + "="*60)
print("TEST 2: Word Length")
print("="*60)
run_agent("How many letters are in the word 'LangChain'?")

print("\n" + "="*60)
print("TEST 3: No Tool Needed")
print("="*60)
run_agent("What is the capital of France?")
