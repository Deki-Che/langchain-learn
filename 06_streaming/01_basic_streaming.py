import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

print("=== Module 6: Streaming Output ===\n")
print("Streaming allows you to see the AI's response in real-time,")
print("token by token, just like ChatGPT!\n")

# Setup Model
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")
llm = ChatOpenAI(model=model_name, base_url=base_url, temperature=0.7)

# --- Compare: invoke() vs stream() ---

print("=" * 60)
print("Method 1: Using .invoke() (waits for complete response)")
print("=" * 60)

prompt = "Write a short poem about Python programming in 4 lines."

print(f"\nPrompt: {prompt}")
print("\nWaiting for complete response...")

# invoke() waits for the entire response before returning
response = llm.invoke(prompt)
print(f"\nComplete Response:\n{response.content}")

print("\n" + "=" * 60)
print("Method 2: Using .stream() (real-time token by token)")
print("=" * 60)

print(f"\nPrompt: {prompt}")
print("\nStreaming response:")

# stream() yields chunks as they become available
for chunk in llm.stream(prompt):
    # Each chunk contains a small piece of the response
    # chunk.content is the text fragment
    print(chunk.content, end="", flush=True)

print("\n\n" + "=" * 60)
print("Key Differences:")
print("=" * 60)
print("• invoke(): Blocks until complete response is ready")
print("• stream(): Yields chunks immediately as they're generated")
print("• stream() provides better UX for long responses")
print("• Both return the same final content")
