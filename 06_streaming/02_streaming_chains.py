import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

print("=== Module 6: Streaming with LCEL Chains ===\n")

# Setup
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")
llm = ChatOpenAI(model=model_name, base_url=base_url, temperature=0.7)

# Create a chain
prompt = ChatPromptTemplate.from_template(
    "You are a storyteller. Write a very short story (3-4 sentences) about {topic}."
)
chain = prompt | llm | StrOutputParser()

print("=" * 60)
print("Streaming a Chain")
print("=" * 60)

topic = "a robot learning to paint"
print(f"\nTopic: {topic}")
print("\nStreaming story:\n")

# Stream the entire chain!
# LCEL chains automatically support streaming
for chunk in chain.stream({"topic": topic}):
    print(chunk, end="", flush=True)

print("\n\n" + "=" * 60)
print("Understanding Stream Chunks")
print("=" * 60)

print("\nLet's examine what each chunk looks like:\n")

for i, chunk in enumerate(chain.stream({"topic": "a cat in space"})):
    # Show first 5 chunks with their content
    if i < 5:
        print(f"Chunk {i}: '{chunk}' (length: {len(chunk)})")
    elif i == 5:
        print("... (more chunks)")
        # Continue streaming but don't print
    # Print the rest without annotations
    else:
        pass

print("\n" + "=" * 60)
print("Key Points:")
print("=" * 60)
print("• Each chunk is a small piece of text (often 1-3 tokens)")
print("• Chunks are yielded as soon as they're generated")
print("• StrOutputParser preserves streaming behavior")
print("• The pipe (|) operator automatically handles streaming")
