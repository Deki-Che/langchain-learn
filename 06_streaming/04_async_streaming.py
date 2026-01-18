import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

print("=== Module 6: Async Streaming ===\n")
print("Async streaming is essential for web applications!")
print("It allows handling multiple requests concurrently.\n")

# Setup
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")
llm = ChatOpenAI(model=model_name, base_url=base_url, temperature=0.7)

prompt = ChatPromptTemplate.from_template(
    "Give me 3 fun facts about {topic}. Keep it brief."
)
chain = prompt | llm | StrOutputParser()


async def stream_response(topic: str):
    """Stream a single response asynchronously."""
    print(f"\n📝 Topic: {topic}")
    print("   Response: ", end="")
    
    # Use .astream() for async streaming
    async for chunk in chain.astream({"topic": topic}):
        print(chunk, end="", flush=True)
    
    print()  # Newline after stream completes


async def parallel_streams():
    """Stream multiple responses in parallel."""
    topics = ["cats", "space", "coffee"]
    
    print("=" * 60)
    print("Parallel Async Streaming (3 topics at once)")
    print("=" * 60)
    
    # Create tasks for parallel execution
    tasks = [stream_response(topic) for topic in topics]
    
    # Run all streams concurrently
    await asyncio.gather(*tasks)


async def sequential_stream():
    """Stream responses one by one."""
    print("=" * 60)
    print("Sequential Async Streaming")
    print("=" * 60)
    
    topics = ["Python", "AI"]
    
    for topic in topics:
        await stream_response(topic)


async def main():
    # Run sequential streaming first
    await sequential_stream()
    
    print("\n")
    
    # Then run parallel streaming
    await parallel_streams()
    
    print("\n" + "=" * 60)
    print("Key Async Methods")
    print("=" * 60)
    print("""
    • .astream()  - Async version of .stream()
    • .ainvoke()  - Async version of .invoke()
    • .abatch()   - Async version of .batch()
    
    Use these in:
    • FastAPI endpoints
    • Async web frameworks
    • High-concurrency applications
    """)


if __name__ == "__main__":
    asyncio.run(main())
