import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

print("=== Module 7: Multi-Query Retrieval ===\n")

# Setup
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")
llm = ChatOpenAI(model=model_name, base_url=base_url, temperature=0)

# Load documents (reuse from Module 4)
print("Loading and preparing documents...")
loader = TextLoader("04_rag/sample_docs.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(base_url=base_url)
vectorstore = FAISS.from_documents(splits, embeddings)

print("\n" + "=" * 60)
print("Problem with Basic RAG")
print("=" * 60)
print("""
When a user asks: "What are the benefits of LangChain?"

Basic RAG:
- Searches for documents similar to this exact question
- Might miss documents that talk about "advantages" or "features"
- Limited to one perspective

Solution: Multi-Query Retrieval
- LLM generates multiple variations of the question
- Each variation searches the vector store
- Results are combined and deduplicated
""")

print("\n" + "=" * 60)
print("Creating Multi-Query Retriever")
print("=" * 60)

# Create a multi-query retriever
# This will automatically generate multiple queries from the user's question
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    llm=llm
)

# Test query
question = "What are the key features of LangChain?"
print(f"\nOriginal Question: {question}")
print("\nThe LLM will generate variations like:")
print("  - What functionalities does LangChain provide?")
print("  - What are LangChain's main capabilities?")
print("  - What can you do with LangChain?")

print("\nRetrieving documents...")
try:
    docs = retriever.get_relevant_documents(question)
    print(f"\nFound {len(docs)} unique documents (after deduplication)")
    
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(doc.page_content[:200] + "...")
        
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Benefits of Multi-Query Retrieval")
print("=" * 60)
print("""
✅ Better recall: Catches documents that use different terminology
✅ More robust: Less sensitive to how the user phrases the question
✅ Automatic: No manual query engineering needed
✅ Deduplication: Removes duplicate results

Trade-offs:
⚠️ More API calls (one for query generation + multiple searches)
⚠️ Slightly slower
""")
