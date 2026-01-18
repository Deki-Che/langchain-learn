import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()

print("=== Module 7: Contextual Compression ===\n")

# Setup
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")
llm = ChatOpenAI(model=model_name, base_url=base_url, temperature=0)

# Load documents
print("Loading documents...")
loader = TextLoader("04_rag/sample_docs.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(base_url=base_url)
vectorstore = FAISS.from_documents(splits, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("\n" + "=" * 60)
print("Problem with Basic Retrieval")
print("=" * 60)
print("""
When retrieving documents, we often get:
- Entire chunks (500 characters)
- But only 50-100 characters are actually relevant
- The rest is noise that confuses the LLM

Example:
Question: "When was LangChain released?"
Retrieved chunk might contain:
  "LangChain is a framework... [200 chars of features]...
   The framework was released in October 2022... [more text]"
   
Only "released in October 2022" is relevant!
""")

print("\n" + "=" * 60)
print("Solution: Contextual Compression")
print("=" * 60)
print("""
How it works:
1. Retrieve documents normally
2. For each document, ask LLM: "Extract only the parts relevant to the question"
3. Return compressed documents
4. Send to final LLM for answer generation

Result: Less noise, more signal!
""")

# Create compressor
compressor = LLMChainExtractor.from_llm(llm)

# Wrap the base retriever with compression
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Test
question = "When was LangChain released?"
print(f"\nQuestion: {question}\n")

print("--- Without Compression ---")
try:
    normal_docs = base_retriever.get_relevant_documents(question)
    for i, doc in enumerate(normal_docs):
        print(f"\nDocument {i+1} ({len(doc.page_content)} chars):")
        print(doc.page_content[:150] + "...")
except Exception as e:
    print(f"Error: {e}")

print("\n--- With Compression ---")
try:
    compressed_docs = compression_retriever.get_relevant_documents(question)
    for i, doc in enumerate(compressed_docs):
        print(f"\nDocument {i+1} ({len(doc.page_content)} chars):")
        print(doc.page_content)
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Benefits")
print("=" * 60)
print("""
✅ Reduces token usage (shorter context)
✅ Improves answer quality (less noise)
✅ Faster generation (less to process)
✅ More focused responses

Trade-offs:
⚠️ Extra LLM call for compression
⚠️ Might accidentally filter out important context
""")
