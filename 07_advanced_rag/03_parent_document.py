import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

load_dotenv()

print("=== Module 7: Parent Document Retriever ===\n")

# Setup
base_url = os.getenv("OPENAI_BASE_URL")
embeddings = OpenAIEmbeddings(base_url=base_url)

# Load documents
print("Loading documents...")
loader = TextLoader("04_rag/sample_docs.txt", encoding="utf-8")
documents = loader.load()

print("\n" + "=" * 60)
print("The Problem with Fixed-Size Chunks")
print("=" * 60)
print("""
Dilemma:
- Small chunks (100 chars): Good for precise retrieval, but lack context
- Large chunks (1000 chars): Good context, but poor retrieval precision

Example:
Small chunk: "LCEL uses the pipe operator"
  ✅ Easy to find
  ❌ Missing: What is LCEL? What does the pipe do?

Large chunk: "LangChain overview... [500 chars]... LCEL uses pipe... [500 chars]"
  ❌ Hard to match query
  ✅ Has full context

Can we have both? YES!
""")

print("\n" + "=" * 60)
print("Solution: Parent Document Retriever")
print("=" * 60)
print("""
Strategy:
1. Split documents into SMALL chunks for embedding/search
2. Store LARGE parent documents separately
3. When searching:
   - Search using small chunks (precise matching)
   - Return the full parent document (complete context)

Best of both worlds!
""")

# Create child splitter (small chunks for search)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# Create parent splitter (large chunks to return)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

# Storage for parent documents
store = InMemoryStore()

# Create vector store for child chunks
vectorstore = FAISS.from_documents([], embeddings)

# Create Parent Document Retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents
print("\nIndexing documents...")
retriever.add_documents(documents)

print(f"Created {len(store.store)} parent documents")
print(f"Created {vectorstore.index.ntotal} child chunks for searching")

# Test
question = "What is LCEL?"
print(f"\nQuestion: {question}\n")

try:
    docs = retriever.get_relevant_documents(question)
    
    print(f"Retrieved {len(docs)} parent document(s):\n")
    for i, doc in enumerate(docs):
        print(f"--- Parent Document {i+1} ({len(doc.page_content)} chars) ---")
        print(doc.page_content[:300] + "...\n")
        
except Exception as e:
    print(f"Error: {e}")

print("=" * 60)
print("How It Works")
print("=" * 60)
print("""
1. User asks: "What is LCEL?"
2. System searches SMALL chunks (200 chars each)
3. Finds match: "LCEL uses the pipe operator |"
4. Returns the LARGE parent (800 chars) containing full context
5. LLM gets complete information to generate answer

Benefits:
✅ Precise retrieval (small chunks match well)
✅ Rich context (large parents provide details)
✅ No information loss
✅ Better answer quality

Trade-offs:
⚠️ More storage (both small and large chunks)
⚠️ More complex setup
""")
