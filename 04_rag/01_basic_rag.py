import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

print("=== Step 1: Load Documents ===")
# Load the sample document
loader = TextLoader("04_rag/sample_docs.txt", encoding="utf-8")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")
print(f"First 200 chars: {documents[0].page_content[:200]}...")

print("\n=== Step 2: Split Documents ===")
# Split the document into smaller chunks
# This is important because:
# 1. Embeddings work better on smaller, focused text
# 2. We can retrieve only the most relevant parts
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Maximum characters per chunk
    chunk_overlap=50  # Overlap to maintain context between chunks
)
splits = text_splitter.split_documents(documents)
print(f"Split into {len(splits)} chunks")

print("\n=== Step 3: Create Embeddings & Vector Store ===")
# Embeddings convert text into numerical vectors
# Similar texts will have similar vectors
base_url = os.getenv("OPENAI_BASE_URL")
embeddings = OpenAIEmbeddings(base_url=base_url)

# FAISS is a fast vector database (runs locally, no server needed)
vectorstore = FAISS.from_documents(splits, embeddings)
print("Vector store created successfully")

# Create a retriever (this will search for relevant chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Return top 2 results

print("\n=== Step 4: Create RAG Chain ===")
# Setup the model
model_name = os.getenv("OPENAI_MODEL_NAME")
model = ChatOpenAI(model=model_name, base_url=base_url, temperature=0)

# Create the prompt template
# {context} will be filled with retrieved documents
# {question} will be the user's question
template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG chain using LCEL
# The chain flow:
# 1. Input: {"question": "..."}
# 2. Retriever finds relevant docs
# 3. Format docs into context string
# 4. Pass context + question to prompt
# 5. Send to model
# 6. Parse output
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print("\n=== Step 5: Ask Questions ===")
# Now we can ask questions about our document!
questions = [
    "What is LCEL?",
    "When was LangChain released?",
    "What are the use cases for LangChain?"
]

for q in questions:
    print(f"\n📝 Question: {q}")
    answer = rag_chain.invoke(q)
    print(f"🤖 Answer: {answer}")
