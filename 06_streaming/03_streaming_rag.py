import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load environment variables
load_dotenv()

print("=== Module 6: Streaming RAG ===\n")

# Setup
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")
llm = ChatOpenAI(model=model_name, base_url=base_url, temperature=0)

# Load and prepare documents (reuse from Module 4)
print("Loading documents...")
loader = TextLoader("04_rag/sample_docs.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

print("Creating vector store...")
embeddings = OpenAIEmbeddings(base_url=base_url)
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Create RAG prompt
template = """Answer the question based on the following context. 
Be detailed and provide a complete answer.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n" + "=" * 60)
print("Streaming RAG Response")
print("=" * 60)

question = "What is LCEL and what are its key features?"
print(f"\nQuestion: {question}")
print("\nStreaming answer:\n")

# Stream the RAG response!
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)

print("\n\n" + "=" * 60)
print("How Streaming RAG Works")
print("=" * 60)
print("""
1. Question comes in
2. Retriever fetches relevant documents (not streamed)
3. Context is formatted and combined with question
4. Prompt is sent to LLM
5. LLM response is STREAMED token by token
6. User sees answer appearing in real-time

Note: The retrieval step is NOT streamed - it happens all at once.
Only the LLM generation is streamed.
""")

print("=" * 60)
print("Practical Use Cases")
print("=" * 60)
print("""
• Chatbots: Show responses as they're generated
• Document Q&A: Display answers progressively
• Long-form content: Keep users engaged while generating
• Real-time translation: Stream translated text
""")
