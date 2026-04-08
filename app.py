import os
import chromadb
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Load .env if it exists, but priority goes to GitHub Secrets
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found! Please ensure it is added to GitHub Secrets and you have reloaded your Codespace.")

client = Groq(api_key=api_key)

# 1. Read PDF with basic error handling
def get_pdf_text(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# 2. Setup ChromaDB (Persistent)
# This creates a 'db' folder so it remembers your data
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("pdf_docs")

# Only process PDF if collection is empty
if collection.count() == 0:
    print("Indexing PDF... please wait.")
    raw_text = get_pdf_text("docs/White-Paper-LLM.pdf")
    
    # Split into chunks (500 chars with a small overlap for context)
    chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 450)]
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )
    print(f"Successfully indexed {len(chunks)} chunks.")
else:
    print("Using existing index from 'chroma_db' folder.")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Interactive Search and Answer
while True:
    question = input("\nAsk a question about the PDF (or type 'quit' to exit): ")
    if question.lower() in ['quit', 'exit']:
        break
        
    question_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3  # Increased to 3 for better context
    )

    relevant_chunks = "\n".join(results["documents"][0])

    # 4. Send to AI
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the user's question accurately. If the answer isn't in the context, say you don't know."},
                {"role": "user", "content": f"Context:\n{relevant_chunks}\n\nQuestion: {question}"}
            ]
        )
        print("\nAnswer:", response.choices[0].message.content)
    except Exception as e:
        print(f"Error calling Groq API: {e}")