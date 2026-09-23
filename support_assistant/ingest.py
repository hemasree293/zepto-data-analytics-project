from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Folder containing the 8 Zepto policy documents
DOCS_DIR = Path(__file__).parent / "docs"

# Load all .txt files
documents = []

for file_path in sorted(DOCS_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")

    documents.append({
        "id": file_path.stem,
        "text": text
    })

print(f"Loaded {len(documents)} documents")

for doc in documents:
    print(doc["id"])
    
    
# Create one chunk per document
chunks = []

for doc in documents:
    chunks.append({
        "id": doc["id"],
        "text": doc["text"]
    })

print(f"Created {len(chunks)} chunks")

for chunk in chunks:
    print(chunk["id"])    
    
# Generate embeddings for all chunks
texts = [chunk["text"] for chunk in chunks]

embeddings = embedding_model.encode(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")

# Store chunks and embeddings in ChromaDB
import chromadb

chroma_path = Path(__file__).parent / "chroma_db"

client = chromadb.PersistentClient(path=str(chroma_path))

collection = client.get_or_create_collection(
    name="zepto_policies"
)

collection.upsert(
    ids=[chunk["id"] for chunk in chunks],
    documents=[chunk["text"] for chunk in chunks],
    embeddings=embeddings.tolist()
)

print(f"Stored {collection.count()} chunks in ChromaDB") 

# Test ChromaDB retrieval
query = "How long does Zepto delivery take?"
query_embedding = embedding_model.encode([query])[0]

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=1
)

print("\nQuery test:")
print("Query:", query)
print("Retrieved document:", results["ids"][0][0])
print("Retrieved text:", results["documents"][0][0])